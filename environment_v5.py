from abc import ABC
from abc import abstractmethod
import copy 
import importlib
import logging
import random 
import numpy as np 

import pdb 

OBSTACLE_REWARD = -2.0
GOAL_REWARD = 10.0
DIST_REWARD = 0.1
SELF_EGO = (2, 2, 2)

kOutOfBound_Encode = -1.0
kObstacle_Encode = -1.0
kEgoPosition_Encode = 0.5
kGoalPosition_Encode = 1

'''
Sep 2:
1. add specific obstacle generation
2. random place obstacles with specific types
'''

class Env(object):
    def __init__(self, args):
        if DIST_REWARD == 0:
            logging.warn("DIST_REWARD set to 0.")
        self.model_type = args.mode
        self.action_dim = args.action_dim
        self.env_size = args.env_size
        self.num_obst = args.num_obst
        self.state_dim = args.state_dim
        self.sensing_range = (args.sensing_range, args.sensing_range, args.sensing_range)

        move_func_string = '_move_{}'.format(self.action_dim)
        self._move_func = getattr(self, move_func_string)

        self._obst_generation_mode = args.obst_generation_mode
        print('Obstacle generating mode: {}'.format(
            self._obst_generation_mode))
        assert self._obst_generation_mode in \
            ['voxel_random', 'voxel_constrain', 'plane_random', 'test', 'random']
        self._select_obst_generator()

        # initialize objs_info
        # objs_info = {
        #     "obst_list": list of nparray,
        #     "goal": nparray,
        #     "drone_pos": nparray
        # }
        # self.reset(self.num_obst)

    def reset(self, num_obst):
        '''Reset environment including: 
            obstacle positions, goal position and drone position
        '''
        if self._obst_generation_mode == "random":
            self._select_obst_generator()
        self.num_obst = num_obst
        self.objs_info = self._reset_objs()

    def get_state(self):
        """ Return the state that encoded by current |objs_info|.
        """
        state = None
        if self.model_type == "linear":
            state = self._encoding_2_state_linear()
        elif self.model_type == "conv":
            state = self._encoding_2_state_conv()
        assert state is not None
        return state

    def step(self, action):
        """ Move drone by action.
        ==========
        Parameters:
        action : int
            action index for moving drone.

        Return:
        reward : float
            calculated reward for action.
        is_done : bool
            if current episode is finished.
        reward_info : dict
            if trajectory ends at goal position.
            {
                'is_goal': False,
                'is_obst': False,
                'reward': None
            }
        """
        assert action < self.action_dim
        drone_pos_curt = self.objs_info['drone_pos'].copy()
        drone_pos_next, outbound = self._move_func(drone_pos_curt, action)
        self.objs_info['drone_pos'] = drone_pos_next
        reward_info = self._calculate_reward(
            drone_pos_curt, drone_pos_next, outbound, action)
        self.prev_action = action

        assert reward_info['reward'] is not None
        # only if UAV reached destination, is_goal is true
        if reward_info['is_goal']:# or reward_info['is_obst'] or outbound:
            return reward_info['reward'], True, reward_info
        else:
            return reward_info['reward'], False, reward_info

    def _select_obst_generator(self):
        self._obst_generator = None
        gen_map = {
            'voxel_random': 'ObstGeneratorVoxelRandom',
            'voxel_constrain': 'ObstGeneratorVoxelConstrain',
            'plane_random': 'ObstGeneratorPlaneRandom',
            'test': 'ObstGeneratorUT'
        }
        if self._obst_generation_mode == "random":
            keys_list = list(gen_map.keys())
            keys_list.remove('test')
            curt_obst_generation_mode = np.random.choice(keys_list)
            logging.info('curt_obst_generation_mode: {}'.format(curt_obst_generation_mode))
        else:
            curt_obst_generation_mode = self._obst_generation_mode

        module = importlib.import_module('environment_v5')
        obst_gen_class = getattr(module, gen_map[curt_obst_generation_mode])
        if curt_obst_generation_mode == 'voxel_random':
            self._obst_generator = obst_gen_class('voxel_random', self.env_size, self.num_obst)
        elif curt_obst_generation_mode == 'voxel_constrain':
            self._obst_generator = obst_gen_class('voxel_constrain', self.env_size, self.num_obst)
        elif curt_obst_generation_mode == 'plane_random':
            self._obst_generator = obst_gen_class('plane_random', self.env_size)
        elif curt_obst_generation_mode == 'test':
            self._obst_generator = obst_gen_class('plane_random', self.env_size)

        assert self._obst_generator is not None

    def _calculate_reward(self, drone_pos_curt, drone_pos_next, outbound, action):
        reward_info = {
            'is_goal': False,
            'is_obst': False,
            'is_bound': False,
            'reward': None
        }
        if outbound:
            reward_info['is_bound'] = True
            reward_info['reward'] = OBSTACLE_REWARD
            return reward_info
        if np.array_equal(self.objs_info['goal'], drone_pos_next):
            reward_info['is_goal'] = True
            reward_info['reward'] = GOAL_REWARD
            return reward_info
        for temp_obst in self.objs_info['obst_list']:
            if np.array_equal(temp_obst, drone_pos_next):
                reward_info['is_obst'] = True
                reward_info['reward'] = OBSTACLE_REWARD
                return reward_info
        # temp_reward = self._calculate_projection_reward(drone_pos_curt, drone_pos_next) \
        #                 + self._calculate_distance_reward(drone_pos_curt, drone_pos_next)
        temp_reward = self._calculate_projection_reward(drone_pos_curt, drone_pos_next)
        # temp_reward = self._calculate_distance_reward(drone_pos_curt, drone_pos_next)

        reward_info['reward'] = temp_reward
        return reward_info

    def _calculate_projection_reward(self, drone_pos_curt, drone_pos_next):
        drone_pos_curt = drone_pos_curt.astype(np.float32)
        drone_pos_next = drone_pos_next.astype(np.float32)
        if np.array_equal(drone_pos_curt, drone_pos_next):
            return 0.0
        goal_dirction = self.objs_info['goal'] - drone_pos_curt 
        goal_dirction_normalize = goal_dirction / np.linalg.norm(goal_dirction)
        move_direction = drone_pos_next - drone_pos_curt 
        move_direction_normalize = move_direction / np.linalg.norm(move_direction)
        projection_normalized = np.dot(goal_dirction_normalize, move_direction_normalize)
        assert projection_normalized >= -1 and projection_normalized <= 1
        return projection_normalized * DIST_REWARD

    def _calculate_distance_reward(self, drone_pos_curt, drone_pos_next):
        drone_pos_curt = drone_pos_curt.astype(np.float32)
        drone_pos_next = drone_pos_next.astype(np.float32)
        dist_curt = ((drone_pos_curt[0] - self.objs_info['goal'][0])**2 + \
            (drone_pos_curt[1] - self.objs_info['goal'][1])**2 + \
                (drone_pos_curt[2] - self.objs_info['goal'][2])**2)**0.5
        dist_next = ((drone_pos_next[0] - self.objs_info['goal'][0])**2 + \
            (drone_pos_next[1] - self.objs_info['goal'][1])**2 + \
                (drone_pos_next[2] - self.objs_info['goal'][2])**2)**0.5
        # dist_reward = float(dist_next < dist_curt) * DIST_REWARD
        dist_ori = ((self.objs_info['drone_pos_start'][0] - self.objs_info['goal'][0])**2 + \
                (self.objs_info['drone_pos_start'][1] - self.objs_info['goal'][1])**2 + \
                    (self.objs_info['drone_pos_start'][2] - self.objs_info['goal'][2])**2)**0.5
        dist_reward = (dist_curt - dist_next) / dist_ori * DIST_REWARD
        return dist_reward

    def _encoding_2_state_linear(self):
        state_placeholder = np.zeros(self.state_dim).astype(np.float32)
        writer_idx = 0
        for temp_obst in self.objs_info['obst_list']:
            temp_dist_normalize = \
                (temp_obst - self.objs_info['drone_pos']) \
                    / float(self.env_size)
            state_placeholder[writer_idx:writer_idx+3] = temp_dist_normalize
            state_placeholder[writer_idx+3] = OBSTACLE_REWARD
            writer_idx += 4
        assert writer_idx + 4 <= self.state_dim, \
            "Not enough space left for state_placeholder."
        goal_dist_normalized = \
            (self.objs_info['goal'] - self.objs_info['drone_pos']) / float(self.env_size)
        drone_pos_normalized = self.objs_info['drone_pos'] / float(self.env_size)
        # state_placeholder[-3:] = drone_pos_normalized
        # state_placeholder[-7:-4] = goal_dist_normalized
        # state_placeholder[-4] = GOAL_REWARD
        state_placeholder[-4:-1] = goal_dist_normalized
        state_placeholder[-1] = GOAL_REWARD
        return state_placeholder

    def _encoding_2_state_conv(self):
        state_placeholder = np.zeros(
            self.sensing_range).astype(np.float32)
        # Encode Ego
        state_placeholder[SELF_EGO] = kEgoPosition_Encode

        # Encode bounds
        local_lower_bound = self._global_to_grid((-1, -1, -1))
        if local_lower_bound[0] >= 0:
            state_placeholder[0:local_lower_bound[0]+1, :, :] = kOutOfBound_Encode
        if local_lower_bound[1] >= 0:
            state_placeholder[:, 0:local_lower_bound[1]+1, :] = kOutOfBound_Encode
        if local_lower_bound[2] >= 0:
            state_placeholder[:, :, 0:local_lower_bound[2]+1] = kOutOfBound_Encode
        local_upper_bound = self._global_to_grid((self.env_size, self.env_size, self.env_size))
        offset = (np.array(self.sensing_range) - 1) - local_upper_bound
        if offset[0] >= 0:
            state_placeholder[self.sensing_range[0]-1-offset[0]:self.sensing_range[0], :, :] = kOutOfBound_Encode
        if offset[1] >= 0:
            state_placeholder[:, self.sensing_range[1]-1-offset[1]:self.sensing_range[1], :] = kOutOfBound_Encode
        if offset[2] >= 0:
            state_placeholder[:, :, self.sensing_range[2]-1-offset[2]:self.sensing_range[2]] = kOutOfBound_Encode

        # Encode Obstacle
        for obst in self.objs_info['obst_list']:
            obst_grid = self._global_to_grid(obst)
            in_bound = True
            for idx in range(3):
                if obst_grid[idx] < 0 or obst_grid[idx] >= self.sensing_range[idx]:
                    in_bound = False
                    break
            if in_bound:
                state_placeholder[obst_grid[0], obst_grid[1], obst_grid[2]] = kObstacle_Encode

        # Encode Goal
        goal_grid = self._global_to_grid(self.objs_info['goal'])
        for idx in range(3):
            goal_grid[idx] = min(max(goal_grid[idx], 0), self.sensing_range[idx] - 1)
        state_placeholder[goal_grid[0], goal_grid[1], goal_grid[2]] = kGoalPosition_Encode

        return (
            state_placeholder, 
            self._global_to_grid(self.objs_info['drone_pos']) / float(self.env_size), 
            self._global_to_grid(self.objs_info['goal']) / float(self.env_size))

    def _global_to_local(self, global_position):
        return (np.array(global_position) - self.objs_info['drone_pos']).astype(np.int)

    def _local_to_grid(self, local_position):
        return (local_position + SELF_EGO).astype(np.int)

    def _global_to_grid(self, global_position):
        return self._local_to_grid(self._global_to_local(global_position))

    def _encode_position(self, global_position):
        for idx in range(3):
            if global_position[idx] < 0 or global_position[idx] >= self.env_size:
                # Out of bound
                return kOutOfBound_Encode
        for temp_obst in self.objs_info['obst_list']:
            if np.array_equal(global_position, np.array(temp_obst).astype(np.int)):
                # Obstacle
                return kObstacle_Encode
        if np.array_equal(global_position, self.objs_info['goal'].astype(np.int)):
            # Obstacle
            return kGoalPosition_Encode
        if np.array_equal(global_position, self.objs_info['drone_pos'].astype(np.int)):
            # Ego drone
            return kEgoPosition_Encode
        return 0

    def _reset_objs(self):
        objs_info = self._obst_generator()
        return objs_info

    def _move_26(self, drone_pos_curt, action):
        assert action < 26, "This is 26 steps moving map."
        act_vec = None
        if action == 0:
            act_vec = np.array([-1, -1, 1])
        elif action == 1:
            act_vec = np.array([-1, 0, 1])
        elif action == 2:
            act_vec = np.array([-1, 1, 1])
        elif action == 3:
            act_vec = np.array([0, -1, 1])
        elif action == 4:
            act_vec = np.array([0, 0, 1])
        elif action == 5:
            act_vec = np.array([0, 1, 1])
        elif action == 6:
            act_vec = np.array([1, -1, 1])
        elif action == 7:
            act_vec = np.array([1, 0, 1])
        elif action == 8:
            act_vec = np.array([1, 1, 1])
        elif action == 9:
            act_vec = np.array([-1, -1, -1])
        elif action == 10:
            act_vec = np.array([-1, 0, -1])
        elif action == 11:
            act_vec = np.array([-1, 1, -1])
        elif action == 12:
            act_vec = np.array([0, -1, -1])
        elif action == 13:
            act_vec = np.array([0, 0, -1])
        elif action == 14:
            act_vec = np.array([0, 1, -1])
        elif action == 15:
            act_vec = np.array([1, -1, -1])
        elif action == 16:
            act_vec = np.array([1, 0, -1])
        elif action == 17:
            act_vec = np.array([1, 1, -1])
        elif action == 18:
            act_vec = np.array([-1, -1, 0])
        elif action == 19:
            act_vec = np.array([-1, 0, 0])
        elif action == 20:
            act_vec = np.array([-1, 1, 0])
        elif action == 21:
            act_vec = np.array([0, -1, 0])
        elif action == 22:
            act_vec = np.array([0, 1, 0])
        elif action == 23:
            act_vec = np.array([1, -1, 0])
        elif action == 24:
            act_vec = np.array([1, 0, 0])
        elif action == 25:
            act_vec = np.array([1, 1, 0])
        else:
            raise ValueError("Invalid action {}".format(action))

        temp_next_pos = drone_pos_curt + act_vec
        # check if new position is outbound
        for temp_next_pos_element in temp_next_pos:
            if temp_next_pos_element < 0 or temp_next_pos_element >= self.env_size:
                return drone_pos_curt, True
        return temp_next_pos, False
    
    def _move_6(self, drone_pos_curt, action):
        assert action < 6, "This is 6 steps moving map."
        act_vec = None
        if action == 0:
            act_vec = np.array([1, 0, 0])
        elif action == 1:
            act_vec = np.array([-1, 0, 0])
        elif action == 2:
            act_vec = np.array([0, 1, 0])
        elif action == 3:
            act_vec = np.array([0, -1, 0])
        elif action == 4:
            act_vec = np.array([0, 0, 1])
        elif action == 5:
            act_vec = np.array([0, 0, -1])
        else:
            raise ValueError("Invalid action {}".format(action))

        temp_next_pos = drone_pos_curt + act_vec
        # check if new position is outbound
        for temp_next_pos_element in temp_next_pos:
            if temp_next_pos_element < 0 or temp_next_pos_element >= self.env_size:
                return drone_pos_curt, True
        return temp_next_pos, False

    def _is_anti_action_6(self, action, prev_action):
        if prev_action == None:
            return False
        anti_action_dict = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
            4: [5],
            5: [4]
        }
        return action in anti_action_dict[prev_action]


class ObstGenerator(ABC):
    def __init__(self, mode, env_size):
        self._mode = mode
        self._env_size = env_size

    def get_mode(self):
        return self._mode

    def get_env_size(self):
        return self._env_size

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError

    def _array_in_list(self, input_list, input_array):
        for temp_array in input_list:
            if np.array_equal(temp_array, input_array):
                return True
        return False


class ObstGeneratorVoxelRandom(ObstGenerator):
    def __init__(self, mode, env_size, num_obst):
        super(ObstGeneratorVoxelRandom, self).__init__(mode, env_size)
        self._num_obst = num_obst

    def __call__(self, **kwargs):
        obst_list = []
        while len(obst_list) < self._num_obst:
            temp_obs = np.random.randint(self._env_size, size=3)
            if not self._array_in_list(obst_list, temp_obs):
                obst_list.append(temp_obs.astype(np.float32))
        goal = None
        while goal is None:
            temp_goal = np.random.randint(self._env_size, size=3)
            if not self._array_in_list(obst_list, temp_goal):
                goal = temp_goal
        drone_pos = None
        while drone_pos is None:
            temp_drone_pos = np.random.randint(self._env_size, size=3)
            if (not self._array_in_list(obst_list, temp_drone_pos)) \
                and (not np.array_equal(temp_drone_pos, goal)):
                drone_pos = temp_drone_pos
        objs_info = {
            "obst_list": tuple(obst_list),
            "goal": goal.astype(np.float32),
            "drone_pos": drone_pos.astype(np.float32),
            "drone_pos_start": drone_pos.astype(np.float32)
        }
        return objs_info


class ObstGeneratorVoxelConstrain(ObstGenerator):
    def __init__(self, mode, env_size, num_obst):
        super(ObstGeneratorVoxelConstrain, self).__init__(mode, env_size)
        self._num_obst = num_obst

    def __call__(self, **kwargs):
        """ Set obstacles with constraint
        """
        drone_pos = None
        while drone_pos is None:
            drone_pos = np.random.randint(self._env_size, size=3)
        goal = None
        while goal is None:
            temp_goal = np.random.randint(self._env_size, size=3)
            if not np.array_equal(temp_goal, drone_pos) and temp_goal[0] != drone_pos[0]:
                goal = temp_goal
        obs_x_range = [i for i in range(self._env_size)]
        obs_x_range.remove(drone_pos[0])
        obs_x_range.remove(goal[0])
        obst_list = []
        while len(obst_list) < self._num_obst:
            temp_obs_x = random.choice(obs_x_range)
            temp_obs_y = np.random.randint(self._env_size)
            temp_obs_z = np.random.randint(self._env_size)
            temp_obs = np.asarray([temp_obs_x, temp_obs_y, temp_obs_z])
            if not np.array_equal(temp_obs, drone_pos) and \
                    not np.array_equal(temp_obs, goal) and \
                    not self._array_in_list(obst_list, temp_obs):
                obst_list.append(temp_obs.astype(np.float32))    
        objs_info = {
            "obst_list": tuple(obst_list),
            "goal": goal.astype(np.float32),
            "drone_pos": drone_pos.astype(np.float32),
            "drone_pos_start": drone_pos.astype(np.float32)
        }
        return objs_info


class ObstGeneratorPlaneRandom(ObstGenerator):
    def __init__(self, mode, env_size):
        super(ObstGeneratorPlaneRandom, self).__init__(mode, env_size)

    def __call__(self, **kwargs):
        wall_normal_axis = np.random.randint(3)
        obst_list, normal_coord = self._generate_wall_obst_list(wall_normal_axis)
        goal = None
        while goal is None:
            goal_normal_idx = np.random.randint(self._env_size)
            if goal_normal_idx != normal_coord:
                goal_along_idx = np.random.randint(self._env_size, size=2)
                goal = np.insert(goal_along_idx, wall_normal_axis, [goal_normal_idx])

        goal_normal_direction = goal[wall_normal_axis] - normal_coord
        drone_along_idx = np.random.randint(self._env_size, size=2)
        if goal_normal_direction > 0:
            drone_normal_idx = np.random.randint(normal_coord)
        else:
            drone_normal_idx = normal_coord + 1 + np.random.randint(self._env_size - normal_coord - 1)
        drone_pos = np.insert(drone_along_idx, wall_normal_axis, [drone_normal_idx])

        objs_info = {
            "obst_list": tuple(obst_list),
            "goal": goal.astype(np.float32),
            "drone_pos": drone_pos.astype(np.float32),
            "drone_pos_start": drone_pos.astype(np.float32)
        }
        return objs_info

    def _generate_wall_obst_list(self, wall_normal_axis):
        wall_size = self._env_size // 2 + \
            np.random.randint(self._env_size // 2 - 1) + 1 #range: [env_size/2+1, env_size-1]
        # wall_size = np.random.randint(self._env_size // 2 - 1) + 1 #range: [1, env_size/2-1]
        wall_start_coord_normal = 1 + \
            np.random.randint(self._env_size - 2) #range: [1, env_size-2]
        wall_start_coord_along = np.random.randint(self._env_size - wall_size, size=2)

        linear_0 = np.linspace(
            wall_start_coord_along[0], wall_start_coord_along[0] + wall_size - 1, wall_size)
        linear_1 = np.linspace(
            wall_start_coord_along[1], wall_start_coord_along[1] + wall_size - 1, wall_size)
        mesh_0, mesh_1 = np.meshgrid(linear_0, linear_1)
        obsts = np.concatenate(
            (np.expand_dims(mesh_0.reshape(mesh_0.size), -1), 
             np.expand_dims(mesh_1.reshape(mesh_1.size), -1)), -1)
        obst_list = np.insert(
            obsts, wall_normal_axis, 
            wall_start_coord_normal * np.ones(wall_size * wall_size), 
            axis=1).astype(np.int).tolist()

        return obst_list, wall_start_coord_normal
        

class ObstGeneratorUT(ObstGenerator):
    def __init__(self, mode, env_size):
        super(ObstGeneratorUT, self).__init__(mode, env_size)

    def __call__(self, **kwargs):
        objs_info = {
            "obst_list": [[3, 3, 3]],
            "goal": np.array([2, 2, 2]).astype(np.float32),
            "drone_pos": np.array([5, 5, 5]).astype(np.float32),
            "drone_pos_start": np.array([5, 5, 5]).astype(np.float32)
        }
        return objs_info