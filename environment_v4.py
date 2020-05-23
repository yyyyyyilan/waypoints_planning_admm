import copy 
import numpy as np 
import random 

import pdb 


OBSTACLE_REWARD = -0.1
GOAL_REWARD = 1.0
DIST_REWARD = 0.1
DRONE_POSITION = 0.1


class Env(object):
    def __init__(self, args):
        self.mode = args.mode
        self.action_size = args.action_dim
        self.grid_size = args.grid_size
        self.num_obst = args.num_obst
        self.state_dim = args.state_dim

        # initialize objs_info
        # objs_info = {
        #     "obst_list": list of nparray,
        #     "goal": nparray,
        #     "drone_pos": nparray
        # }
        self.reset(self.num_obst)

    def reset(self, num_obst):
        '''Reset environment including: 
            obstacle positions, goal position and drone position
        '''
        self.num_obst = num_obst
        self.objs_info = self._reset_objs()

    def copy(self, env):
        self.num_obst = env.num_obst
        self.state_dim = env.state_dim
        self.objs_info = env.objs_info.copy()

    def get_state(self):
        """ Return the state that encoded by current |objs_info|.
        """
        state = None
        if self.mode == "linear":
            state = self._encoding_2_state_linear()
        elif self.mode == "conv":
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
        drone_pos_curt = self.objs_info['drone_pos'].copy()
        if self.action_size == 26:
            drone_pos_next, outbound = self._move_26(drone_pos_curt, action)
        elif self.action_size == 6:
            drone_pos_next, outbound = self._move_6(drone_pos_curt, action)
        self.objs_info['drone_pos'] = drone_pos_next
        reward_info = self._calculate_reward(
            drone_pos_curt, drone_pos_next, outbound)
        assert reward_info['reward'] is not None
        if reward_info['is_goal']:# or reward_info['is_obst'] or outbound:
            return reward_info['reward'], True, reward_info
        else:
            return reward_info['reward'], False, reward_info

    def _calculate_reward(self, drone_pos_curt, drone_pos_next, outbound):
        reward_info = {
            'is_goal': False,
            'is_obst': False,
            'reward': None
        }
        if outbound:
            reward_info['is_obst'] = True
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
        temp_reward = self._calculate_projection_reward(drone_pos_curt, drone_pos_next) 
        #                 + self._calculate_distance_reward(drone_pos_curt, drone_pos_next)
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
                    / float(self.grid_size)
            state_placeholder[writer_idx:writer_idx+3] = temp_dist_normalize
            state_placeholder[writer_idx+3] = OBSTACLE_REWARD
            writer_idx += 4
        assert writer_idx + 4 <= self.state_dim, \
            "Not enough space left for state_placeholder."
        goal_dist_normalized = \
            (self.objs_info['goal'] - self.objs_info['drone_pos']) / float(self.grid_size)
        drone_pos_normalized = self.objs_info['drone_pos'] / float(self.grid_size)
        # state_placeholder[-3:] = drone_pos_normalized
        # state_placeholder[-7:-4] = goal_dist_normalized
        # state_placeholder[-4] = GOAL_REWARD
        state_placeholder[-4:-1] = goal_dist_normalized
        state_placeholder[-1] = GOAL_REWARD
        return state_placeholder

    def _encoding_2_state_conv(self):
        state_placeholder = np.zeros(
            (self.grid_size, self.grid_size, self.grid_size)).astype(np.float32)
        for temp_obst in self.objs_info['obst_list']:
            state_placeholder[
                int(temp_obst[0]), int(temp_obst[1]), int(temp_obst[2])] = OBSTACLE_REWARD
        state_placeholder[
            int(self.objs_info['goal'][0]), 
            int(self.objs_info['goal'][1]), 
            int(self.objs_info['goal'][2])] = GOAL_REWARD
        state_placeholder[
            int(self.objs_info['drone_pos'][0]), 
            int(self.objs_info['drone_pos'][1]), 
            int(self.objs_info['drone_pos'][2])] = DRONE_POSITION
        return (state_placeholder, self.objs_info['drone_pos'] / float(self.grid_size))

    def _reset_objs(self):
        obst_list = []
        while len(obst_list) < self.num_obst:
            temp_obs = np.random.randint(self.grid_size, size=3)
            if not self._array_in_list(obst_list, temp_obs):
                obst_list.append(temp_obs.astype(np.float32))
        goal = None
        while goal is None:
            temp_goal = np.random.randint(self.grid_size, size=3)
            if not self._array_in_list(obst_list, temp_goal):
                goal = temp_goal
        drone_pos = None
        while drone_pos is None:
            temp_drone_pos = np.random.randint(self.grid_size, size=3)
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

    def _array_in_list(self, input_list, input_array):
        for temp_array in input_list:
            if np.array_equal(temp_array, input_array):
                return True
        return False

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
            if temp_next_pos_element < 0 or temp_next_pos_element >= self.grid_size:
                return drone_pos_curt, True
        return temp_next_pos, False

    def _move_6(self, drone_pos_curt, action):
        assert action < 6, "This is 6 steps moving map."
        act_vec = None
        if action == 0:
            act_vec = np.array([-1, 0, 0])
        elif action == 1:
            act_vec = np.array([1, 0, 0])
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
            if temp_next_pos_element < 0 or temp_next_pos_element >= self.grid_size:
                return drone_pos_curt, True
        return temp_next_pos, False