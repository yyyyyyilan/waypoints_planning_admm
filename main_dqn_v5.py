"""
- environment with unfixed number of obstacles, env.reset(number of obstacles), maximum number == 50
- action-dim: 6 / 26 actions
- agent.reset_optimizer: leraning rate decay
- reward: projection reward
- add target model, update its weights every target_update epochs
- add force reward for each step by envoking traj_mwpts function in generate_trajectory_try_2
"""
import argparse
import csv
from datetime import datetime
import logging
import math
import numpy as np
from numpy import *
import os
import random
import time
import torch
from torch import nn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from environment_v5 import Env
from generate_trajectory_try_2 import traj_mwpts

import pdb

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# np.random.seed(0)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, item):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))

    def size(self):
        return len(self.buffer)


class LinearNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(LinearNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.fc_in = nn.Linear(self.input_size, 64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 256)
        self.fc_out = nn.Linear(256, self.num_actions)
    
    def forward(self, s_input):
        x = self.fc_in(s_input)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x


class ConvNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        """Network structure is defined here
        """
        super(ConvNetwork, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv3d(1, 128, 3, stride=1)
        self.conv2 = nn.Conv3d(128, 256, 3, stride=1)
        self.conv1_1 = nn.Conv3d(1, 256, 5, stride=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(3, 32)
        self.fc3 = nn.Linear(3, 32)
        self.fc4 = nn.Linear(576, 128)
        self.fc_out = nn.Linear(128, self.num_actions)
    
    def forward(self, s_input):
        (state, loc, dest) = s_input
        state = state.unsqueeze(1)
        x_state_fe = self.conv1(state)
        x_state_fe = self.relu(x_state_fe)
        x_state_fe = self.conv2(x_state_fe)
        x_state_fe = self.relu(x_state_fe)
        x_state_fe_1 = self.conv1_1(state)
        x_state_fe_1 = self.relu(x_state_fe_1)
        x_state_fe = x_state_fe + x_state_fe_1
        x_state = self.fc1(x_state_fe.view(state.shape[0], -1))
        x_state = self.relu(x_state)
        x_loc = self.fc2(loc)
        x_loc = self.relu(x_loc)
        x_dest = self.fc3(dest)
        x_dest = self.relu(x_dest)
        x = torch.cat([x_state, x_loc, x_dest], -1)
        x = self.fc4(x)
        x = self.relu(x)
        out = self.fc_out(x)
        return out


class Agent(object):
    def __init__(self, args):
        self.is_training = not args.eval
        self.load_pretrained = args.load_pretrained
        assert args.buffer_size >= args.batch_size
        self.batch_size = args.batch_size
        self.buffer = ReplayBuffer(args.buffer_size)
        self.action_dim = args.action_dim
        self.gamma = args.gamma
        self.lr = args.lr
        self.model = None
        self.target_model = None
        if args.mode == "linear":
            self.model = LinearNetwork(args.state_dim, args.action_dim).cuda()
            self.target_model = LinearNetwork(args.state_dim, args.action_dim).cuda()
        elif args.mode == "conv":
            self.model = ConvNetwork(args.sensing_range, args.action_dim).cuda()
            self.target_model = ConvNetwork(args.sensing_range, args.action_dim).cuda()
        assert self.model is not None
        assert self.target_model is not None
        if args.load_pretrained:
            pre_weight_path = os.path.join(
                args.save_weights_dir, 'saved_weights_{}_yantao.pth.tar'.format(args.mode))
                # args.save_weights_dir, 'saved_weights_{}_10_0.7.pth.tar'.format(args.mode))
                # args.save_weights_dir, 'saved_weights_{}.pth.tar'.format(args.mode))
            if os.path.isfile(pre_weight_path):
                print("=> loading checkpoint '{}'".format(pre_weight_path))
                checkpoint = torch.load(pre_weight_path)
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError('Weight path does not exist.')
        self.update_target()
        self.model.train()
        self.target_model.eval()
        self.reset_optimizer(self.lr)
    
    def print_model_weight(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    
    def reset_optimizer(self, lr):
        """reset optimizer learning rate.
        """
        if args.optimizer == 'admm':
            self.model_optim = torch.optim.Adam(
                self.model.parameters(), lr=lr)
        elif args.optimizer == 'sgd':
            self.model_optim = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.5, 
                weight_decay=args.weight_decay)
        return

    def update_target(self):
        print("=> updating target network weights...")
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon=0.0, prev_is_hit_bound=False, topk_rand=1):
        """Output an action.
        """
        if not self.is_training:
            epsilon = 0.0
        if random.random() >= epsilon:
            if isinstance(state, tuple):
                state_var = []
                for temp in state:
                    state_var.append(
                        torch.tensor(
                            temp, dtype=torch.float).unsqueeze(0).cuda()
                    )
                state_var = tuple(state_var)
            else:
                state_var = torch.tensor(
                    state, dtype=torch.float).unsqueeze(0).cuda()
            # self.model.eval()
            logits = self.model(state_var).detach().cpu().numpy()
            if not prev_is_hit_bound:
                actions_sort = np.argsort(logits[0], -1)
                rand_idx = np.random.randint(topk_rand)
                action = actions_sort[-1 * rand_idx - 1]
            else:
                action = random.randrange(self.action_dim)
        else:
            assert self.is_training == True
            action = random.randrange(self.action_dim)
        return action

    def learning(self):
        """Extract from buffer and train for one epoch.
        """
        data_list = self.buffer.sample(self.batch_size)
        (states_curt, action_curt, rewards_curt, states_next, is_dones) = \
            self._stack_to_numpy(data_list)
        if isinstance(states_curt, tuple):
            states_curt_var = []
            for temp in states_curt:
                states_curt_var.append(
                    torch.tensor(
                        temp, dtype=torch.float).cuda()
                )
            states_curt_var = tuple(states_curt_var)
        else:
            states_curt_var = torch.tensor(
                states_curt, dtype=torch.float).cuda()
        action_curt_var = torch.tensor(
            action_curt, dtype=torch.long).cuda()
        rewards_curt_var = torch.tensor(
            rewards_curt, dtype=torch.float).cuda()
        if isinstance(states_next, tuple):
            states_next_var = []
            for temp in states_next:
                states_next_var.append(
                    torch.tensor(
                        temp, dtype=torch.float).cuda()
                )
            states_next_var = tuple(states_next_var)
        else:
            states_next_var = torch.tensor(
                states_next, dtype=torch.float).cuda()
        is_dones_var = torch.tensor(
            is_dones, dtype=torch.float).cuda()
        # if self.is_training and not self.load_pretrained:
        if self.is_training:
            self.model.train()
        else:
            self.model.eval()
        logits_curt_var = self.model(states_curt_var)
        q_value = logits_curt_var.gather(1, action_curt_var.unsqueeze(1)).squeeze(1)
        logits_next_var = self.target_model(states_next_var)
        next_q_value = logits_next_var.max(1)[0]
        expected_q_value = rewards_curt_var + \
            self.gamma * next_q_value * (1 - is_dones_var)
        
        loss_mse = (q_value - expected_q_value.detach()).pow(2).mean()
        loss_mae = torch.abs(q_value - expected_q_value.detach()).mean()
        loss = torch.max(loss_mse, loss_mae)
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()
        return loss.detach().item()

    def _stack_to_numpy(self, data_list):
        ret = []
        for temp_data in data_list:
            if isinstance(temp_data[0], tuple):
                temp_list = []
                tuple_size = len(temp_data[0])
                for _ in range(tuple_size):
                    temp_list.append([])
                for curt_tup in temp_data:
                    for idx in range(tuple_size):
                        temp_list[idx].append(curt_tup[idx])
                temp_ret_list = []
                for temp in temp_list:
                    temp_ret_list.append(np.array(temp))
                ret.append(tuple(temp_ret_list))
            else:
                temp_np = np.array(temp_data)
                ret.append(temp_np)
        return ret


class Trainer(object):
    def __init__(self, agent, env, args):
        self.args = args
        self.agent = agent
        self.env = env
        self.max_steps = args.max_steps
        self.batch_size = args.batch_size
        self.save_epochs = args.save_epochs
        self.save_weights_dir = args.save_weights_dir
        self.num_obst = args.num_obst
        # non-Linear epsilon decay
        epsilon_final = args.epsilon_min
        epsilon_start = args.epsilon
        epsilon_decay = args.epsilon_decay
        if args.enable_epsilon:
            self.epsilon_by_frame = \
                lambda frame_idx: epsilon_final + \
                    (epsilon_start - epsilon_final) * math.exp(
                        -1. * (frame_idx // epsilon_decay))
        else:
            self.epsilon_by_frame = lambda frame_idx: 0.0

    def train(self):
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H_%M_%S")
        loss_reward_filepath = os.path.join('.', 'loss_reward_{}.csv'.format(time_str))
        if os.path.exists(loss_reward_filepath):
            os.remove(loss_reward_filepath)
        lr = self.agent.lr
        episode = 0
        while True:
            episode += 1
            episode_reward = 0
            is_done = False
            is_goal = False
            prev_is_hit_bound = False
            steps = 0
            num_obst = 0
            num_outbound = 0
            epsilon = self.epsilon_by_frame(episode)
            logging.info('epsilon: {0:.04f}'.format(epsilon))
            actions = []
            loss_list = []

            rewards = []
            # Intialize environment with different number of obstacles
            self.num_obst = random.randint(0, 10)
            self.env.reset(self.num_obst)
            state_curt = self.env.get_state()
            segment  = np.array(state_curt[1] * self.args.env_size)
            velocity_curt = np.array((0, 0, 0.001))
            acceler_curt = np.array((0, 0, 0))
            gerk_curt = np.array((0, 0, 0))
            waypoints = []
            while (not is_done) and (steps <= self.max_steps):
                action_curt = self.agent.act(state_curt, epsilon=epsilon, prev_is_hit_bound=prev_is_hit_bound, topk_rand=2)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                num_outbound += int(reward_info['is_bound'])
                prev_is_hit_bound = reward_info['is_bound']
                if reward_info['is_goal']:
                    is_goal = True
                waypoints.append(list(self.env.objs_info['drone_pos']))
                state_next = self.env.get_state()
                # calculate force reward
                if self.args.thrust_reward:
                    segment = vstack((segment, np.array(state_next[1] * self.args.env_size)))
                    num = segment.shape[0]
                    t = np.asarray([0])
                    for i in range(num - 1):
                        t = hstack((t, 6 * (i + 1)))
                    path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                        traj_mwpts(t, segment.T, np.array([velocity_curt]).T, 
                                   np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                    force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / \
                        self.args.grid_resolution / self.args.env_size
                    reward_curt -= force_reward

                self.agent.buffer.add((state_curt, action_curt, reward_curt, state_next, is_done))
                state_curt = state_next
                episode_reward += reward_curt
                rewards.append(reward_curt)
            #     loss = 0.0
            #     if self.agent.buffer.size() >= self.batch_size:
            #         loss = self.agent.learning()
            #         loss_list.append(loss)
                steps += 1

            if self.agent.buffer.size() >= self.batch_size:
                loss = self.agent.learning()
                loss_list.append(loss)

            loss_avg = sum(loss_list) / max(len(loss_list), 1)
            waypoints.append(list(self.env.objs_info['drone_pos']))
            # plot_env(self.env, waypoints)
            # update target model weights
            if episode % args.target_update == 0:
                self.agent.update_target()

            if int(args.verbose) >= 2:
                print('actions: ', actions)
            logging.info('loss_avg: {0:.04f}'.format(loss_avg))
            
            print('episode: {0:05d}, step: {1:03d}, reward: {2:.04f}, num_obst: {3:01d}, num_outbound: {7:01d}, is_goal: {4}, start: {5}, target: {6}'.format(
                episode,
                steps,
                episode_reward,
                num_obst,
                is_goal,
                self.env.objs_info['drone_pos_start'],
                self.env.objs_info['goal'],
                num_outbound
            ))
            if episode % 100 == 0:
                print('actions: \n', actions)

            # learning decay
            # if episode % 5000 == 0:
            #     lr *= 0.8
            #     self.agent.reset_optimizer(lr)

            # plot reward and loss
            with open(loss_reward_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([loss_avg, episode_reward, num_obst, int(is_goal)])

            if episode % self.save_epochs == 0:
                save_dic = {
                    'args' : args,
                    'episode' : episode,
                    'state_dict' : self.agent.model.state_dict()
                }
                if not os.path.exists(self.save_weights_dir):
                    os.mkdir(self.save_weights_dir)
                torch.save(save_dic, os.path.join(
                    self.save_weights_dir, 'saved_weights_{}_yantao.pth.tar'.format(args.mode)))

    def eval(self):
        episode = 0
        success = 0
        while True:
            episode += 1
            episode_reward = 0
            is_done = False
            is_goal = False
            steps = 0
            num_obst = 0
            actions = []
            loss_list = []

            rewards = []
            # Intialize environment
            obs_num = random.randint(0, 10)
            self.env.reset(obs_num)
            state_curt = self.env.get_state()
            segment  = np.array(state_curt[1] * self.args.env_size)
            velocity_curt = np.array((0, 0, 0.001))
            acceler_curt = np.array((0, 0, 0))
            gerk_curt = np.array((0, 0, 0))

            while (not is_done) and (steps <= self.max_steps):
                epsilon = self.epsilon_by_frame(episode)
                action_curt = self.agent.act(state_curt, epsilon=0.0)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                    success += 1
                state_next = self.env.get_state()
                #calculate force reward
                segment = vstack((segment, np.array(state_next[1] * self.args.env_size)))
                num = segment.shape[0]
                t = np.asarray([0])
                for i in range(num - 1):
                    t = hstack((t, 6 * (i + 1)))
                path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                    traj_mwpts(t, segment.T, np.array([velocity_curt]).T, 
                               np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / \
                    self.args.grid_resolution / self.args.env_size
                reward_curt -= force_reward

                state_curt = state_next
                episode_reward += reward_curt
                rewards.append(reward_curt)
                steps += 1
                
            if episode % 100 == 0:
                print('Evaluating success ratio: {0:.03f}'.format(float(success / episode)))
                break

            print('episode: {0:05d}, step: {1:03d}, reward: {2:.01f}, num_obst: {3:03d}, is_goal: {4}, start: {5}, target: {6}'.format(
                episode,
                steps - 1,
                episode_reward,
                num_obst,
                is_goal,
                self.env.objs_info['drone_pos_start'],
                self.env.objs_info['goal']
            ))
  
def plot_env(env, waypoints):
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlabel('Z', color='k')  
    ax.set_ylabel('Y', color='k')
    ax.set_xlabel('X', color='k')   
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_title('3D animate')
    ax.legend(loc='lower right')
    pdb.set_trace()
    # plot_x, plot_y, plot_z = _trajectory(waypoints)
    plot_x = [item[0] for item in waypoints]
    plot_y = [item[1] for item in waypoints]
    plot_z = [item[2] for item in waypoints]
    ax.plot(plot_x, plot_y, plot_z, 'r-')
    obs_list = env.objs_info['obst_list']
    for temp_obst in obs_list:
        ax.scatter(int(temp_obst[0]), int(temp_obst[1]), int(temp_obst[2]), s=20*4)
    pyplot.show()

    
def main(args):
    env = Env(args)
    agent = Agent(args)
    trainer = Trainer(agent, env, args)
    if not args.eval:
        trainer.train()
    else:
        trainer.eval()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--mode", default='linear', choices=['linear', 'conv'], type=str)
    parser.add_argument("--batch-size", default=200, type=int)
    parser.add_argument("--optimizer", default='admm', choices=['admm', 'sgd'], type=str)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--env-size", default=10, type=int)
    parser.add_argument("--sensing-range", default=5, type=int)
    parser.add_argument("--grid-resolution", default=0.7, type=float)
    parser.add_argument("--num-obst", default=5, type=int)
    parser.add_argument("--state-dim", default=204, type=int, help='Maximum number: 50')
    parser.add_argument("--action-dim", default=26, choices=[6, 26], type=int)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--buffer-size", default=2000, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--enable-epsilon", action='store_true')
    parser.add_argument("--epsilon", default=1.0, type=float)
    parser.add_argument("--epsilon-min", default=0.1, type=float)
    parser.add_argument("--epsilon-decay", default=200, type=int)
    parser.add_argument("--max-steps", default=200, type=int)
    parser.add_argument("--save-epochs", default=1000, type=int)
    parser.add_argument("--save-weights-dir", default='./saved_weights', type=str)
    parser.add_argument("--load-pretrained", action='store_true')
    parser.add_argument("--thrust-reward", action='store_true')
    parser.add_argument("--target-update", default=30, type=int)
    parser.add_argument("--obst-generation-mode", 
                        default="voxel_random", 
                        choices=['voxel_random', 'plane_random', 'voxel_constrain', 'test', 'random'], 
                        type=str)
    parser.add_argument("--verbose", default='2', type=str)

    return parser.parse_args()


def setup_logging(args, log_path=None):
    """Setup logging module
    """
    lvl = {
        '0': logging.ERROR,
        '1': logging.WARN,
        '2': logging.INFO
    }
    
    logging.basicConfig(
        level=lvl[args.verbose],
        filename=log_path) 


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    main(args)