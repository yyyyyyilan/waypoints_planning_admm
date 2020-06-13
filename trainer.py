import argparse
import os
import sys
import random
import csv
from datetime import datetime
import math
import time

from environment_v4 import Env
from agent import Agent
from generate_trajectory_try_2 import traj_mwpts
from utils import *

import pdb

class Trainer(object):
    """waypoints planning trainer is defined here
    """
    def __init__(self, agent, env, args):
        self.args = args
        self.agent = agent
        self.env = env
        # non-Linear epsilon decay
        epsilon_final = args.epsilon_min
        epsilon_start = args.epsilon
        epsilon_decay = args.epsilon_decay
        if args.enable_epsilon:
            self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
                -1. * frame_idx / epsilon_decay)
        else:
            self.epsilon_by_frame = lambda frame_idx: 0.0

    def train(self):
        """train the model
        """
        timestamp = datetime.now()
        time_str = timestamp.strftime("%H_%M_%S")
        loss_reward_filepath = os.path.join('.', self.args.mode, '_loss_reward_{}.csv'.format(time_str))
        if os.path.exists(loss_reward_filepath):
            os.remove(loss_reward_filepath)
        lr = self.agent.lr
        episode = 0
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
            self.num_obst = random.randint(5, 10)
            self.env.reset(self.num_obst)
            state_curt = self.env.get_state()
            if self.args.thrust:
                # Initialize uav configuration
                velocity_curt = np.array((0, 0, 0.25))
                acceler_curt = np.array((0, 0, 0))
                gerk_curt = np.array((0, 0, 0))

            while (not is_done) and (steps <= self.args.max_steps):
                epsilon = self.epsilon_by_frame(episode)
                action_curt = self.agent.act(state_curt, epsilon=epsilon)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                state_next = self.env.get_state()
                #calculate force reward
                if self.args.thrust:
                    start_pos = state_curt[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                    end_pos = state_next[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                    start_end = hstack((np.array([start_pos]).T, np.array([end_pos]).T))
                    t = array([0, 6])
                    path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                            traj_mwpts(t, start_end, np.array([velocity_curt]).T, \
                            np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                    velocity_curt = velocity_next
                    acceler_curt = acceler_next
                    gerk_curt = gerk_next
                    force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / self.args.grid_size
                    reward_curt -= force_reward
                self.agent.buffer.add((state_curt, action_curt, reward_curt, state_next, is_goal))
                state_curt = state_next
                episode_reward += reward_curt
                rewards.append(reward_curt)
                loss = 0.0
                if self.agent.buffer.size() >= self.args.batch_size:
                    loss = self.agent.learning()
                    loss_list.append(loss)
                steps += 1
            loss_avg = sum(loss_list) / max(len(loss_list), 1)
            # update target model weights
            if episode % self.args.target_update == 0:
                self.agent.update_target()

            print('loss_avg: ', loss_avg)
            print('episode: {0:05d}, step: {1:03d}, reward: {2:.04f}, num_obst: {3:01d}, is_goal: {4}, start: {5}, target: {6}'.format(
                episode,
                steps - 1,
                episode_reward,
                num_obst,
                is_goal,
                self.env.objs_info['drone_pos_start'],
                self.env.objs_info['goal']
            ))
            if episode % 1000 == 0:
                print('actions: \n', actions)

            # learning decay
            if episode % 5000 == 0:
                lr *= 0.8
                self.agent.reset_optimizer(lr)

            # plot reward and loss
            with open(loss_reward_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([loss_avg, episode_reward, num_obst, int(is_goal)])
                
            # save trained model weigths
            if episode % self.args.save_epochs == 0:
                save_dic = {
                    'args' : self.args,
                    'episode' : episode,
                    'state_dict' : self.agent.model.state_dict()
                }
                save_path = os.path.join(self.args.save_weights_dir, 'pretrained')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(save_dic, os.path.join(save_path, 'saved_weights_{}.pth.tar'.format(self.mode)))

    def eval(self):
        """evaluate pretrained model and admm pruned model
           Args:
                self.args.load_pretrained: True -> evaluate pretrained model
                self.args.load_admm_pruned: True -> evaluate admm pruned model
        """
        episode = 0
        if self.args.load_admm_pruned:
            self.agent.load_admm_model()
        
        success = 0
        admm_success = 0
        batch_time = AverageMeter()
        batch_steps = AverageMeter()
        admm_batch_time = AverageMeter()
        admm_batch_steps = AverageMeter()
        while True:
            episode += 1                                         
            # Intialize environment
            self.num_obst = random.randint(5, 10)
            self.env.reset(self.num_obst)
            state_curt = self.env.get_state()
            if self.args.thrust:
                # Initialize uav configuration
                velocity_curt = np.array((0, 0, 0.25))
                acceler_curt = np.array((0, 0, 0))
                gerk_curt = np.array((0, 0, 0))
            # copy env for admm and load admm pruned weights
            if self.args.load_admm_pruned:
                admm_env = Env(self.args)
                admm_env.copy(self.env)

            #evaluate pretrained model performance
            if self.args.load_pretrained:
                episode_reward = 0
                is_done = False
                is_goal = False
                steps = 0
                num_obst = 0 
                actions = []            
                rewards = []  
                self.agent.load_admm = False
                
                while (not is_done) and (steps <= self.args.max_steps):
                    end_time = time.time()
                    action_curt = self.agent.act(state_curt, epsilon=0.0)
                    eval_time = end = time.time() - end_time
                    batch_time.update(eval_time)
                    actions.append(action_curt)
                    reward_curt, is_done, reward_info = self.env.step(action_curt)
                    num_obst += int(reward_info['is_obst'])
                    if reward_info['is_goal']:
                        is_goal = True
                        if num_obst == 0:
                            success = success + 1
                    state_next = self.env.get_state()
                    #calculate force reward
                    if self.args.thrust:
                        start_pos = state_curt[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                        end_pos = state_next[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                        start_end = hstack((np.array([start_pos]).T, np.array([end_pos]).T))
                        t = array([0, 6])
                        path, f, norm_f, velocity_next, acceler_next, gerk_next = \
                                traj_mwpts(t, start_end, np.array([velocity_curt]).T, \
                                np.array([acceler_curt]).T, np.array([gerk_curt]).T)
                        velocity_curt = velocity_next
                        acceler_curt = acceler_next
                        gerk_curt = gerk_next
                        force_reward = 1 / (1 + math.exp(-1 * np.sum(norm_f)/norm_f.shape[1])) / self.args.grid_size
                        reward_curt -= force_reward
                    state_curt = state_next
                    episode_reward += reward_curt
                    rewards.append(reward_curt)
                    steps += 1     
                # print(actions)
                batch_steps.update(steps)
                print('[Pretrained]  [{0:05d}] step: [{1:03d}][{batch_steps.avg:5.2f}]  reward: {2:.04f}  is_goal: {3}  obs: {4}  time: {batch_time.val:.6f} [{batch_time.avg:.6f}]'.format(
                    episode,
                    steps-1,
                    episode_reward,
                    is_goal,
                    num_obst,
                    batch_steps=batch_steps,
                    batch_time=batch_time
                ))
                if episode % 100 == 0:
                    print('Pretrained model: ')
                    print('\tsuccess ratio: {0:.02f}%'.format(float(success / episode * 100.0)))

            #evaluate admm pruned model performance
            if self.args.load_admm_pruned:
                self.agent.load_admm = True
                admm_episode_reward = 0
                admm_is_done = False
                admm_is_goal = False
                admm_steps = 0
                admm_num_obst = 0
                admm_actions = []
                admm_rewards = []
                admm_state_curt = admm_env.get_state()
                if self.args.thrust:
                    # Initialize uav configuration
                    admm_velocity_curt = np.array((0, 0, 0.25))
                    admm_acceler_curt = np.array((0, 0, 0))
                    admm_gerk_curt = np.array((0, 0, 0))
                
                while (not admm_is_done) and (admm_steps <= self.args.max_steps):
                    end_time = time.time()
                    admm_action_curt = self.agent.act(admm_state_curt, epsilon=0.0)
                    eval_time = time.time() - end_time
                    admm_batch_time.update(eval_time)
                    admm_actions.append(admm_action_curt)
                    admm_reward_curt, admm_is_done, admm_reward_info = admm_env.step(admm_action_curt)
                    admm_num_obst += int(admm_reward_info['is_obst'])
                    if admm_reward_info['is_goal']:
                        admm_is_goal = True
                        if admm_num_obst == 0:
                            admm_success = admm_success + 1
                    admm_state_next = admm_env.get_state()
                    #calculate force reward
                    if self.args.thrust:
                        admm_start_pos = admm_state_curt[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                        admm_end_pos = admm_state_next[1] * 10 * self.args.grid_resolution + self.args.grid_resolution / 2 
                        admm_start_end = hstack((np.array([admm_start_pos]).T, np.array([admm_end_pos]).T))
                        t = array([0, 6])
                        admm_path, admm_f, admm_norm_f, admm_velocity_next, admm_acceler_next, admm_gerk_next = \
                                traj_mwpts(t, admm_start_end, np.array([admm_velocity_curt]).T, \
                                np.array([admm_acceler_curt]).T, np.array([admm_gerk_curt]).T)
                        admm_velocity_curt = admm_velocity_next
                        admm_acceler_curt = admm_acceler_next
                        admm_gerk_curt = admm_gerk_next
                        admm_force_reward = 1 / (1 + math.exp(-1 * np.sum(admm_norm_f)/admm_norm_f.shape[1])) / self.args.grid_size
                        admm_reward_curt -= admm_force_reward
                    admm_state_curt = admm_state_next
                    admm_episode_reward += admm_reward_curt
                    admm_rewards.append(admm_reward_curt)
                    admm_steps += 1
                # print(admm_actions)
                admm_batch_steps.update(admm_steps)
                print('[ADMM_Pruned] [{0:05d}] step: [{1:03d}][{admm_batch_steps.avg:5.2f}]  reward: {2:.04f}  is_goal: {3}  obs: {4}  time: {admm_batch_time.val:.6f} [{admm_batch_time.avg:.6f}]'.format(
                    episode,
                    admm_steps-1,
                    admm_episode_reward,
                    admm_is_goal,
                    admm_num_obst,
                    admm_batch_steps=admm_batch_steps,
                    admm_batch_time=admm_batch_time
                ))
                if episode % 100 == 0:
                    print('ADMM pruned model: ')
                    print('\tsuccess ratio: {0:.02f}%'.format(float(admm_success / episode * 100.0)))
                    pdb.set_trace()
            print('\n')