import argparse
import os
import sys
import random
import csv
from datetime import datetime
import math

from environment_v4 import Env
from agent import Agent
from utils import *

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
            steps = 0
            num_obst = 0
            actions = []
            loss_list = []
            rewards = []
            # Intialize environment
            self.num_obst = random.randint(5, 10)
            self.env.reset(self.num_obst)
            state_curt = self.env.get_state()

            while (not is_done) and (steps <= self.args.max_steps):
                epsilon = self.epsilon_by_frame(episode)
                action_curt = self.agent.act(state_curt, epsilon=epsilon)
                actions.append(action_curt)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                state_next = self.env.get_state()
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

        while True:
            episode += 1                                         
            # Intialize environment
            self.num_obst = random.randint(5, 10)
            self.env.reset(self.num_obst)
            state_curt = self.env.get_state()
            # copy env for admm and load admm pruned weights
            if self.args.load_admm_pruned:
                admm_env = Env(self.args)
                admm_env.copy(self.env)

            batch_time = AverageMeter()
            admm_batch_time = AverageMeter()

            #evaluate pretrained model performance
            if self.args.load_pretrained:
                episode_reward = 0
                is_done = False
                is_goal = False
                steps = 0
                num_obst = 0 
                actions = []            
                rewards = []  
                success = 0
                self.agent.load_admm = False
                while (not is_done) and (steps <= self.args.max_steps):
                    end_time = time.time()
                    action_curt = self.agent.act(state_curt, epsilon=0.0)
                    eval_time = time.time() - end_time
                    batch_time.update(eval_time)
                    actions.append(action_curt)
                    reward_curt, is_done, reward_info = self.env.step(action_curt)
                    num_obst += int(reward_info['is_obst'])
                    if reward_info['is_goal']:
                        is_goal = True
                        if num_obst == 0:
                            success += 1
                    state_next = self.env.get_state()
                    state_curt = state_next
                    episode_reward += reward_curt
                    rewards.append(reward_curt)
                    steps += 1     
                # print(actions)
                print('[Pretrained]  [{0:05d}] step: {1:03d}, reward: {2:.04f}, num_obst: {3:03d}, is_goal: {4}, time: {5:03f}, start: {6}, target: {7}'.format(
                    episode,
                    steps - 1,
                    episode_reward,
                    num_obst,
                    is_goal,
                    batch_time.avg,
                    self.env.objs_info['drone_pos_start'],
                    self.env.objs_info['goal']
                ))

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
                admm_success = 0
                admm_state_curt = admm_env.get_state()
                
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
                            admm_success += 1
                    admm_state_next = admm_env.get_state()
                    admm_state_curt = admm_state_next
                    admm_episode_reward += admm_reward_curt
                    admm_rewards.append(admm_reward_curt)
                    admm_steps += 1
                # print(admm_actions)
                print('[ADMM_Pruned] [{0:05d}] step: {1:03d}, reward: {2:.04f}, num_obst: {3:03d}, is_goal: {4}, time: {5:03f}, start: {6}, target: {7}'.format(
                    episode,
                    admm_steps - 1,
                    admm_episode_reward,
                    admm_num_obst,
                    admm_is_goal,
                    admm_batch_time.avg,
                    admm_env.objs_info['drone_pos_start'],
                    admm_env.objs_info['goal']
                ))
            print('\n')