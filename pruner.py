import argparse
import numpy as np
import os
import sys
import math
import random
import time
import shutil

from environment_v4 import Env
from replaybuffer import ReplayBuffer
from agent import Agent
import admm
from utils import *

import pdb


class Pruner(object):
    """admm pruner is defined here
    """
    def __init__(self, agent, env, args):
        self.args = args
        self.agent = agent
        self.env = env
        self.scheduler = None
        if args.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.agent.model_optim, T_max=args.epochs * args.max-episode, eta_min=4e-08)
        elif args.lr_scheduler == 'default':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.agent.model_optim, step_size=max(1, int(args.epochs * 0.2)), gamma=0.5)
        else:
            raise Exception('unknown lr scheduler')
        if args.warmup:
            self.scheduler = GradualWarmupScheduler(self.agent.model_optim, multiplier=args.lr / args.warmup_lr, 
        								   total_iter=args.warmup_epochs * args.max-episode, after_scheduler=self.scheduler)
        # bag of tricks set-ups
        args.smooth = args.smooth_eps > 0.0
        args.mixup = args.alpha > 0.0
        self.optimizer_init_lr = args.warmup_lr if args.warmup else args.lr

    def pruning(self):
        """admm_prune first, then masked_retrain
        """
        if (self.args.admm and self.args.masked_retrain):
            raise ValueError('cannot do both masked retrain and admm')
        print('General config:')
        for k, v in sorted(vars(self.args).items()):
            print('\t{}: {}'.format(k, v))
        # multi-rho admm train
        if self.args.admm:
            self.admm_prune(self.args.rho, self.agent.model_optim, self.scheduler)
        # masked retrain
        if self.args.masked_retrain:                
            self.admm_masked_retrain(self.args.rho, self.agent.model_optim, self.scheduler)

    def admm_prune(self, initial_rho, optimizer, scheduler):
        """admm prune training
        """
        for i in range(self.args.rho_num):
            current_rho = initial_rho * 10 ** i
            # load pretrained model / resume
            if i == 0 and not self.args.resume:
                pre_weight_path = os.path.join(self.args.save_weights_dir, 'pretrained', 'saved_weights_{}.pth.tar'.format(self.args.mode))
            elif i > 0 and not self.args.resume:
                pre_weight_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}.pth.tar'.format(self.args.mode, current_rho / 10))
            else:
                pre_weight_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}.pth.tar'.format(self.args.mode, current_rho))
            if os.path.isfile(pre_weight_path):
                print("=> loading checkpoint '{}'".format(pre_weight_path))
                checkpoint = torch.load(pre_weight_path)
            else:
                raise ValueError('Weight path does not exist.')              
            self.load_multi_gpu(self.agent.model, checkpoint, optimizer, first=(i == 0 and not self.args.resume))
            self.agent.model.cuda()

            start_epoch = 1
            best_success_rate = 0.
            best_epoch = 0
            if self.args.resume:
                start_epoch = checkpoint['epoch'] + 1
                try:
                    checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
                    best_epoch = checkpoint['epoch']
                    best_success_rate = checkpoint['success_rate']
                except:
                    pass
            ADMM = admm.ADMM(self.agent.model, file_name=os.path.join(self.args.config_file + self.args.mode + '.yaml'), rho=current_rho)
            # initialize Z variable
            admm.admm_initialization(self.args, ADMM=ADMM, model=self.agent.model)  
            if i == 0:
                print('Prune config:')
                for k, v in ADMM.prune_cfg.items():
                    print('\t{}: {}'.format(k, v))
                print('')
                shutil.copy(os.path.join(self.args.config_file + self.args.mode + '.yaml'), \
                    os.path.join(self.args.save_weights_dir, self.args.config_file + self.args.mode + '.yaml'))

            save_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}.pth.tar'.format(self.args.mode, current_rho))
            for epoch in range(start_epoch, self.args.epochs + 1):
                print('current rho: {}'.format(current_rho))
                # admm training 
                self.train(ADMM, optimizer, scheduler, epoch)
                # admm evaluation
                eval_success_rate = self.eval(self.agent.model)

                is_best = eval_success_rate > best_success_rate
                save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': self.agent.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'success_rate': eval_success_rate,
                },
                is_best, save_path)
                if is_best:
                    best_success_rate = eval_success_rate
                    best_epoch = epoch
                print('[Eval]  Best success_rate: {:.3f}%  Best epoch: {}'.format(best_success_rate, best_epoch))
                print('')
                if self.args.sparsity_type != 'blkcir' and ((epoch - 1) % self.args.admm_epochs == 0 or epoch == self.args.epochs):
                    print('Weight < 1e-4:')
                    for layer in ADMM.prune_cfg.keys():
                        weight = self.agent.model.state_dict()[layer]
                        zeros = len((abs(weight)<1e-4).nonzero())
                        weight_size = torch.prod(torch.tensor(weight.shape))
                        print('   {}: {}/{} = {:.4f}'.format(layer, zeros, weight_size, float(zeros)/float(weight_size)))
                    print('')
            # save best
            os.rename(save_path.replace('.pth.tar', '_best.pth.tar'), \
                save_path.replace('.pth.tar', '_epoch-{}_best-{:.3f}.pth.tar'.format(best_epoch, best_success_rate)))

    def admm_masked_retrain(self, initial_rho, optimizer, scheduler):
        """masked retrain
           output corresponding sparse network
        """
        if not self.args.resume:
            load_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}.pth.tar'.format(self.args.mode, initial_rho * 10 ** (self.args.rho_num - 1)))
            print("=> loading checkpoint '{}'".format(load_path))
        else:
            load_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}rhos.pth.tar'.format(self.args.mode, self.args.rho_num))
        
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
        else:
            exit('Checkpoint does not exist.')
        self.load_multi_gpu(self.agent.model, checkpoint, optimizer, first=True)
        self.agent.model.cuda()

        start_epoch = 1
        success_rate_list = [0.]
        best_epoch = 0
        if self.args.resume:
            start_epoch = checkpoint['epoch'] + 1
            try:
                checkpoint = torch.load(load_path.replace('.pt', '_best.pt'), map_location='cpu')
                best_epoch = checkpoint['epoch']
                best_success_rate = checkpoint['success_rate']
            except:
                pass
        
        # restore scheduler
        for epoch in range(1, start_epoch):
            for _ in range(self.args.admm_batch_num):
                scheduler.step()
        ADMM = admm.ADMM(self.agent.model, file_name=os.path.join(self.args.save_weights_dir, self.args.config_file + self.args.mode + '.yaml'), rho=initial_rho)
        print('Prune config:')
        for k, v in ADMM.prune_cfg.items():
            print('\t{}: {}'.format(k, v))
        print('')

        # admm hard prune
        admm.hard_prune(self.args, ADMM, self.agent.model)
        epoch_loss_dict = {}
        save_path = os.path.join(self.args.save_weights_dir, self.args.mode, 'saved_weights_{}_{}rhos.pth.tar'.format(self.args.mode, self.args.rho_num))
        for epoch in range(start_epoch, self.args.epochs + 1):
            idx_loss_dict = self.train(ADMM, optimizer, scheduler, epoch)
            eval_success_rate = self.eval(self.agent.model)
            best_success_rate = max(success_rate_list)
            is_best = eval_success_rate > best_success_rate

            save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': self.agent.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'success_rate': eval_success_rate,
            },
            is_best, save_path)
            if is_best:
                best_success_rate = eval_success_rate
                best_epoch = epoch
            print('[Eval]  Best success_rate: {:.3f}%  Best epoch: {}\n'.format(best_success_rate, best_epoch))
            epoch_loss_dict[epoch] = idx_loss_dict
            success_rate_list.append(eval_success_rate)
        # save best
        os.rename(save_path.replace('.pt', '_best.pt'), \
            save_path.replace('.pt', '_epoch-{}_top1-{:.3f}.pt'.format(best_epoch, best_success_rate)))

    def train(self, ADMM, optimizer, scheduler, epoch):
        """admm training
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # percentage of successfullly reaching target
        success = AverageMeter()
        idx_loss_dict = {}

        # switch to train mode
        self.agent.model.train()
        if self.args.masked_retrain:
            print('full acc re-train masking')
        elif self.args.combine_progressive:
            print('progressive admm-train/re-train masking')
        if self.args.masked_retrain or self.args.combine_progressive:
            masks = {}
            for name, W in self.agent.model.named_parameters():
                weight = W.detach()
                non_zeros = weight != 0 
                zero_mask = non_zeros.type(torch.float32)
                masks[name] = zero_mask
        end = time.time()
        epoch_start_time = time.time()

        for i in range(self.args.admm_batch_num):
            # measure data loading time
            data_time.update(time.time() - end)
            if self.args.admm:
                admm.admm_adjust_learning_rate(optimizer, epoch, self.args)
            else: 
                scheduler.step()
            
            # generate data in train_buffer
            data_buffer, load_times = self.data_loader()
            train_buffer = data_buffer.sample(data_buffer.size())
            (states_curt, action_curt, rewards_curt, states_next, is_dones) = \
                self.agent._stack_to_numpy(train_buffer)

            if isinstance(states_curt, tuple):
                states_curt_var = []
                for temp in states_curt:
                    states_curt_var.append(
                        torch.tensor(temp, dtype=torch.float).cuda())
                states_curt_var = tuple(states_curt_var)
            else:
                states_curt_var = torch.tensor(states_curt, dtype=torch.float).cuda()
            action_curt_var = torch.tensor(action_curt, dtype=torch.long).cuda()
            rewards_curt_var = torch.tensor(rewards_curt, dtype=torch.float).cuda()
            if isinstance(states_next, tuple):
                states_next_var = []
                for temp in states_next:
                    states_next_var.append(
                        torch.tensor(temp, dtype=torch.float).cuda())
                states_next_var = tuple(states_next_var)
            else:
                states_next_var = torch.tensor(states_next, dtype=torch.float).cuda()
            is_dones_var = torch.tensor(is_dones, dtype=torch.float).cuda()
            logits_curt_var = self.agent.model(states_curt_var)
            logits_next_var = self.agent.model(states_next_var)
            next_q_value = logits_next_var.max(1)[0]
            q_value = logits_curt_var.gather(1, action_curt_var.unsqueeze(1)).squeeze(1)
            expected_q_value = rewards_curt_var + self.args.gamma * next_q_value * (1 - is_dones_var)

            ce_loss = (q_value - expected_q_value.detach()).pow(2).mean()
            if self.args.admm:
                # update Z and U variables
                admm.z_u_update(self.args, ADMM, self.agent.model, epoch, i)  
                # append admm losss
                ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(self.args, ADMM, self.agent.model, ce_loss)  

            # measure success rate and record loss
            train_success_rate = float(is_dones.sum() / load_times * 100.0)
            losses.update(ce_loss.item(), data_buffer.size())
            success.update(train_success_rate, load_times * 100.0)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if self.args.admm:
                mixed_loss.backward()
            else:
                ce_loss.backward()
            if self.args.masked_retrain or self.args.combine_progressive:
                with torch.no_grad():
                    for name, W in self.agent.model.named_parameters():
                        if name in masks:
                            W.grad *= masks[name]
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.args.log_interval == 0:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                print('({0}) lr [{1:.6f}]  '
                    'Epoch [{2}][{3:3d}/{4}]  '
                    'Status [admm-{5}][retrain-{6}]  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'Success_rate {success.val:7.3f}% ({success.avg:7.3f}%)'
                    .format(self.args.optimizer, current_lr,
                    epoch, i, self.args.admm_batch_num, self.args.admm, self.args.masked_retrain, batch_time=batch_time, loss=losses, success=success))
            if i % 10 == 0:
                idx_loss_dict[i] = losses.avg
        print('[Train] Loss {:.4f}   Success_rate {:.3f}%   Time {}'.format(
            losses.avg, success.avg, int(time.time() - epoch_start_time)))
        return idx_loss_dict

    def eval(self, model):
        """admm evaluation
        """
        self.agent.model.eval()
        losses = AverageMeter()
        success = 0
        epoch_start_time = time.time()
        # generate data to test
        for idx in range(self.args.admm_test_epoch):
            episode_reward = 0
            is_done = False
            is_goal = False
            steps = 0
            num_obst = 0
            # Intialize environment
            epoch_num_obst = random.randint(0, 10)
            self.env.reset(epoch_num_obst)
            state_curt = self.env.get_state()
            while (not is_done) and (steps <= self.args.max_steps):
                action_curt = self.agent.act(state_curt, epsilon=0.0)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                    if num_obst == 0:
                        success = success + 1
                state_next = self.env.get_state()
                state_curt = state_next
                episode_reward += reward_curt
                steps += 1
            # print('episode: {0:05d}, step: {1:03d}, reward: {2:.04f}, num_obst: {3:01d}, is_goal: {4}, start: {5}, target: {6}'.format(
            #     idx,
            #     steps - 1,
            #     episode_reward,
            #     num_obst,
            #     is_goal,
            #     self.env.objs_info['drone_pos_start'],
            #     self.env.objs_info['goal']
            # ))
        return float(success / self.args.admm_test_epoch * 100.0)

    def load_multi_gpu(self, model, checkpoint, optimizer, first=False):
        """ baseline model for pruning, pruned model for retrain
        """
        try:
            state_dict = checkpoint['state_dict']
            if not first:
                optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            state_dict = checkpoint
        try:
            model.load_state_dict(state_dict)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                newkey = 'module.' + key
                new_state_dict[newkey] = value
            model.load_state_dict(new_state_dict)

    def data_loader(self):
        """ genetate admm training data and save in agent buffer
        """
        episode = 0
        self.agent.buffer = ReplayBuffer(self.args.buffer_size)
        for idx in range(int(self.args.buffer_size / self.args.max_steps)):
            episode += 1
            episode_reward = 0
            is_done = False
            is_goal = False
            steps = 0
            num_obst = 0
            # Intialize environment
            num_obst = random.randint(0, 10)
            self.env.reset(num_obst)
            state_curt = self.env.get_state()
            while (not is_done) and (steps <= self.args.max_steps):
                action_curt = self.agent.act(state_curt, epsilon=0.0)
                reward_curt, is_done, reward_info = self.env.step(action_curt)
                num_obst += int(reward_info['is_obst'])
                if reward_info['is_goal']:
                    is_goal = True
                state_next = self.env.get_state()
                self.agent.buffer.add((state_curt, action_curt, reward_curt, state_next, is_done))
                state_curt = state_next
                episode_reward += reward_curt
                steps += 1
        return self.agent.buffer, int(self.args.buffer_size / self.args.max_steps)
