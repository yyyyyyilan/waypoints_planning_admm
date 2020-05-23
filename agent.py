import argparse
import numpy as np
import os
import sys
import random
import torch
from replaybuffer import ReplayBuffer
from network import LinearNetwork, ConvNetwork

class Agent(object):
    """Agent is defined here
    """
    def __init__(self, args):
        self.is_training = not args.eval
        self.mode = args.mode
        self.load_pretrained = args.load_pretrained
        assert args.buffer_size >= args.batch_size
        self.batch_size = args.batch_size
        self.buffer = ReplayBuffer(args.buffer_size)
        self.grid_size = args.grid_size
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.gamma = args.gamma
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.save_weights_dir = args.save_weights_dir
        self.weight_decay = args.weight_decay
        self.model = None
        self.target_model = None
        self.load_admm = False
        self.admm_model = None
        if self.mode == "linear":
            self.model = LinearNetwork(self.state_dim, self.action_dim).cuda()
            self.target_model = LinearNetwork(self.state_dim, self.action_dim).cuda()
        elif self.mode == "conv":
            self.model = ConvNetwork(self.grid_size, self.action_dim).cuda()
            self.target_model = ConvNetwork(self.grid_size, self.action_dim).cuda()
        assert self.model is not None
        assert self.target_model is not None
        self.update_target()
        if self.load_pretrained:
            pre_weight_path = os.path.join(self.save_weights_dir, 'pretrained', 'saved_weights_{}.pth.tar'.format(self.mode))
            if os.path.isfile(pre_weight_path):
                print("=> loading checkpoint '{}'".format(pre_weight_path))
                checkpoint = torch.load(pre_weight_path)
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError('Weight path does not exist.')
        self.model.train()
        self.target_model.eval()
        self.reset_optimizer(self.lr)
    
    def print_model_weight(self):
        """print model weights
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    
    def reset_optimizer(self, lr):
        """reset optimizer learning rate.
        """
        if self.optimizer == 'adam':
            self.model_optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.optimizer == 'sgd':
            self.model_optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.5, weight_decay=self.weight_decay)
        return
    
    def update_target(self):
        """uodate target network weigths
        """
        print("=> updating target network weights...")
        self.target_model.load_state_dict(self.model.state_dict())
    
    def load_admm_model(self):
        """load admm pruned model
        """
        if self.mode == "linear":
            self.admm_model = LinearNetwork(self.state_dim, self.action_dim).cuda()
        if self.mode == "conv":
            self.admm_model = ConvNetwork(self.grid_size, self.action_dim).cuda()
        assert self.admm_model is not None
        admm_weight_path = os.path.join(self.save_weights_dir, self.mode, 'saved_weights_{}.pth.tar'.format(self.mode))
        if os.path.isfile(admm_weight_path):
            print("=> loading ADMM checkpoint '{}'".format(admm_weight_path))
            checkpoint = torch.load(admm_weight_path)
            self.admm_model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError('ADMM weight path does not exist.')
        self.admm_model.eval()

    def act(self, state, epsilon=0.0):
        """Output an action.
        """
        if not self.is_training:
            epsilon = 0.0
        if random.random() > epsilon:
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
            if not self.load_admm:
                self.model.eval()
                logits = self.model(state_var).detach().cpu().numpy()
            else:
                self.admm_model.eval()
                logits = self.admm_model(state_var).detach().cpu().numpy()
            action = np.argmax(logits)
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
                    torch.tensor(temp, dtype=torch.float).cuda())
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
                    torch.tensor(temp, dtype=torch.float).cuda())
            states_next_var = tuple(states_next_var)
        else:
            states_next_var = torch.tensor(
                states_next, dtype=torch.float).cuda()
        is_dones_var = torch.tensor(
            is_dones, dtype=torch.float).cuda()
        if self.is_training: # and not self.load_pretrained:
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
    
    def eval_learning(self):
        """Extract from buffer and eval for one epoch.
        """
        data_list = self.buffer.sample(self.batch_size)
        (states_curt, action_curt, rewards_curt, states_next, is_dones) = \
            self._stack_to_numpy(data_list)
        if isinstance(states_curt, tuple):
            states_curt_var = []
            for temp in states_curt:
                states_curt_var.append(
                    torch.tensor(temp, dtype=torch.float).cuda())
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
        if self.is_training: # and not self.load_pretrained:
            self.model.train()
        else:
            self.model.eval()
        logits_curt_var = self.model(states_curt_var)
        logits_next_var = self.target_model(states_next_var)
        next_q_value = logits_next_var.max(1)[0]
        q_value = logits_curt_var.gather(1, action_curt_var.unsqueeze(1)).squeeze(1)
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
