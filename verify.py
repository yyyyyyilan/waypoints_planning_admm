import os
import torch
from torch import nn
import argparse
import numpy as np

import pdb

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

def main(args):
    pretrained_model = LinearNetwork(args.state_dim, args.action_dim).cuda()
    admm_model = LinearNetwork(args.state_dim, args.action_dim).cuda()
    pretrained_weight_path = os.path.join(('saved_weights'), 'pretrained', 'saved_weights_{}.pth.tar'.format(args.mode))
    admm_weight_path = os.path.join(('saved_weights'), args.mode, 'saved_weights_linear.pth.tar')

    if os.path.isfile(pretrained_weight_path):
        print("=> loading checkpoint '{}'".format(pretrained_weight_path))
        checkpoint = torch.load(pretrained_weight_path)
        pretrained_model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('Weight path does not exist.')

    if os.path.isfile(admm_weight_path):
        print("=> loading checkpoint '{}'".format(admm_weight_path))
        checkpoint = torch.load(admm_weight_path)
        admm_model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('Weight path does not exist.')
    
    pretrained = pretrained_model.fc1.weight
    admm = admm_model.fc1.weight
    total_num = pretrained.detach().cpu().numpy().size
    before = np.where(pretrained.detach().cpu().numpy() == 0)[0].shape[0]
    zeros_after_prune = np.where(admm.detach().cpu().numpy() == 0)[0].shape[0]
    print('Zero weights ratio: {0:.3f}'.format(float(zeros_after_prune / total_num)))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch waypoints planning training and pruning weights verification')
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--mode", default='linear', choices=['linear', 'conv'], type=str)
    parser.add_argument("--grid-size", default=10, type=int)
    parser.add_argument("--num-obst", default=5, type=int)
    parser.add_argument("--state-dim", default=204, type=int, help='Maximum number: 50')
    parser.add_argument("--action-dim", default=26, choices=[6, 26], type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)