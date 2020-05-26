from __future__ import print_function
import torch
import torch.nn as nn
from numpy import linalg as LA
import yaml
from utils import *


class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file
        """
        if not isinstance(config, str):
            raise Exception('filename must be a str')
        with open(config, 'r') as stream:
            try:
                raw_dict = yaml.full_load(stream)
                if 'prune_ratios' in raw_dict:
                    self.prune_cfg = raw_dict['prune_ratios']
                for k, v in self.prune_cfg.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.named_parameters():
                    if name not in self.prune_cfg:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z

            except yaml.YAMLError as exc:
                print(exc)


def weight_pruning(args, weight_in, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero
    """
    weight = weight_in.cpu().detach().numpy()  # convert cpu tensor to numpy
    percent = prune_ratio * 100

    if (args.sparsity_type == 'filter'):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm <= percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        # weight2d[weight2d < 1e-40] = 0
        weight = weight.reshape(shape)
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        #return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sparsity_type == 'channel'):
        shape = weight.shape
        #print('channel pruning...', weight.shape)
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, percent)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
        for i in range(weight3d.shape[1]):
            expand_above_threshold[:, i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        #return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == 'column'):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm <= percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def hard_prune(args, ADMM, model):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda
    """
    import os
    import numpy as np
    index_dir = os.path.join(args.save_weights_dir, args.mode, '{}_{}_index'.format(args.mode, args.sparsity_type))
    os.makedirs(index_dir, exist_ok=True)
    
    print('hard pruning\n')
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_cfg:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if args.admm or args.masked_retrain:
            retained_index, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_cfg[name])  # get sparse model in cuda
            np.save(os.path.join(index_dir, name.split('module.')[-1]), retained_index.cpu())
        W.data = cuda_pruned_weights  # replace the data field in variable


def admm_initialization(args, ADMM, model):
    """admm Z valiable initialization
    """
    if not args.admm:
        return    
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_cfg:
            continue
        if args.admm:
            _, updated_Z = weight_pruning(args, W, ADMM.prune_cfg[name])  # Z(k+1) = W(k+1)+U(k)
        ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, epoch, batch_idx):
    """Z, U variables update
    """
    if not args.admm:
        return
    if epoch != 1 and (epoch - 1) % args.admm_epochs == 0 and batch_idx == 0:
        for (name, W) in model.named_parameters():
            if name not in ADMM.prune_cfg:
                continue
            Z_prev = None
            W_detach = W.detach()
            U_detach = ADMM.ADMM_U[name].detach()
            ADMM.ADMM_Z[name] = W_detach + U_detach  # Z(k+1) = W(k+1) + U[k]
            if args.admm:
                _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name], ADMM.prune_cfg[name])
            ADMM.ADMM_Z[name] = updated_Z
            Z_detach = ADMM.ADMM_Z[name].detach()
            ADMM.ADMM_U[name] = W_detach - Z_detach + U_detach  # U(k+1) = W(k+1) - Z(k+1) + U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    """
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss
    """
    admm_loss = {}

    if args.admm:
        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_cfg:
                continue
            # admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm((W + ADMM.ADMM_U[name])[ADMM.ADMM_Z[name]==0], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)
    """
    admm_epoch = args.admm_epochs
    lr = None
    if (epoch - 1) % admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = (epoch - 1) % admm_epoch
        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.
        lr = args.lr * (0.5 ** (admm_epoch_offset // admm_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        