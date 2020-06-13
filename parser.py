import argparse

""" arguments for waypoints planning 
"""
parser = argparse.ArgumentParser(description='PyTorch waypoints planning training and pruning')
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--mode", default='linear', choices=['linear', 'conv'], type=str)
parser.add_argument("--batch-size", default=200, type=int)
parser.add_argument("--optimizer", default='adam', choices=['adam', 'sgd'], type=str)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--grid-size", default=10, type=int)
parser.add_argument("--grid-resolution", default=10, type=int)
parser.add_argument("--num-obst", default=5, type=int, help='Maximum number: 50')
parser.add_argument("--state-dim", default=204, type=int)
parser.add_argument("--action-dim", default=26, choices=[6, 26], type=int)
parser.add_argument("--eval", action='store_true')
parser.add_argument("--buffer-size", default=2000, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--enable-epsilon", action='store_true')
parser.add_argument("--epsilon", default=1, type=int)
parser.add_argument("--epsilon-min", default=0.01, type=float)
parser.add_argument("--epsilon-decay", default=500, type=int)
parser.add_argument("--max-steps", default=200, type=int)
parser.add_argument("--save-epochs", default=1000, type=int)
parser.add_argument("--save-weights-dir", default='./saved_weights', type=str)
parser.add_argument("--load-pretrained", action='store_true')
parser.add_argument("--target-update", default=30, type=int)
parser.add_argument("--thrust", action='store_true', help='take thrust reward into consideration while training')

""" arguments for pruning 
"""
parser.add_argument("--pruning", action='store_true')
parser.add_argument("--admm", action='store_true')
parser.add_argument("--masked-retrain", action='store_true')
parser.add_argument("--max-episode", default=10000, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument('--lr-scheduler', default='default', choices=['cosine', 'default'], type=str)
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--warmup-lr', default=0.0001, metavar='M', type=float)
parser.add_argument('--warmup-epochs', default=5, metavar='M', type=int)
parser.add_argument('--smooth', action='store_true', help='lable smooth')
parser.add_argument('--smooth-eps', default=0.0, metavar='M', type=float,
                    help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
parser.add_argument('--resume',  action='store_true',
                    help='resume from last epoch if model exists')
parser.add_argument('--rho', default = 0.0001, type=float, 
                    help='define rho for ADMM')
parser.add_argument('--rho-num', default = 4, type=int, 
                    help='define how many rhos for ADMM training')
parser.add_argument('--config-file', default='waypoint_planning_', type=str, 
                    help='config file name')
parser.add_argument('--combine-progressive', action='store_true',
                    help='for filter pruning after column pruning')
parser.add_argument('--mixup', action='store_true', 
                    help='ce mixup')
parser.add_argument('--alpha', default=0.0, metavar='M', type=float, 
                    help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
parser.add_argument('--admm-epochs', default=10, metavar='N', type=int, 
                    help='number of interval epochs to update admm (default: 10)')
parser.add_argument('--admm-batch-num', default=100, type=int)
parser.add_argument('--log-interval', default=10, metavar='N', type=int, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--admm-test-epoch', default=100, type=int)
parser.add_argument('--sparsity-type', default='filter', type=str, 
                    help='define sparsity_type: [filter, channel, column]')
parser.add_argument("--load-admm-pruned", action='store_true',
                    help='load admm pruned model weights')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='for multi-gpu training')