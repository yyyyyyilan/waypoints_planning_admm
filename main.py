import argparse
import os
import sys

from parser import parser
from environment_v4 import Env
from agent import Agent
from trainer import Trainer
from pruner import Pruner

import pdb

def main(args):
    env = Env(args)
    agent = Agent(args)
    if not args.pruning:
        trainer = Trainer(agent, env, args)
        if not args.eval:
            trainer.train()
        else:
            trainer.eval()
    else:
        pruner = Pruner(agent, env, args)
        pruner.pruning()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)