import numpy as np
import os
import sys
import math
import random

class ReplayBuffer(object):
    """ReplayBuffer is defined here
    """
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