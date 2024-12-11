import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Normal

class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean_actions, std_actions, old_actions):
        distribution = Normal(mean_actions, std_actions)
        actions_with_exploration = distribution.sample()

        if old_actions is None:
            log_actions = distribution.log_prob(actions_with_exploration)
        else:
            log_actions = distribution.log_prob(old_actions)

        return distribution.mean, actions_with_exploration, log_actions, distribution.entropy()


class Policy(nn.Module):
    continuous = True

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

    def forward(self, x):
        return x
    
    def act(self, state):
        return 

    def train(self):
        return

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
