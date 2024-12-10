import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

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
