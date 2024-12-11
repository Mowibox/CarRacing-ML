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
    

class PPO(nn.Module):
    def __init__(self, output_size):
        super().__init__(PPO, self).__init__()

        # CNN
        self.conv2D_0 = nn.Conv2d(1, 256, kernel_size=3, stride=1)
        self.conv2D_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1)
        self.conv2D_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        self.conv2D_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv2D_4 = nn.Conv2d(32, 16, kernel_size=3, stride=1)

        # Actor
        self.action_mean = nn.Linear(16, output_size)
        self.action_std = nn.Linear(16, output_size)

        # Critic
        self.critic_output = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.gaussian = Gaussian()

        # Orthogonal weights initialization
        for layer in [self.conv2D_0, self.conv2D_1, self.conv2D_2, self.conv2D_3,\
                      self.conv2D_4, self.action_mean, self.critic_output]:
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, old_actions=None) -> tuple:
        """
        Computes the forward pass through the PPO
        @param x: The current state
        @param old_actions: The old-actions for log-probability calculation
        """
        x = self.relu(self.conv2D_0(x))
        x = self.relu(self.conv2D_1(x))
        x = self.relu(self.conv2D_2(x))
        x = self.relu(self.conv2D_3(x))
        x = self.relu(self.conv2D_4(x))

        x = x.view(x.shape[0], -1)

        # Actor
        x_action_mean = self.action_mean(x)
        x_action_std = self.relu(self.action_std(x))

        mean, actions, log_actions, entropy = self.gaussian(x_action_mean, x_action_std, old_actions)
        
        # Critic
        x_value = self.critic_output(x)

        return mean, actions, log_actions, entropy, x_value
     

class Policy(nn.Module):
    continuous = True
    env = gym.make('CarRacing-v2', continuous=continuous, render_mode="rgb_array")

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

        # PPO Hyperparameters
        self.gamma = 0.98
        self.epsilon = 0.2
        self.value_factor = 0.5
        self.entropy_factor = 0.005

        self.episodes = 10000
        self.updates_per_episode = 5

        self.ppoAgent = PPO(output_size=3).to(device)

    def compute_losses(self, 
                       states: torch.Tensor, 
                       actions: torch.Tensor, 
                       returns: torch.Tensor, 
                       log_actions: torch.Tensor, 
                       advantages: torch.Tensor) -> tuple:
        """
        Computes the losses for the PPO training phases

        @param states: The observed states
        @param actions: The actions
        @param returns: The discounted rewards
        @param log_actions: The actions log-probabilities 
        @param advantages: The advantages estimates
        """
        _, _, log_actions_new, entropy, values = self.ppoAgent(states, actions)

        ratios = torch.exp(log_actions_new - log_actions)

        policy_loss = torch.min(ratios*advantages, torch.clip(ratios, 1-self.epsilon, 1+self.epsilon)*advantages)
        policy_loss = -torch.mean(policy_loss)

        value_loss = ((advantages + values)**2).mean()
        entropy_loss = -torch.mean(entropy)
        
        return policy_loss, value_loss, entropy_loss

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
