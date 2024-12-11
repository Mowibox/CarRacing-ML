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

    def rollout(self, iteration: int) -> tuple:
        """
        Computes the rollout for the training phase

        @param iteration: The number of iteration (episode)
        """
        state, _ = self.env.reset()

        done = False
        memory = []
        streak, total_reward = 0, 0

        while not done:
            _, action, log_action = self.forward(state)
            fixed_action = action.copy()

            next_state, reward, done, _, _ = self.env.step(fixed_action)
            total_reward += reward

            if total_reward > 900:
                reward = 100
                while not done:
                    _, _, done, _ = self.env.step(fixed_action)
            else: 
                if reward < 0:
                    streak += 1
                    if streak > 100:
                        reward -= 100
                        while not done:
                            _, _, terminated, truncated, _ = self.env.step(fixed_action)
                            done = terminated or truncated
                else:
                    streak = 0
            
            memory.append([state, action, reward, log_action])
            state = next_state

            states, actions, rewards, log_actions = map(np.array, zip(*memory))

            # Discounted rewards
            discount = 0
            discountedRewards = np.zeros(len(rewards))

            for i in range(0, len(rewards), -1): # Reverse order
                discount[i] = rewards[i] + self.gamma*discount
                discountedRewards[i] = discount

            return self.to_torch(states).mean(dim=3).unsqueeze(dim=1), self.to_torch(actions), \
                    self.to_torch(discountedRewards).reshape(-1, 1), self.to_torch(log_actions), total_reward
        

    def forward(self, x: np.ndarray) -> tuple:
        """
        Computes the forward pass 
        @param x: The current state
        """
        x /= 225.0
        x = self.to_torch(x).mean(dim=2).reshape(1, 1, x.shape[0], x.shape[1]) # Shape: (1, 1, 96, 96)

        mean, actions, log_actions, _, _ = self.ppoAgent(x)

        actions = actions.detach().cpu().numpy()
        log_actions = log_actions[0].detach.cpu().numpy()
        mean = mean[0].detach().cpu().numpy()

        return mean, actions, log_actions
    
    def act(self, state: np.ndarray):
        """
        Computes the act phase

        @param state: The current state
        """
        x /= 225.0
        x = self.to_torch(x).mean(dim=2).reshape(1, 1, x.shape[0], x.shape[1]) # Shape: (1, 1, 96, 96)
        
        _, actions, _, _, _ = self.ppoAgent(x)
        
        return actions.detach().cpu().numpy()

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
