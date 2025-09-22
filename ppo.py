"""
    @file        ppo.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       PPO algorithm implementation 
    @version     1.0
    @date        2024-12-11
    
"""

# Imports 
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import FrameStack
import torch.nn.functional as F
from torch.distributions import Normal



class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean_actions: torch.Tensor, std_actions: torch.Tensor, old_actions: torch.Tensor = None) -> tuple:
        """
        Computes the forward pass for the Gaussian distributiion

        @param mean_actions: The mean of the Gaussian distribution 
        @param std_actions: The standard deviation of the Gaussian distribution 
        @param old_actions: The previous actions 
        """
        distribution = Normal(mean_actions, std_actions)

        if old_actions is None:
            actions = distribution.sample()
            log_actions = distribution.log_prob(actions).sum(dim=-1)
        else:
            actions = old_actions
            log_actions = distribution.log_prob(old_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return distribution.mean, actions, log_actions, entropy
    

class PPO(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        in_channels, H, W = input_size

        # CNN
        self.conv2D_0 = nn.Conv2d(in_channels, 8, kernel_size=4, stride=2)  
        self.conv2D_1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv2D_2 = nn.Conv2d(16, 64, kernel_size=3, stride=2)
        self.conv2D_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2D_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.gaussian = Gaussian()
        with torch.no_grad():

            dummy = torch.zeros(1, in_channels, H, W)
            y = self.forward_conv(dummy)
            conv_out_size = y.view(1, -1).shape[1]

        # Actor
        self.action_mean = nn.Linear(conv_out_size, output_size)
        self.action_std = nn.Linear(conv_out_size, output_size)

        # Critic
        self.critic_output = nn.Linear(conv_out_size, 1)


        # Orthogonal weights initialization
        for layer in [self.conv2D_0, self.conv2D_1, self.conv2D_2, self.conv2D_3,\
                      self.conv2D_4, self.action_mean, self.action_std, self.critic_output]:
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv2D_0(x))
        x = self.relu(self.conv2D_1(x))
        x = self.relu(self.conv2D_2(x))
        x = self.relu(self.conv2D_3(x))
        x = self.relu(self.conv2D_4(x))
        return x

    
    def forward(self, x: torch.Tensor, old_actions=None) -> tuple:
        """
        Computes the forward pass through the PPO
        @param x: The current state
        @param old_actions: The old-actions for log-probability calculation
        """
        x = self.forward_conv(x)
        x = x.view(x.shape[0], -1)

        # Actor
        x_action_mean = self.action_mean(x)
        x_action_std = self.softplus(self.action_std(x)) + 1e-6 # To stay positive! :)

        mean, actions, log_actions, entropy = self.gaussian(x_action_mean, x_action_std, old_actions)
        
        # Critic
        x_value = self.critic_output(x)

        return mean, actions, log_actions, entropy, x_value
     

class Policy(nn.Module):


    def __init__(self, device=torch.device('cpu'), stack_size=8):
        super().__init__()
        self.device = device
        self.stack_size = stack_size
        self.continuous = True

        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode="rgb_array")
        self.env = FrameStack(self.env, self.stack_size)
        obs, _ = self.env.reset()
        stack, H, W, C = obs.shape
        in_channels = stack * C   

        # PPO Hyperparameters
        self.gamma = 0.99
        self.epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

        self.episodes = 10000
        self.updates_per_episode = 5

        self.ppoAgent = PPO(output_size=3, input_size=(in_channels, H, W)).to(device)

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
        actions = actions.detach()
        log_actions = log_actions.detach()
        _, _, log_actions_new, entropy, values = self.ppoAgent(states, actions)
        advantages = advantages.view(-1)
        ratios = torch.exp(log_actions_new - log_actions)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))

        value_loss = F.mse_loss(values, returns)
        entropy_loss = -torch.mean(entropy)
        return policy_loss, value_loss, entropy_loss
    

    def rollout(self) -> tuple:
        """
        Computes the rollout for the training phase
        """
        state, _ = self.env.reset()
        state = np.array(state)

        done = False
        memory = []
        streak, total_reward = 0, 0.0

        while not done:
            mean, action, log_action = self.forward(state)
            fixed_action = action.copy()

            next_state, reward, terminated, truncated, info = self.env.step(fixed_action)
            next_state = np.array(next_state)
            done = terminated or truncated
            total_reward += reward

            if total_reward > 900:
                reward = 100
                memory.append([state, action, reward, log_action])
                break
            else: 
                if reward < 0:
                    streak += 1
                    if streak > 100:
                        reward = -100
                        memory.append([state, action, reward, log_action])
                        break
                else:
                    streak = 0
            
            memory.append([state, action, reward, log_action])
            state = next_state

        states, actions, rewards, log_actions = map(np.array, zip(*memory))

        # Discounted rewards
        discount = 0
        discountedRewards = np.zeros(len(rewards))

        for i in reversed(range(len(rewards))): # Reverse order
            discount = rewards[i] + self.gamma*discount
            discountedRewards[i] = discount

        state_tensors = []
        for s in states: 
            s = s / 255.0
            s = s[:, :-12, 6:-6, :]
            stack_s, Hs, Ws, Cs = s.shape
            s = s.transpose(0, 3, 1, 2).reshape(stack_s * Cs, Hs, Ws)  
            state_tensors.append(s)

        states_t = torch.tensor(np.stack(state_tensors, axis=0), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(discountedRewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        log_actions_t = torch.tensor(log_actions, dtype=torch.float32, device=self.device)


        return states_t, actions_t, returns_t, log_actions_t, total_reward
        

    def forward(self, x: np.ndarray) -> tuple:
        """
        Computes the forward pass 
        @param x: The current state
        """
        x = np.array(x)
        x = x.astype(np.float32) / 255.0  # Data normalization (only once)
        x = x[:, :-12, 6:-6, :]  # crop height and width
        stack, H, W, C = x.shape
        x = x.transpose(0, 3, 1, 2).reshape(stack * C, H, W)  # Shape: (stack*C, H, W)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        mean, actions, log_actions, _, _ = self.ppoAgent(x)

        actions = actions[0].detach().cpu().numpy()
        log_actions = log_actions[0].detach().cpu().numpy()
        mean = mean[0].detach().cpu().numpy()

        return mean, actions, log_actions
    
    
    def act(self, state: np.ndarray):
        """
        Computes the act phase

        @param state: The current state
        """
        state = np.array(state)
        state = state.astype(np.float32) / 255.0
        state = state[:, :-12, 6:-6, :]  # crop
        stack, H, W, C = state.shape
        state = state.transpose(0, 3, 1, 2).reshape(stack * C, H, W)  # Shape: (stack*C, H, W)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        _, actions, _, _, _ = self.ppoAgent(state)
        return actions[0].detach().cpu().numpy()
    

    def train(self):
        """
        Computes the training phase
        """
        # Adam optimizer
        optimizer = torch.optim.Adam(self.ppoAgent.parameters(), lr=1e-4)

        scores = []
        best_score = -float('inf')

        for episode in range(self.episodes):
            with torch.no_grad():
                self.ppoAgent.eval()
                states, actions, returns, log_actions, episode_score = self.rollout()

            scores.append(episode_score)
            print(f"Score at episode {episode}: {scores[-1]:.2f}")
            if episode_score > best_score:
                best_score = episode_score
                self.save() 

            _, _, _, _, values = self.ppoAgent(states)

            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

            self.ppoAgent.train() # Update

            for n_step in range(self.updates_per_episode):
                optimizer.zero_grad()
                policy_loss, value_loss, entropy_loss = self.compute_losses(states, actions, returns, log_actions, advantages)

                loss = policy_loss + self.value_coeff*value_loss + self.entropy_coeff*entropy_loss
                print(f"Loss at step n°{n_step+1}: {loss:.4f}")

                loss.backward() # Backpropagation

                torch.nn.utils.clip_grad_norm_(self.ppoAgent.parameters(), 0.5)

                optimizer.step()
        return
    

    def save(self):
        """
        Saves the model
        """
        torch.save(self.state_dict(), 'modelPPO.pt')


    def load(self):
        """
        Loads the model
        """
        self.load_state_dict(torch.load('modelPPO.pt', map_location=self.device, weights_only=True))


    def to(self, device: torch.device) -> nn.Module:
        """
        Moves the model to the device

        @param device: The target device
        """
        ret = super().to(device)
        ret.device = device
        return ret
    

    def to_torch(self, np_array: np.ndarray) -> torch.tensor:
        """
        Converts a numpy array to a tensor

        @param nparray: The numpy array
        """
        return torch.tensor(np_array.copy(), dtype=torch.float32, device=self.device)
