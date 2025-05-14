# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
import math

# Define the structure of a transition (experience)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-Network model."""
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Simple Feedforward Network
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Output Q-values for each action

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_capacity=10000,
                 batch_size=128, gamma=0.99, lr=1e-4,
                 tau=0.005, epsilon_start=0.9, epsilon_end=0.05,
                 epsilon_decay=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma # Discount factor
        self.lr = lr       # Learning rate
        self.tau = tau     # Target network update rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Policy Network: Gets updated frequently
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        # Target Network: Gets updated slowly, provides stable targets
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(replay_capacity)

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        # Calculate epsilon based on decay schedule
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        self.epsilon = eps_threshold # Store current epsilon for monitoring

        if sample > eps_threshold:
            # Exploitation: Choose the best action based on the policy network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_values = self.policy_net(state_tensor)
                # Return the action with the highest Q-value
                return action_values.max(1)[1].view(1, 1)
        else:
            # Exploration: Choose a random action
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def store_transition(self, state, action, next_state, reward, done):
        """Stores a transition in the replay memory."""
        # Convert inputs to tensors before storing
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        # Action is already a tensor from select_action
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device) if next_state is not None else None
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([done], dtype=torch.bool, device=self.device)

        self.memory.push(state_t, action, next_state_t, reward_t, done_t)

    def learn(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return # Not enough memories to learn yet

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details)
        batch = Transition(*zip(*transitions))

        # Filter out transitions where next_state is None (terminal states)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                       device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        # Concatenate batch elements
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        # These are the Q-values the policy network *predicts* for the actions that were taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad(): # We don't need gradients for the target computation
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values (target values)
        # target = reward + gamma * max_a' Q_target(s', a')
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss (less sensitive to outliers than MSE)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return loss.item() # Return loss for monitoring

    def save_model(self, path="flappy_bird_dqn.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="flappy_bird_dqn.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
            self.policy_net.eval() # Set to evaluation mode if not training
            self.target_net.eval()
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
        except Exception as e:
            print(f"Error loading model: {e}")