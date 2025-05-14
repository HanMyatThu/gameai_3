import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from memory import ReplayBuffer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()
        # Common feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Combine to get Q values: V(s) + (A(s,a) - mean(A(s,.)))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

class NewAgent:
    """
    Dueling Double DQN agent using uniform experience replay.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 memory_size=50000,
                 batch_size=64,
                 gamma=0.99,
                 lr=1e-4,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 update_target_every=300,
                 chkpt_dir='models/new_agent'):
        # Environment and training parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_every = update_target_every
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size)

        # Q-Networks
        self.q_online = DuelingQNetwork(state_dim, action_dim).to(device)
        self.q_target = DuelingQNetwork(state_dim, action_dim).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.steps_done = 0

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.q_online(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """Sample a batch and perform a learning step."""
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Current Q values
        q_current = self.q_online(states_t).gather(1, actions_t)
        # Double DQN: next action from online, value from target
        next_actions = self.q_online(next_states_t).argmax(dim=1, keepdim=True)
        q_next = self.q_target(next_states_t).gather(1, next_actions)
        q_target = rewards_t + (1 - dones_t) * self.gamma * q_next

        # Compute loss and optimize
        loss = self.loss_fn(q_current, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_models(self, tag='latest'):
        """Save model checkpoints."""
        torch.save(self.q_online.state_dict(), os.path.join(self.chkpt_dir, f'q_online_{tag}.pth'))
        torch.save(self.q_target.state_dict(), os.path.join(self.chkpt_dir, f'q_target_{tag}.pth'))

    def load_models(self, tag='latest'):
        """Load model checkpoints."""
        online_path = os.path.join(self.chkpt_dir, f'q_online_{tag}.pth')
        target_path = os.path.join(self.chkpt_dir, f'q_target_{tag}.pth')
        if os.path.exists(online_path) and os.path.exists(target_path):
            self.q_online.load_state_dict(torch.load(online_path, map_location=device))
            self.q_target.load_state_dict(torch.load(target_path, map_location=device))
            self.q_online.eval()
            self.q_target.eval()
        else:
            raise FileNotFoundError(f"Checkpoint files not found for tag '{tag}'")
