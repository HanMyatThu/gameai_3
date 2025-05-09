# double_dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

from memory import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, memory_size=50000, batch_size=64, gamma=0.99,
                 lr=1e-4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, update_target_every=300,
                 chkpt_dir='models/doubleDQN/best'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_every = update_target_every
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.q_online = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_online.to(self.device)
        self.q_target.to(self.device)

        # self.replay_buffer = deque(maxlen=memory_size)
        self.replay_buffer = ReplayBuffer(capacity=memory_size)
        self.steps_done = 0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_online(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        # self.replay_buffer.append((state, action, reward, next_state, done))
          
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # batch = random.sample(self.replay_buffer, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_online(states).gather(1, actions)

        # Double DQN: select next action via online network, evaluate using target
        with torch.no_grad():
            next_actions = self.q_online(next_states).argmax(1, keepdim=True)
            next_q = self.q_target(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_models(self, tag="latest"):
        print(f"Saving DDQN model as '{tag}'")
        torch.save(self.q_online.state_dict(), os.path.join(self.chkpt_dir, f'q_online_{tag}.pth'))
        torch.save(self.q_target.state_dict(), os.path.join(self.chkpt_dir, f'q_target_{tag}.pth'))

    def load_models(self, tag="latest"):
        q_online_path = os.path.join(self.chkpt_dir, f'q_online_{tag}.pth')
        q_target_path = os.path.join(self.chkpt_dir, f'q_target_{tag}.pth')

        if os.path.exists(q_online_path) and os.path.exists(q_target_path):
            self.q_online.load_state_dict(torch.load(q_online_path, map_location=self.device))
            self.q_target.load_state_dict(torch.load(q_target_path, map_location=self.device))
            self.q_online.eval()
            self.q_target.eval()
            print(f"Loaded models from tag '{tag}'")
        else:
            raise FileNotFoundError(f"Model files not found for tag: {tag}")