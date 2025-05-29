import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from memory import ReplayBuffer # Adjusted import

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, memory_size=50000, batch_size=64, gamma=0.99,
                 lr=1e-4, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 target_update_freq=300,
                 chkpt_dir='DQN/models_dqn'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=memory_size)
        self.steps_done = 0

    def get_action(self, state):
        if random.random() < self.epsilon: # Epsilon check first
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def get_greedy_action(self, state):
        """Always pick the highest-Q action, ignoring ε."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        # Handle None in next_states for terminal states
        # Create a mask for non-final next states
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool, device=self.device)

        non_final_next_states_list = [s for s in next_states if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.FloatTensor(np.array(non_final_next_states_list)).to(self.device)
        else:
            # If all next_states are None, we create an empty tensor of appropriate shape to avoid errors
            # And we ensure that subsequent operations handle empty tensors.
            non_final_next_states = torch.empty((0, self.state_dim), device=self.device)


        current_q = self.policy_net(states).gather(1, actions)

        # Compute Q values for next states: Q_target(s', argmax_a Q_online(s', a))
        # For DQN, we take max Q value from the target network for next states
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        if len(non_final_next_states_list) > 0:
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        target_q = rewards.squeeze(1) + (1 - torch.FloatTensor(np.array(dones)).to(self.device)) * self.gamma * next_q_values
        target_q = target_q.unsqueeze(1)


        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0: # Matched Ddqn_agent.py
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min


    def save_models(self, tag="latest"): # Matched Ddqn_agent.py
        print(f"Saving DQN model as '{tag}'")
        # Save policy_net as q_online and target_net
        torch.save(self.policy_net.state_dict(), os.path.join(self.chkpt_dir, f'q_online_{tag}.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(self.chkpt_dir, f'q_target_{tag}.pth'))

    def load_models(self, tag="latest", model_path_override=None):
        if model_path_override:
            if os.path.exists(model_path_override):
                print(f"Loading DQN model from override path: {model_path_override}")
                loaded_data = torch.load(model_path_override, map_location=self.device)
                if isinstance(loaded_data, dict) and "policy_net_state_dict" in loaded_data:
                    # Handle case where the loaded file is a checkpoint dictionary
                    print("Loaded data is a checkpoint dictionary. Extracting 'policy_net_state_dict'.")
                    self.policy_net.load_state_dict(loaded_data["policy_net_state_dict"])
                elif isinstance(loaded_data, dict) and "net.0.weight" not in loaded_data:
                    # Attempt to find a state_dict if common wrapper keys are present
                    possible_keys = ["policy_net_state_dict", "policy_state_dict", "q_online_state_dict", "state_dict"]
                    found_key = None
                    for key in possible_keys:
                        if key in loaded_data:
                            found_key = key
                            break
                    if found_key:
                        print(f"Loaded data is a dictionary. Extracting from key: '{found_key}'.")
                        self.policy_net.load_state_dict(loaded_data[found_key])
                    else:
                        raise RuntimeError(f"DQN model file at {model_path_override} is a dictionary but does not contain a recognized state_dict key.")
                else:
                    self.policy_net.load_state_dict(loaded_data)
                
                self.policy_net.eval()
                self.target_net.load_state_dict(self.policy_net.state_dict()) # Copy policy to target
                self.target_net.eval()
                print(f"Successfully loaded DQN model from '{model_path_override}' and updated target network.")
            else:
                raise FileNotFoundError(f"DQN model file not found at override path: {model_path_override}")
            return

        q_online_path = os.path.join(self.chkpt_dir, f'q_online_{tag}.pth')
        q_target_path = os.path.join(self.chkpt_dir, f'q_target_{tag}.pth')

        if not os.path.exists(q_online_path):
            raise FileNotFoundError(f"DQN q_online model file not found for tag: {tag} at {q_online_path}")

        self.policy_net.load_state_dict(torch.load(q_online_path, map_location=self.device))
        self.policy_net.eval()

        if os.path.exists(q_target_path):
            self.target_net.load_state_dict(torch.load(q_target_path, map_location=self.device))
            self.target_net.eval()
            print(f"Loaded DQN models (q_online and q_target) from tag '{tag}' in {self.chkpt_dir}")
        else: 
            print(f"Warning: DQN q_target model file not found for tag: {tag} at {q_target_path}. Copying from policy_net.")
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            print(f"Loaded DQN q_online model from tag '{tag}' in {self.chkpt_dir} and copied to target_net.") 