import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random # Added random for action selection if needed, though SAC is typically deterministic or samples from policy
import os

from memory import ReplayBuffer # Adjusted import
from torch.distributions import Normal # For GaussianPolicy

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming init for layers before ReLU, Xavier for output (tanh)
            if module is self.mean or module is self.log_std:
                 nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state, reparameterize=True):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        if reparameterize:
            x_t = normal.rsample()  # Reparameterization trick
        else:
            x_t = normal.sample()
            
        action = torch.tanh(x_t)  # Squash to [-1, 1] for environments with bounded actions
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) # Correction for tanh squashing

        if action.shape[-1] > 1: 
            log_prob = log_prob.sum(axis=-1, keepdim=True)

        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # Take state and action as input
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, # action_dim for Flappy Bird is 2 (flap or not)
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 memory_size=50000,
                 batch_size=64,
                 hidden_dim=256,
                 target_entropy_ratio=0.98,
                 auto_entropy_tuning=True,
                 chkpt_dir='SAC/models_sac',
                 action_space_type='discrete'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        self.chkpt_dir = chkpt_dir
        self.action_space_type = action_space_type
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Using device: {self.device}")

        # Actor Network (Policy)
        if self.action_space_type == 'discrete':
            self.policy = ActorNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
        else:
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic Networks (Q-functions)
        if self.action_space_type == 'discrete':
            self.q1 = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q2 = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q1_target = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q2_target = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
        else:
            self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(capacity=memory_size)
        
        if self.auto_entropy_tuning:
            if self.action_space_type == 'discrete':
                # Target entropy for discrete is often log(|A|) * target_entropy_ratio
                self.target_entropy = -np.log((1.0/action_dim)) * target_entropy_ratio 
            else:
                self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item() * target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        
        self.steps_done = 0

    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.action_space_type == 'discrete':
            if evaluate:
                with torch.no_grad():
                    action, _, _ = self.policy.sample(state, evaluate=True) 
            else: # Sample action
                with torch.no_grad():
                    action, _, _ = self.policy.sample(state, evaluate=False)
            return action.item()
        else:
            with torch.no_grad():
                if evaluate:
                    # For GaussianPolicy, sample returns (tanh(x_t), log_prob, mean)
                    _, _, action_mean = self.policy.sample(state, reparameterize=False)
                    action = torch.tanh(action_mean) # Squash mean for deterministic action
                else:
                    action, _, _ = self.policy.sample(state, reparameterize=True)
            return action.cpu().numpy().item() if self.action_dim == 1 else action.cpu().numpy()[0]

    def get_greedy_action(self, state):
        """Selects action based on the mean of the policy distribution (greedy)."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                action, _, _ = self.policy.sample(state, evaluate=True)
                return action.item()
            else:
                _, _, action_mean = self.policy.sample(state, reparameterize=False)
                action = torch.tanh(action_mean)
                return action.item() if self.action_dim == 1 else action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        if self.action_space_type == 'discrete':
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        else:
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            if actions.ndim == 1: actions = actions.unsqueeze(1)

        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(np.uint8)).unsqueeze(1).to(self.device)

        # Update Q-functions
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                next_state_actions, next_state_log_pi, _ = self.policy.sample(next_states, evaluate=False)
                q1_next_target_all_actions = self.q1_target(next_states)
                q2_next_target_all_actions = self.q2_target(next_states)

                next_action_probs = torch.exp(next_state_log_pi)

                min_q_next_target_expected = torch.min(
                    torch.sum(next_action_probs * q1_next_target_all_actions, dim=1, keepdim=True),
                    torch.sum(next_action_probs * q2_next_target_all_actions, dim=1, keepdim=True)
                )

                entropy_term = self.alpha * torch.sum(next_action_probs * (-next_state_log_pi), dim=1, keepdim=True)
                next_q_value = min_q_next_target_expected + entropy_term # Add because we use -log_pi for entropy calc
                
            else:
                next_state_actions, next_state_log_pi, _ = self.policy.sample(next_states)
                q1_next_target = self.q1_target(next_states, next_state_actions)
                q2_next_target = self.q2_target(next_states, next_state_actions)
                min_q_next_target = torch.min(q1_next_target, q2_next_target)
                next_q_value = min_q_next_target - self.alpha * next_state_log_pi

            target_q = rewards + (1 - dones) * self.gamma * next_q_value

        if self.action_space_type == 'discrete':
            # Q networks output all action Qs, gather for the taken action
            q1_current = self.q1(states).gather(1, actions.long()) 
            q2_current = self.q2(states).gather(1, actions.long())
        else:
            q1_current = self.q1(states, actions)
            q2_current = self.q2(states, actions)
        
        q1_loss = F.mse_loss(q1_current, target_q)
        q2_loss = F.mse_loss(q2_current, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy
        for p in self.q1.parameters(): p.requires_grad = False
        for p in self.q2.parameters(): p.requires_grad = False

        if self.action_space_type == 'discrete':
            sampled_actions, log_pi, _ = self.policy.sample(states, evaluate=False)
            q1_pi = self.q1(states) # Q values for all actions
            q2_pi = self.q2(states)
            min_q_pi = torch.min(q1_pi, q2_pi)

            action_probs = torch.exp(log_pi)
            policy_loss_terms = action_probs * (self.alpha * log_pi - min_q_pi)
            policy_loss = policy_loss_terms.sum(dim=1).mean()
        else:
            pi_actions, log_pi, _ = self.policy.sample(states)
            q1_pi = self.q1(states, pi_actions)
            q2_pi = self.q2(states, pi_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for p in self.q1.parameters(): p.requires_grad = True
        for p in self.q2.parameters(): p.requires_grad = True

        # Update alpha (entropy coefficient)
        if self.auto_entropy_tuning:
            if self.action_space_type == 'discrete':
                _, log_pi_alpha, _ = self.policy.sample(states, evaluate=False)
                action_probs_alpha = torch.exp(log_pi_alpha)
                current_entropy = -torch.sum(action_probs_alpha * log_pi_alpha, dim=1).mean()
                alpha_loss = -(self.log_alpha * (current_entropy + self.target_entropy).detach()).mean()
            else:
                _, log_pi, _ = self.policy.sample(states, reparameterize=False)
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.steps_done += 1
        return q1_loss.item(), policy_loss.item()

    def save_models(self, tag="latest"):
        print(f"Saving SAC model as '{tag}'")
        model_path = os.path.join(self.chkpt_dir, f'sac_model_{tag}.pth')
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'alpha': self.alpha,
            'steps_done': self.steps_done,
            'action_space_type': self.action_space_type
        }
        torch.save(checkpoint, model_path)

    def load_models(self, tag="latest", model_path_override=None): # Adjusted
        load_path = model_path_override if model_path_override else os.path.join(self.chkpt_dir, f'sac_model_{tag}.pth')
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"SAC Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        if 'action_space_type' in checkpoint:
            loaded_action_space_type = checkpoint['action_space_type']
            if self.action_space_type != loaded_action_space_type:
                print(f"Warning: Agent initialized with action_space_type '{self.action_space_type}' but model saved with '{loaded_action_space_type}'. Attempting to load with saved type.")
                pass

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        # Loading alpha and related components
        self.alpha = checkpoint.get('alpha', self.alpha)

        if self.auto_entropy_tuning:
            loaded_log_alpha = checkpoint.get('log_alpha')
            if loaded_log_alpha is not None:
                if hasattr(self, 'log_alpha') and self.log_alpha is not None:
                    self.log_alpha.data = loaded_log_alpha
                    alpha_opt_state = checkpoint.get('alpha_optimizer_state_dict')
                    if alpha_opt_state is not None:
                        if hasattr(self, 'alpha_optimizer') and self.alpha_optimizer is not None:
                            try:
                                self.alpha_optimizer.load_state_dict(alpha_opt_state)
                            except Exception as e:
                                print(f"Warning: Could not load alpha_optimizer_state_dict: {e}")
                        else:
                            print("Warning: Checkpoint has alpha_optimizer_state_dict, but agent has no alpha_optimizer.")
                else:
                    print("Warning: Checkpoint has log_alpha, but agent has no log_alpha attribute or it is None.")
            else:
                print(f"Info: Agent has auto_entropy_tuning=True, but 'log_alpha' not found in checkpoint. "
                      f"Agent will use its initial log_alpha for any new tuning. Current alpha value: {self.alpha}")

        self.steps_done = checkpoint.get('steps_done', 0)
        
        self.set_eval_mode()
        print(f"Loaded SAC model from '{load_path}'")

    def set_eval_mode(self):
        self.policy.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()

    def set_train_mode(self):
        self.policy.train()
        self.q1.train()
        self.q2.train()
        self.q1_target.eval() 
        self.q2_target.eval()

# --- Networks for Discrete SAC ---
class ActorNetworkDiscrete(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.actor(state)
        return logits

    def sample(self, state, evaluate=False):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if evaluate:
            action = torch.argmax(probs, dim=-1) # Greedy action
        else:
            action = dist.sample() # Sample action
        
        log_probs_all_actions = F.log_softmax(logits, dim=-1)
        return action, log_probs_all_actions, action

class QNetworkDiscrete(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions))
        self.critic = nn.Sequential(*layers)

    def forward(self, state):
        q_values = self.critic(state)
        return q_values