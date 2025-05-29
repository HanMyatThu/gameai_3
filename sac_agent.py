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
                                  # For Flappy Bird (discrete), this will need adjustment in the agent
        
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds correction for log_prob
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) # Correction for tanh squashing
        
        # For discrete actions, this log_prob might need to be handled differently.
        # If action_dim is > 1 and represents a multi-discrete space, sum might be okay.
        # If it's a single discrete action selected via argmax later, this log_prob isn't directly used for that selection.
        
        # The original check was `if action_dim > 1:`. 
        # This needs to be based on the actual dimension of the action tensor here.
        # `action` tensor shape is (batch_size, action_output_dim_of_this_policy)
        # If the policy's action output dimension is greater than 1, then sum log_prob across that dimension.
        if action.shape[-1] > 1: 
            log_prob = log_prob.sum(axis=-1, keepdim=True)
        # else: log_prob remains as is (e.g., for a single continuous action, shape is [batch_size, 1])

        return action, log_prob, mean # Return mean for deterministic actions

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        # SAC Q-networks usually take state and action as input
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), # For continuous actions
            # If discrete actions, this might need adjustment (e.g. state_dim only, output Q for all actions)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single Q-value for the state-action pair
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        # This concatenation is for continuous actions.
        # For discrete actions, the Q network architecture might be different:
        # Input: state, Output: Q-values for each action. Then gather the Q-value for 'action'.
        # For now, assuming the environment might be adapted or this agent is for continuous.
        # If Flappy Bird (discrete), this needs change.
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, # action_dim for Flappy Bird is typically 2 (flap or not)
                 lr=3e-4, # learning_rate from original
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2, # Entropy coefficient
                 memory_size=50000, # buffer_size from original, to match other agents
                 batch_size=64, # batch_size from original was 256, adjusted to common value
                 hidden_dim=256,
                 target_entropy_ratio=0.98, # For auto entropy tuning, as a ratio of action_dim
                 auto_entropy_tuning=True,
                 chkpt_dir='SAC/models_sac', # Checkpoint directory
                 action_space_type='discrete'): # Added to handle action space
        
        self.state_dim = state_dim
        self.action_dim = action_dim # This is num_actions for discrete, or dim of action vector for continuous
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # Initial alpha if not auto-tuning
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        self.chkpt_dir = chkpt_dir
        self.action_space_type = action_space_type
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SAC Using device: {self.device}")

        # Actor Network (Policy)
        # For discrete SAC, actor outputs logits for a categorical distribution.
        if self.action_space_type == 'discrete':
            self.policy = ActorNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
        else: # Continuous SAC (original implementation)
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic Networks (Q-functions)
        # For discrete SAC, Q-networks output Q-values for all actions given a state.
        if self.action_space_type == 'discrete':
            self.q1 = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q2 = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q1_target = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
            self.q2_target = QNetworkDiscrete(state_dim, action_dim, (hidden_dim, hidden_dim)).to(self.device)
        else: # Continuous SAC
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
        
        self.replay_buffer = ReplayBuffer(capacity=memory_size) # Using ReplayBuffer
        
        if self.auto_entropy_tuning:
            if self.action_space_type == 'discrete':
                # Target entropy for discrete is often log(|A|) * target_entropy_ratio
                self.target_entropy = -np.log((1.0/action_dim)) * target_entropy_ratio 
            else: # Continuous
                self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item() * target_entropy_ratio
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item() # Initialize alpha from log_alpha
        
        self.steps_done = 0 # For potential scheduling or logging

    def get_action(self, state, evaluate=False): # Renamed from select_action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.action_space_type == 'discrete':
            if evaluate: # Greedy action
                with torch.no_grad():
                    # For discrete, sample method with evaluate=True returns argmax action
                    action, _, _ = self.policy.sample(state, evaluate=True) 
            else: # Sample action
                with torch.no_grad():
                    action, _, _ = self.policy.sample(state, evaluate=False)
            return action.item()
        else: # Continuous
            with torch.no_grad():
                if evaluate:
                    # For GaussianPolicy, sample returns (tanh(x_t), log_prob, mean)
                    # For greedy continuous action, we want tanh(mean)
                    _, _, action_mean = self.policy.sample(state, reparameterize=False) # Get mean for eval
                    action = torch.tanh(action_mean) # Squash mean for deterministic action
                else:
                    action, _, _ = self.policy.sample(state, reparameterize=True)
            # .cpu().numpy() converts to numpy array, [0] extracts the scalar/vector
            # If action_dim=1, this results in a scalar float. If action_dim > 1, it's a numpy array.
            return action.cpu().numpy().item() if self.action_dim == 1 else action.cpu().numpy()[0]

    def get_greedy_action(self, state):
        """Selects action based on the mean of the policy distribution (greedy)."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                # ActorNetworkDiscrete.sample(evaluate=True) returns argmax action
                action, _, _ = self.policy.sample(state, evaluate=True)
                return action.item()
            else: # Continuous (GaussianPolicy)
                # GaussianPolicy.sample returns (tanh(x_t), log_prob, mean)
                # For greedy continuous action, we take tanh(mean)
                _, _, action_mean = self.policy.sample(state, reparameterize=False) # reparameterize=False is fine for getting mean
                action = torch.tanh(action_mean) # Squash mean
                # If action_dim=1, .item() converts a single-element tensor to a Python number.
                # If action_dim > 1, .cpu().numpy()[0] might be needed if sample returns a more complex structure,
                # but given action_dim is 1 for this model, .item() is cleaner.
                return action.item() if self.action_dim == 1 else action.cpu().numpy()[0] # Match get_action structure

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None # Indicate learning was skipped

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        # Actions need to be shaped correctly for discrete vs continuous
        if self.action_space_type == 'discrete':
            actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        else: # Continuous
            actions = torch.FloatTensor(np.array(actions)).to(self.device) # Potentially (batch, action_dim)
            if actions.ndim == 1: actions = actions.unsqueeze(1) # Ensure (batch, 1) if action_dim is 1

        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) # Assuming next_state is never None here, or handled by ReplayBuffer
        dones = torch.FloatTensor(np.array(dones).astype(np.uint8)).unsqueeze(1).to(self.device)

        # Update Q-functions
        with torch.no_grad():
            if self.action_space_type == 'discrete':
                next_state_actions, next_state_log_pi, _ = self.policy.sample(next_states, evaluate=False)
                q1_next_target_all_actions = self.q1_target(next_states)
                q2_next_target_all_actions = self.q2_target(next_states)
                
                # Get probabilities for Soft Q update
                next_action_probs = torch.exp(next_state_log_pi)
                
                # Weighted sum of Q-values by action probabilities for E[Q(s',a')]
                min_q_next_target_expected = torch.min(
                    torch.sum(next_action_probs * q1_next_target_all_actions, dim=1, keepdim=True),
                    torch.sum(next_action_probs * q2_next_target_all_actions, dim=1, keepdim=True)
                )
                # Subtract entropy term: alpha * H(A|s') = alpha * E_a~pi [ -log pi(a|s') ] = -alpha * sum(pi * log pi)
                # Here, next_state_log_pi is log pi(a_sampled | s'), so sum(pi * log pi) is E[log pi]
                entropy_term = self.alpha * torch.sum(next_action_probs * (-next_state_log_pi), dim=1, keepdim=True)
                next_q_value = min_q_next_target_expected + entropy_term # Add because we use -log_pi for entropy calc
                
            else: # Continuous
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
        else: # Continuous
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
        # Freeze Q-networks for policy update
        for p in self.q1.parameters(): p.requires_grad = False
        for p in self.q2.parameters(): p.requires_grad = False

        if self.action_space_type == 'discrete':
            # For discrete, sample actions, get log_probs, then get Q values for these actions
            sampled_actions, log_pi, _ = self.policy.sample(states, evaluate=False) # log_pi is [batch, num_actions]
            q1_pi = self.q1(states) # Q values for all actions
            q2_pi = self.q2(states)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Policy loss: E_s [ E_a~pi [alpha * log pi(a|s) - Q(s,a)] ]
            # For discrete: sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a))
            action_probs = torch.exp(log_pi)
            policy_loss_terms = action_probs * (self.alpha * log_pi - min_q_pi)
            policy_loss = policy_loss_terms.sum(dim=1).mean()
        else: # Continuous
            pi_actions, log_pi, _ = self.policy.sample(states)
            q1_pi = self.q1(states, pi_actions)
            q2_pi = self.q2(states, pi_actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Unfreeze Q-networks
        for p in self.q1.parameters(): p.requires_grad = True
        for p in self.q2.parameters(): p.requires_grad = True

        # Update alpha (entropy coefficient)
        if self.auto_entropy_tuning:
            if self.action_space_type == 'discrete':
                _, log_pi_alpha, _ = self.policy.sample(states, evaluate=False) # Use current policy's log_probs
                action_probs_alpha = torch.exp(log_pi_alpha)
                # entropy = -torch.sum(action_probs_alpha * log_pi_alpha, dim=1)
                # alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
                # Simpler: use log_pi of the sampled action directly if target_entropy is scalar
                # The objective is E[-alpha * log pi(a|s) - alpha * target_entropy]
                # log_pi here is log_prob of all actions for discrete case
                # We need E_a~pi [log pi(a|s)] which is sum_a pi(a|s)log pi(a|s)
                current_entropy = -torch.sum(action_probs_alpha * log_pi_alpha, dim=1).mean()
                alpha_loss = -(self.log_alpha * (current_entropy + self.target_entropy).detach()).mean()
            else: # Continuous
                _, log_pi, _ = self.policy.sample(states, reparameterize=False) # No need to reparam for alpha loss
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
        return q1_loss.item(), policy_loss.item() # Return losses for logging

    def save_models(self, tag="latest"): # Adjusted to be similar to Ddqn_agent.py style
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
            'action_space_type': self.action_space_type # Save action space type
        }
        torch.save(checkpoint, model_path)

    def load_models(self, tag="latest", model_path_override=None): # Adjusted
        load_path = model_path_override if model_path_override else os.path.join(self.chkpt_dir, f'sac_model_{tag}.pth')
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"SAC Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Load action_space_type first if it was saved, to correctly initialize networks if needed
        if 'action_space_type' in checkpoint:
            loaded_action_space_type = checkpoint['action_space_type']
            if self.action_space_type != loaded_action_space_type:
                print(f"Warning: Agent initialized with action_space_type '{self.action_space_type}' but model saved with '{loaded_action_space_type}'. Attempting to load with saved type.")
                # This might require re-initializing the networks if they are structurally different
                # For simplicity here, we assume the current agent instance is already configured correctly or this is a new instance.
                # If networks are different, they need to be re-instantiated before loading state_dict.
                # This simplified load assumes networks are compatible or already re-instantiated.
                pass # Potentially re-init networks here based on loaded_action_space_type

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        # Load alpha and related components more robustly
        self.alpha = checkpoint.get('alpha', self.alpha) # Load saved alpha if present, else keep current agent's alpha

        if self.auto_entropy_tuning:
            loaded_log_alpha = checkpoint.get('log_alpha') # Use .get() to safely access
            if loaded_log_alpha is not None:
                # Ensure self.log_alpha exists (it should from __init__ if auto_entropy_tuning is True)
                if hasattr(self, 'log_alpha') and self.log_alpha is not None:
                    self.log_alpha.data = loaded_log_alpha
                    # Try to load optimizer state if log_alpha was present and optimizer state exists
                    alpha_opt_state = checkpoint.get('alpha_optimizer_state_dict')
                    if alpha_opt_state is not None:
                        if hasattr(self, 'alpha_optimizer') and self.alpha_optimizer is not None:
                            try:
                                self.alpha_optimizer.load_state_dict(alpha_opt_state)
                            except Exception as e:
                                print(f"Warning: Could not load alpha_optimizer_state_dict: {e}")
                        else:
                            print("Warning: Checkpoint has alpha_optimizer_state_dict, but agent has no alpha_optimizer.")
                    # else: print("Debug: log_alpha loaded but alpha_optimizer_state_dict not found in checkpoint.") # Less critical warning
                else:
                    print("Warning: Checkpoint has log_alpha, but agent has no log_alpha attribute or it is None.")
            else:
                # If auto_entropy_tuning is True for this agent, but no log_alpha in checkpoint,
                # the agent's self.log_alpha (from __init__) will be used for any subsequent tuning.
                # The self.alpha value (loaded above or from __init__) will be used.
                print(f"Info: Agent has auto_entropy_tuning=True, but 'log_alpha' not found in checkpoint. Agent will use its initial log_alpha for any new tuning. Current alpha value: {self.alpha}")
        # If self.auto_entropy_tuning is False for the current agent, self.alpha is loaded if present in checkpoint, otherwise it remains as set in __init__.

        self.steps_done = checkpoint.get('steps_done', 0) # Use .get for backward compatibility
        
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
        # Target networks usually stay in eval, but can be train if needed for specific exploration.
        # Keeping them eval by convention unless specified.
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
        layers.append(nn.Linear(input_dim, n_actions)) # Outputs logits for each action
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
        # log_prob of the taken/greedy action. For categorical, gather is needed.
        # action from sample() is already (batch_size), need to keepdim for gather
        log_prob_taken_action = dist.log_prob(action) # This is already (batch_size)
        
        # For SAC discrete, policy loss uses sum_a pi(a|s) * (alpha * log pi(a|s) - Q(s,a))
        # So we need log_probs_all_actions (i.e. log_softmax output)
        # And for entropy term in Q target, we need sum_a pi(a|s) * (-log pi(a|s))
        
        return action, log_probs_all_actions, action # returning action twice to match continuous tuple for now, one is for greedy

class QNetworkDiscrete(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions)) # Outputs Q-value for each action
        self.critic = nn.Sequential(*layers)

    def forward(self, state):
        q_values = self.critic(state)
        return q_values # Returns Q-values for all actions 