import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import os

from memory import PPOMemory

class ActorNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dims=(256, 256)):
        super(ActorNetwork, self).__init__()

        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions))

        self.actor = nn.Sequential(*layers)

    def forward(self, state):
        distribution_params = self.actor(state)
        dist = Categorical(logits=distribution_params)
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, n_observations, hidden_dims=(256, 256)):
        super(CriticNetwork, self).__init__()

        layers = []
        input_dim = n_observations
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.critic = nn.Sequential(*layers)

    def forward(self, state):
        value = self.critic(state)
        return value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, entropy_coef=0.01, value_loss_coef=0.5,
                 chkpt_dir='models/ppo_1'):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = PPOMemory(self.batch_size, self.device)

        self.total_steps = 0

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def select_action(self, observation):
        """Selects action based on current policy, returns action, log_prob, value"""
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor(observation)
            value = self.critic(observation)

        action = dist.sample()
        log_probs = dist.log_prob(action)


        action = action.squeeze().item()
        log_probs = log_probs.squeeze().item()
        value = value.squeeze().item()

        return action, log_probs, value

    def learn(self):
        """Performs the PPO update step."""
        if len(self.memory) < self.batch_size:
             print(f"Skipping learn step: Memory size {len(self.memory)} < Batch size {self.batch_size}")
             return {}, 0.0

        for _ in range(self.n_epochs):
            (state_arr, action_arr, old_prob_arr, vals_arr,
             reward_arr, dones_arr, batches) = self.memory.generate_batches()

            advantages = self._calculate_gae(reward_arr, vals_arr, dones_arr)
            returns = advantages + vals_arr
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy_loss = 0
            num_batches = len(batches)

            # Processing data in batches
            for batch_indices in batches:
                states = state_arr[batch_indices]
                old_probs = old_prob_arr[batch_indices]
                actions = action_arr[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze() # Shape: [batch_size]

                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Calculate the ratio (pi_new / pi_old)
                prob_ratio = torch.exp(new_probs - old_probs)

                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                critic_loss = F.mse_loss(critic_value, batch_returns) # value vs returns

                total_loss = (actor_loss +
                              self.value_loss_coef * critic_loss -
                              self.entropy_coef * entropy)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()


        # Clear memory after updates
        self.memory.clear_memory()

        avg_losses = {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'entropy_loss': total_entropy_loss / num_batches
        }
        avg_total_loss = avg_losses['actor_loss'] + self.value_loss_coef * avg_losses['critic_loss'] - self.entropy_coef * avg_losses['entropy_loss']

        return avg_losses, avg_total_loss


    def _calculate_gae(self, rewards, values, dones):
        """Calculates Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0
            else:
                 next_non_terminal = 1.0 - dones[t+1].float()
                 next_value = values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            next_adv_term = self.gamma * self.gae_lambda * last_advantage * next_non_terminal
            advantages[t] = delta + next_adv_term
            last_advantage = advantages[t]

        return advantages

    def get_greedy_action(self, observation):
        """Selects the action with the highest probability (greedy)."""
        # Convert observation to tensor if it's not already
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        with torch.no_grad():
            dist = self.actor(observation)
            action = torch.argmax(dist.probs, dim=1)

        action = action.squeeze().item()
        return action


    def save_models(self, episode):
        """Saves actor and critic models."""
        print(f'... saving models at episode {episode} ...')
        torch.save(self.actor.state_dict(), os.path.join(self.chkpt_dir, f'actor_ppo_{episode}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self.chkpt_dir, f'critic_ppo_{episode}.pth'))

    def load_models(self, episode):
        """Loads actor and critic models."""
        actor_path = os.path.join(self.chkpt_dir, f'actor_ppo_{episode}.pth')
        critic_path = os.path.join(self.chkpt_dir, f'critic_ppo_{episode}.pth')

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            print(f'... loading models from episode {episode} ...')
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.actor.eval()
            self.critic.eval()
        else:
            print(f"Error: Model files not found for episode {episode} in {self.chkpt_dir}")

    def set_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def set_train_mode(self):
        self.actor.train()
        self.critic.train()