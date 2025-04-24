# memory.py
import numpy as np
import torch

class PPOMemory:
    def __init__(self, batch_size, device):
        self.states = []
        self.probs = []  # Log probabilities of actions taken
        self.vals = []   # Critic value estimates for states
        self.actions = []
        self.rewards = []
        self.dones = []  # Episode termination flags

        self.batch_size = batch_size
        self.device = device

    def generate_batches(self):
        """Generates randomized batches from the stored trajectory."""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Return data arrays converted to tensors
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device),
            torch.tensor(self.actions, dtype=torch.long).to(self.device),
            torch.tensor(self.probs, dtype=torch.float32).to(self.device),
            torch.tensor(self.vals, dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards, dtype=torch.float32).to(self.device),
            torch.tensor(self.dones, dtype=torch.bool).to(self.device),
            batches  # List of index arrays for each batch
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        """Stores a single step transition."""
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clears the memory after an update cycle."""
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)