import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """Initialize the buffer with a fixed maximum capacity."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        capacity: max number of transitions to store
        alpha: how much prioritization is used (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha    = alpha
        self.buffer   = []
        self.priorities = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        """Add a new transition with maximum priority so far."""
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_prio)
        else:
            # overwrite the oldest
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, returning:
          states, actions, rewards, next_states, dones, indices, is_weights
        beta: exponent for importance‐sampling weights (0=no correction, 1=full correction)
        """
        N = len(self.buffer)
        # compute probabilities
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios**self.alpha
        probs /= probs.sum()

        # sample indices according to probs
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # compute importance‐sampling weights
        total   = N
        weights = (total * probs[indices])**(-beta)
        weights /= weights.max()  # normalize to [0,1]

        # unzip
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, new_priorities):
        """After learning from a batch, update those transitions’ priorities."""
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)