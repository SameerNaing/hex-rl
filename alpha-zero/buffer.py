import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, obs, masks, policy, value):
        self.buffer.append((obs, masks, policy, value))
        
    def sample(self, batch_size):
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        obs, masks, action_probs, values = zip(*batch)
        return obs, masks, action_probs, values

    def __len__(self):
        return len(self.buffer)
