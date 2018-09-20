from collections import deque, namedtuple
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, sample_size, max_len=int(1e5)):
        self.buffer = deque(maxlen=max_len)
        self.sample_size = sample_size
        self.experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state'))

    def add(self, state, action, reward, next_state):
        exp = self.experience(state, action, reward, next_state)

        self.buffer.append(exp)

    def sample(self):
        # _, exps = zip(*random.sample(self.buffer, self.sample_size))
        exps = random.sample(self.buffer, self.sample_size)
        states = np.vstack([e.state for e in exps if e is not None])
        actions = np.vstack([e.action for e in exps if e is not None])
        rewards = np.vstack([e.reward for e in exps if e is not None])
        next_states = np.vstack([e.next_state for e in exps if e is not None])

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)