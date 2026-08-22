import random
from collections import namedtuple

# 1. Removed "policies" and "action_masks" to match the modified MA2C agent
Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer (Acts as a Rollout Buffer for the on-policy A2C)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 2. Updated parameters to only accept the 5 core MDP transitions
    def _push_one(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states, dones):
        # 3. Cleaned up the loop to directly zip the 5 lists provided by agent.explore()
        if isinstance(states, list):
            for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
                self._push_one(s, a, r, n_s, d)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
            
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        # Reset the memory 
        # (This is correct behavior for on-policy Actor-Critic algorithms, 
        # ensuring the networks only train on fresh data).
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)