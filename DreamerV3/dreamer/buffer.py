"""Flat-stream replay buffer with is_first flags (DreamerV3-style)."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros((capacity, action_dim), np.float32)  # prev action
        self.reward = np.zeros((capacity,), np.float32)
        self.is_first = np.zeros((capacity,), bool)
        self.is_terminal = np.zeros((capacity,), bool)  # terminal (not truncation)
        self.idx, self.full = 0, False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, prev_action, reward, is_first, is_terminal):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = prev_action
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_terminal[i] = is_terminal
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch, length, device):
        n = len(self)
        starts = np.random.randint(0, n - length, size=batch)
        idx = starts[:, None] + np.arange(length)[None, :]
        batch_dict = {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "action": torch.as_tensor(self.action[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "is_first": torch.as_tensor(self.is_first[idx], device=device),
            "cont": torch.as_tensor(
                1.0 - self.is_terminal[idx].astype(np.float32), device=device),
        }
        return batch_dict
