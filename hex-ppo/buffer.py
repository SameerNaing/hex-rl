import numpy as np
import scipy

import torch

import config


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    def __init__(self, size, gamma=config.BUF_GAMMA, lam=config.GAE_LAMBDA, board_size=config.BOARD_SIZE):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.N = board_size

        self.obs_buf = np.zeros((size, config.OBS_CH * self.N * self.N), dtype=np.float32)
        self.action_buf = np.zeros(size, dtype=np.int64)
        self.action_mask_buf = np.zeros((size, self.N * self.N), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.advantage_buf = np.zeros(size, dtype=np.float32)
        self.value_buf = np.zeros(size, dtype=np.float32)
        self.return_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.idx, self.start_idx = 0, 0
        
    def store(self, obs, action, action_mask, reward, logp, value):
        self.obs_buf[self.idx] = obs
        self.action_buf[self.idx] = action
        self.action_mask_buf[self.idx] = action_mask
        self.reward_buf[self.idx] = reward
        self.logp_buf[self.idx] = logp
        self.value_buf[self.idx] = value
        self.idx += 1
        
    
    def calculate(self, winner=0, last_value=0.0):
        s = slice(self.start_idx, self.idx)
        rewards = self.reward_buf[s]

        if winner != 0:
            rw = rewards.copy()
            rw[::2] = winner
            rw[1::2] = -winner
            self.reward_buf[s] = rw
            rewards = rw

        rewards_ext = np.append(rewards, last_value)
        values_ext = np.append(self.value_buf[s], last_value)

        deltas = rewards_ext[:-1] + self.gamma * values_ext[1:] - values_ext[:-1]
        self.advantage_buf[s] = discount_cumsum(deltas, self.gamma * self.lam)
        self.return_buf[s] = discount_cumsum(rewards_ext, self.gamma)[:-1]

        self.start_idx = self.idx
    
    def get(self):
        mean, std = self.advantage_buf.mean(), self.advantage_buf.std()
        self.advantage_buf = (self.advantage_buf - mean) / std
        data = dict(
            obs= torch.as_tensor(self.obs_buf.reshape(self.size, config.OBS_CH, config.BOARD_SIZE, config.BOARD_SIZE), dtype=torch.float32, device=config.DEVICE),
            act=torch.as_tensor(self.action_buf, dtype=torch.long, device=config.DEVICE),
            act_mask=torch.as_tensor(self.action_mask_buf, dtype=torch.bool, device=config.DEVICE),
            ret=torch.as_tensor(self.return_buf, dtype=torch.float32, device=config.DEVICE),
            adv=torch.as_tensor(self.advantage_buf, dtype=torch.float32, device=config.DEVICE),
            logp=torch.as_tensor(self.logp_buf, dtype=torch.float32, device=config.DEVICE),
        )
        self.start_idx, self.idx = 0, 0
        return data 
        
    def __len__(self):
        return self.reward_buf.shape[0]