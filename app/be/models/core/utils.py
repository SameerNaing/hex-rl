import torch

def get_obs(state):
    return torch.tensor(state.observation_tensor()).reshape(1, 9, 5, 5)

def get_mask(state):
    return torch.tensor(state.legal_actions_mask()).unsqueeze(0).bool()
