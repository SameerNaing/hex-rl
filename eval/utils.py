import torch

from policy_net import PolicyValueNet

def get_obs(state):
    return torch.tensor(state.observation_tensor()).reshape(1, 9, 5, 5)

def get_mask(state):
    return torch.tensor(state.legal_actions_mask()).unsqueeze(0).bool()


def get_policy_model(weight_path):
    weights = torch.load(weight_path, map_location=torch.device("cpu"))
    model = PolicyValueNet().to("cpu")
    model.load_state_dict(weights)
    
    return model