import os

from torch.distributions import Categorical
from functools import lru_cache


from .core.policy_net import get_policy_model
from .core.mcts import MCTS
from .core.utils import get_obs, get_mask


def _weights_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))   # directory of this file
    path = os.path.normpath(os.path.join(base_dir, "core", "weights", filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights not found at: {path}")
    return path

@lru_cache(maxsize=1)
def _load_policy_model_ppo():
    path = _weights_path("ppo.pt")
    model = get_policy_model(path)
    model.eval()
    return model


@lru_cache(maxsize=1)
def _load_policy_model_ppo_alphazero():
    path = _weights_path("ppo_alphazero.pt")
    model = get_policy_model(path)
    model.eval()
    return model



def play_ppo(state):
    obs  = get_obs(state)   
    mask = get_mask(state)  

    model = _load_policy_model_ppo()
    probs, _ = model.play(obs, mask)  

    dist = Categorical(probs=probs)
    action = dist.sample().item()

    return int(action)

def play_alphazero(state):
    model = _load_policy_model_ppo_alphazero()
    mcts = MCTS(state.clone(), model)
    root = mcts.run(120)
    action = root.select_action(0)
    
    return action 

    
    
    
    
    
