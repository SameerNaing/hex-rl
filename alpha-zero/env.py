import torch
import pyspiel

from config import BOARD_SIZE, OBS_CH, DEVICE


# game = pyspiel.load_game("tic_tac_toe")
game = pyspiel.load_game("hex(board_size=5)")

def get_obs(state, device=DEVICE):
    return torch.tensor(state.observation_tensor(), device=device).reshape(1, OBS_CH, BOARD_SIZE, BOARD_SIZE)

def get_mask(state, device=DEVICE):
    return torch.tensor(state.legal_actions_mask(), device=device).unsqueeze(0).bool()
