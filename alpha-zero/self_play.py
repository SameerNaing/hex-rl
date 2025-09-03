from env import game, get_obs, get_mask

from mcts import MCTS
from config import *

def self_play_game(model, num_simulations, buffer):
    state = game.new_initial_state()
    trajectory = []
    
    move_number = 0

    while not state.is_terminal():
        temp = 1.0 if move_number < TEMP_MOVES else 0.0
        
        current_player = 1 if state.current_player() == 0 else -1
        
        mcts = MCTS(state.clone(), model)
        root = mcts.run(num_simulations)
        action_probs = root.get_action_probs(temp=temp)
        obs = get_obs(state, "cpu").squeeze(0).numpy()
        mask = get_mask(state, "cpu").squeeze(0).numpy()
        trajectory.append((obs, mask, current_player, action_probs))
        
        action = root.select_action(temp)
        state.apply_action(action)
        
        move_number += 1

    
    reward = 0
    if state.player_reward(0) == 1.0:
        reward = 1.0
    elif state.player_reward(1) == 1.0:
        reward = -1.0
    
    for obs, mask, player, action_probs in trajectory:
        value = reward if player == 1 else -reward
        buffer.add(obs, mask, action_probs, value)
    
