import torch
import random
import os
from collections import defaultdict
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator, MCTSBot

from mcts import MCTS as MCTSAlpha
from env import game


def terminal_reward_for_player(state, player_id):
    if hasattr(state, "player_reward"):  # some wrappers provide this
        return float(state.player_reward(player_id))
    # OpenSpiel states expose returns() at terminal
    if state.is_terminal():
        returns = state.returns()  # list of per-player returns
        return float(returns[player_id])
    return 0.0


def model_move(state, model, simulations=100, temperature=0):
    """Get move from model using our AlphaZero-style MCTS."""
    mcts = MCTSAlpha(state.clone(), model)
    root = mcts.run(simulations)
    action = root.select_action(temperature)
    return action


def random_move(state):
    return random.choice(state.legal_actions())


def evaluate_vs_random(model, games=50, simulations=100, temperature=0):
    results = defaultdict(int)

    for game_idx in range(games):
        state = game.new_initial_state()
        model_player = game_idx % 2  # Alternate who goes first

        while not state.is_terminal():
            if state.current_player() == model_player:
                action = model_move(state, model, simulations, temperature)
            else:
                action = random_move(state)
            state.apply_action(action)

        # Record result (draws returned as 0.0)
        reward = terminal_reward_for_player(state, model_player)
        if reward > 0:
            results["wins"] += 1
        elif reward < 0:
            results["losses"] += 1
        else:
            results["draws"] += 1

    win_rate = results["wins"] / games
    return win_rate, results

def evaluate_vs_pure_mcts(model, games=30, model_sims=100, mcts_sims=500, temperature=0, rollout_count=8):
    results = defaultdict(int)

    evaluator = RandomRolloutEvaluator(rollout_count)
    mcts_bot = MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=mcts_sims,
        evaluator=evaluator,
        solve=False,
        random_state=random.Random(42),
    )

    for game_idx in range(games):
        state = game.new_initial_state()
        model_player = game_idx % 2

        while not state.is_terminal():
            if state.current_player() == model_player:
                action = model_move(state, model, model_sims, temperature)
            else:
                action = mcts_bot.step(state)
            state.apply_action(action)

        reward = terminal_reward_for_player(state, model_player)
        if reward > 0:
            results["wins"] += 1
        elif reward < 0:
            results["losses"] += 1
        else:
            results["draws"] += 1

    win_rate = results["wins"] / games
    return win_rate, results
