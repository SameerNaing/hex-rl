import random
from collections import defaultdict
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator, MCTSBot

from env import get_obs, get_mask
from env import game


def terminal_reward_for_player(state, player_id: int) -> float:
    if hasattr(state, "player_reward"): 
        return float(state.player_reward(player_id))
    if state.is_terminal():
        returns = state.returns()  
        return float(returns[player_id])
    return 0.0


def model_move(state, model):
    obs, mask = get_obs(state), get_mask(state)
    action, _, _ = model.play(obs, mask)
    return int(action.item())


def random_move(state):
    return random.choice(state.legal_actions())


def evaluate_vs_random(model, games=50, simulations=100, temperature=0):
    results = defaultdict(int)

    for game_idx in range(games):
        state = game.new_initial_state()
        model_player = game_idx % 2  # Alternate who goes first

        while not state.is_terminal():
            if state.current_player() == model_player:
                action = model_move(state, model)
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
                action = model_move(state, model)
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
