import random
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from open_spiel.python.algorithms.mcts import RandomRolloutEvaluator, MCTSBot
import pyspiel

from torch.distributions import Categorical

from mcts import MCTS
from utils import get_mask, get_obs, get_policy_model


game = pyspiel.load_game("hex(board_size=5)")
evaluator = RandomRolloutEvaluator(n_rollouts=8)
alpha_zero_model = get_policy_model("./weights/ppo_alphazero.pt")
ppo_model = get_policy_model("./weights/ppo.pt")


def play_ppo(state):
    obs  = get_obs(state)   
    mask = get_mask(state)  

    probs, _ = ppo_model.play(obs, mask)  

    dist = Categorical(probs=probs)
    action = dist.sample().item()

    return int(action)

def play_alpha_zero(state, simulations):
    mcts = MCTS(state.clone(), alpha_zero_model)
    root = mcts.run(simulations)
    action = root.select_action(0)
    
    return action

def play_mcts(state, simulations):
    mcts_bot = MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=simulations,
        evaluator=evaluator,
        solve=False,
        random_state=random.Random(42),
    )
    
    action = mcts_bot.step(state)
    
    return action 


models = [
    ("PPO", play_ppo),
    ("AlphaZero(20)", lambda s: play_alpha_zero(s, 20)),
    ("AlphaZero(60)", lambda s: play_alpha_zero(s, 60)),
    ("AlphaZero(100)", lambda s: play_alpha_zero(s, 100)),
    ("MCTS(20)", lambda s: play_mcts(s, 20)),
    ("MCTS(60)", lambda s: play_mcts(s, 60)),
    ("MCTS(100)", lambda s: play_mcts(s, 100)),  # Fixed: was using 60 simulations
    ("Random", lambda s: random.choice(s.legal_actions())),
]


INITIAL_ELO = 1400
K = 32

# Game
game = pyspiel.load_game("hex(board_size=5)")

# Elo tracker
elo = {name: INITIAL_ELO for name, _ in models}
results = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})

def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def update_elo(r1, r2, score1):
    score2 = 1 - score1
    e1 = expected_score(r1, r2)
    e2 = expected_score(r2, r1)
    return r1 + K * (score1 - e1), r2 + K * (score2 - e2)

def play_match(model1, name1, model2, name2, num_games=10):
    for game_idx in range(num_games):
        state = game.new_initial_state()
        players = [model1, model2] if game_idx % 2 == 0 else [model2, model1]
        names = [name1, name2] if game_idx % 2 == 0 else [name2, name1]

        while not state.is_terminal():
            move = players[state.current_player()](state)
            state.apply_action(move)

        reward0, reward1 = state.returns()
        
        if game_idx % 2 == 0:
            name1_won = 1 if reward0 > reward1 else 0
            name2_won = 1 - name1_won
        else:
            # name2 is player 0, name1 is player 1  
            name1_won = 1 if reward1 > reward0 else 0
            name2_won = 1 - name1_won

        # Update Elo using actual win/loss (0 or 1)
        elo1, elo2 = elo[name1], elo[name2]
        new_elo1, new_elo2 = update_elo(elo1, elo2, name1_won)
        elo[name1], elo[name2] = new_elo1, new_elo2

        # Track win/loss
        results[name1]["wins"] += name1_won
        results[name1]["losses"] += name2_won
        results[name1]["games"] += 1

        results[name2]["wins"] += name2_won
        results[name2]["losses"] += name1_won
        results[name2]["games"] += 1

# Run matches
for i in tqdm(range(len(models))):
    for j in range(i+1, len(models)):
        name1, model1 = models[i]
        name2, model2 = models[j]
        if "MCTS" in name1 and "MCTS" in name2:
            continue  # Skip MCTS vs MCTS
        play_match(model1, name1, model2, name2, num_games=10)

# Build ranking table
data = []
for name in elo:
    wins = results[name]["wins"]
    losses = results[name]["losses"]
    games = results[name]["games"]
    data.append({
        "Model": name,
        "Win Rate": round(wins / games, 3) if games > 0 else 0,
        "Games": games,
        "Elo Rating": round(elo[name])
    })

df = pd.DataFrame(data)
df = df.sort_values(by="Elo Rating", ascending=False)
df.insert(0, "Rank", range(1, len(df)+1))
print(df)