import os 

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim


from eval import evaluate_vs_random, evaluate_vs_pure_mcts
from model import PolicyValueNet
from buffer import ReplayBuffer
from self_play import self_play_game
import config

mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "test-alpha-zero"
best_score = 0.0


# Ensure experiment exists
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id
    

def log_outcome_metrics_with_histogram(tag: str, results: dict, games: int, step: int):
    # --- metrics ---
    w = int(results.get("wins", 0))
    d = int(results.get("draws", 0))
    l = int(results.get("losses", 0))


    # --- compact fig ---
    outcomes = ["Wins", "Draws", "Losses"]
    counts   = [w, d, l]
    colors   = ["#2E8B57", "#FF8C00", "#DC143C"]  # green / orange / red

    # small but crisp
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    bars = ax.bar(outcomes, counts, color=colors, alpha=0.85,
                  width=0.65, edgecolor="#333", linewidth=0.6)

    ymax = max(1, max(counts))
    ax.set_ylim(0, ymax * 1.25)
    ax.margins(x=0.05)

    for bar, count in zip(bars, counts):
        pct = (count / games) if games else 0.0
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + ymax * 0.05,
                f"{count}\n({pct:.0%})" if games else f"{count}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{tag.replace('_',' ').upper()} · Step {step}", fontsize=11, pad=6)
    ax.set_xlabel("Outcome", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_handles = [Patch(facecolor=c, edgecolor="#333", label=o) for o, c in zip(outcomes, colors)]
    ax.legend(handles=legend_handles, title="Outcome",
              fontsize=8, title_fontsize=9, loc="upper right", frameon=True)

    plt.tight_layout(pad=0.4)
    mlflow.log_figure(fig, f"hist/{step}/{tag}_outcomes_compact_step_{step}.png")
    plt.close(fig)
    
def train(model, optimizer, buffer):
    obs, masks, action_probs, values = buffer.sample(config.BATCH_SIZE)
    obs = torch.tensor(np.array(obs), dtype=torch.float32).to(config.DEVICE)
    mask = torch.tensor(np.array(masks), dtype=torch.bool).to(config.DEVICE)
    target_p = torch.tensor(np.array(action_probs), dtype=torch.float32).to(config.DEVICE)
    target_v = torch.tensor(np.array(values), dtype=torch.float32).to(config.DEVICE)
    
    dist, pred_v = model(obs, mask)
    
    
    with torch.no_grad():
        probs = dist.probs
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        kl = kl_loss(torch.log(probs + 1e-8), target_p).item()
        kl_dir = kl_loss(torch.log(target_p + 1e-8), probs).item()
        ent = dist.entropy().mean().item()
    
    logp = torch.log(dist.probs + 1e-8)
    loss_p = -(target_p * logp).sum(dim=1).mean()
    
    loss_v = nn.MSELoss()(pred_v, target_v)
    loss = loss_p + loss_v
    
    optimizer.zero_grad()
    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item(), dict(ent=ent, kl=kl, kl_dir=kl_dir)


def eval(model, idx):
    global best_score
    
    model.eval()
    
    random_wr, random_results = evaluate_vs_random(model, games=40, simulations=100)
    log_outcome_metrics_with_histogram("random", random_results, games=40, step=idx)
   
    mcts_wr_10, mcts10_results = evaluate_vs_pure_mcts(model, games=20, model_sims=30, mcts_sims=10)
    mcts_wr_30, mcts30_results = evaluate_vs_pure_mcts(model, games=20, model_sims=30, mcts_sims=30)
    mcts_wr_60, mcts60_results = evaluate_vs_pure_mcts(model, games=20, model_sims=30, mcts_sims=60)
   
    log_outcome_metrics_with_histogram("mcts_10", mcts10_results, games=20, step=idx)
    log_outcome_metrics_with_histogram("mcts_30", mcts30_results, games=20, step=idx)
    log_outcome_metrics_with_histogram("mcts_60", mcts60_results, games=20, step=idx)
        
    
    mlflow.log_metric("eval_vs_random", random_wr, step=idx)
    mlflow.log_metric("eval_vs_mcts_10", mcts_wr_10, step=idx)
    mlflow.log_metric("eval_vs_mcts_30", mcts_wr_30, step=idx)
    mlflow.log_metric("eval_vs_mcts_60", mcts_wr_60, step=idx)

    composite_score = (
        random_wr * 0.3 +           # 30% weight - should be close to 1.0
        mcts_wr_10 * 0.2 +          # 20% weight - vs weak MCTS
        mcts_wr_30 * 0.3 +          # 30% weight - vs medium MCTS  
        mcts_wr_60 * 0.2            # 20% weight - vs strong MCTS
    )
    
    mlflow.log_metric("composite_score", composite_score, step=idx)
    if composite_score > best_score:
        best_score = composite_score
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/best_model.pt")
    
def load_weights(model):
    if config.PRETRAINED:
        checkpoint_path = "./checkpoints/ppo_weights.pt"
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained weights from {checkpoint_path} ...")
            weights = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(weights)
        else:
            print(f"[Warning] Pretrained flag is set but no checkpoint found at {checkpoint_path}")
            
def main():
    model = PolicyValueNet().to(config.DEVICE)
    
    load_weights(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    
    
    warmup_epochs = 20
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.EPOCHS - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    

    
    buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    p_bar = tqdm(range(config.EPOCHS), desc="Training")

    with mlflow.start_run(experiment_id=experiment_id):
        model.train(True)
        
        for idx in p_bar:
            for _ in range(config.SELF_PLAY_GAMES):
                self_play_game(model, config.MCTS_SIMULATIONS, buffer)
            
            if len(buffer) < config.BATCH_SIZE:
                continue
            
            for step in range(config.TRAINING_STEPS):
                loss, info = train(model, optimizer, buffer) 
   
                d = {
                    "loss":loss, 
                    "entropy": info["ent"],
                    "kl": info["kl"],
                    "kl_dir": info["kl_dir"]    
                }
                p_bar.set_postfix(d)
                
                mlflow.log_metrics(d, step=idx * config.TRAINING_STEPS + step)

                mlflow.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=idx * config.TRAINING_STEPS + step)
                
            
            if idx % config.EVAL_EPOCH == 0 and idx != 0:
                eval(model, idx)
            
            scheduler.step()
                
        
            
            
if __name__ == "__main__":
    main()
  