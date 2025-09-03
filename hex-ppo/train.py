import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mlflow
from mlflow.tracking import MlflowClient

import torch
import torch.nn as nn
import torch.optim as optim

from buffer import Buffer
from ppo import update
from eval import evaluate_vs_pure_mcts, evaluate_vs_random
import config 
from rollout import collect_rollouts
from model import PolicyValueNet


mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "hex-ppo"
best_score = 0.0

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

    # breathing room above bars
    ymax = max(1, max(counts))
    ax.set_ylim(0, ymax * 1.25)
    ax.margins(x=0.05)

    # labels on bars (count + optional %)
    for bar, count in zip(bars, counts):
        pct = (count / games) if games else 0.0
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + ymax * 0.05,
                f"{count}\n({pct:.0%})" if games else f"{count}",
                ha="center", va="bottom", fontsize=9)

    # tidy styling
    ax.set_title(f"{tag.replace('_',' ').upper()} · Step {step}", fontsize=11, pad=6)
    ax.set_xlabel("Outcome", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # compact legend
    legend_handles = [Patch(facecolor=c, edgecolor="#333", label=o) for o, c in zip(outcomes, colors)]
    ax.legend(handles=legend_handles, title="Outcome",
              fontsize=8, title_fontsize=9, loc="upper right", frameon=True)

    plt.tight_layout(pad=0.4)
    mlflow.log_figure(fig, f"hist/{step}/{tag}_outcomes_compact_step_{step}.png")
    plt.close(fig)
    
    
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
    
    
    
def main():
    model = PolicyValueNet().to(config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    # warmup_epochs = 30
    # warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.EPOCHS - warmup_epochs))
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    
    buffer = Buffer(size=config.N_STEPS)
    p_bar = tqdm(range(config.EPOCHS), desc="Training")
    
    with mlflow.start_run(experiment_id=experiment_id):
        model.train(True)
        
        for idx in p_bar:
            collect_rollouts(model, buffer)
            data = buffer.get()
            loss, info = update(optimizer, model, data)
            log_data = dict(
                    **info, 
                    loss=loss,
                    learning_rate=optimizer.param_groups[0]["lr"]
                )
            mlflow.log_metrics(
                log_data,
                step=idx
            )
            p_bar.set_postfix(
                dict(
                    **info, 
                    loss=loss,
                    learning_rate=optimizer.param_groups[0]["lr"]
                )
            )
            if idx % config.EVAL_EPOCH == 0 and idx != 0:
                eval(model, idx)
            
            # scheduler.step()
                  
if __name__ == "__main__":
    # print("HELLO")
    main()
  