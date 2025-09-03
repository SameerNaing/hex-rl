import torch


BOARD_SIZE  = 5
OBS_CH       = 9      
N_ACTIONS    = BOARD_SIZE * BOARD_SIZE 

# --- Self-play / MCTS ---
SELF_PLAY_GAMES   = 20
MCTS_SIMULATIONS  = 120
DIRICHLET_ALPHA   = 0.8     # root noise
DIRICHLET_EPS     = 0.2
TEMP_MOVES        = 3        # τ=1.0 for first 6 plies, then 0

PRETRAINED        = False

# --- Training ---
REPLAY_BUFFER_SIZE = 30_000
BATCH_SIZE         = 256
TRAINING_STEPS     = 20       # SGD updates per epoch
EPOCHS             = 100
EVAL_EPOCH         = 5
WEIGHT_DECAY       = 1e-4
LEARNING_RATE      = 1e-4

# --- Misc ---
SEED = 42
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
