import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import OBS_CH

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = x + r
        return F.relu(x, inplace=True)

class PolicyValueNet(nn.Module):
    """
    AlphaZero-style net for square boards (TTT 3×3, Hex 5×5).
    - board_size: H==W (3 for TTT, 5 for Hex-5×5)
    - action_mask_bool: [B, H*W] boolean (True = legal)
    """
    def __init__(self, board_size: int = 5, channels: int = 64, in_ch: int = OBS_CH, num_res_blocks: int = 4):
        super().__init__()
        self.board_size = board_size
        self.n_actions  = board_size * board_size

        # Stem
        self.input_block = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        # Policy head: keep spatial map → per-cell logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.Flatten(),  # -> [B, H*W]
        )

        # Value head: GAP over full channels → MLP → tanh
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden_v = max(32, channels // 2)
        self.value_head = nn.Sequential(
            nn.Linear(channels, hidden_v, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_v, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, action_mask_bool: torch.Tensor | None = None):
        """
        obs: [B, C, H, W] with H=W=board_size
        action_mask_bool: [B, H*W] (True=legal)
        """
        B, _, H, W = obs.shape
        assert H == self.board_size and W == self.board_size, "obs size != board_size"

        x = self.input_block(obs)
        x = self.res_blocks(x)

        # Policy logits
        logits = self.policy_head(x)  # [B, H*W]

        # --- simple masking, as requested ---
        if action_mask_bool is not None:
            logits = logits.masked_fill(~action_mask_bool, -1e9)

        dist = Categorical(logits=logits)

        # Value
        g = self.gap(x).flatten(1)          # [B, channels]
        value = self.value_head(g).squeeze(-1)  # [B]
        return dist, value

    @torch.no_grad()
    def play(self, obs: torch.Tensor, action_mask_bool: torch.Tensor):
        dist, value = self.forward(obs, action_mask_bool)
        return dist.probs, value
