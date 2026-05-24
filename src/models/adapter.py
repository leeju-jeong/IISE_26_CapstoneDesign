"""
MLP Adapter: MotionBERT feature → CLIP text embedding space

Linear(512 → 512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512 → 512) → L2 norm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        clip_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
