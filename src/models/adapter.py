"""
MLP Adapter: skeleton feature → CLIP text embedding space

Input : (B, feature_dim)   e.g. 256
Output: (B, clip_dim)      e.g. 512, L2 normalized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdapter(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 256,
                 clip_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)
