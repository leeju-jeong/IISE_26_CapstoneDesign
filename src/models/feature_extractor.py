"""
PointNet-style feature extractor (Sato et al. CVPR 2023 구조 기반)

Input : (B, N, 7)  N = frames_per_clip × joints (e.g. 100 × 11 = 1100)
Output: (B, feature_dim)

구조:
  Shared MLP : 7 → 64 → 128 → feature_dim  (Conv1d pointwise)
  Residual blocks × 2
  Global MaxPool → feature vector x ∈ R^feature_dim

freeze_backbone=True  → shared MLP + res_block1 frozen, res_block2 학습
freeze_backbone=False → 전체 학습
"""
import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class PointNetExtractor(nn.Module):
    def __init__(self, in_dim: int = 7, feature_dim: int = 256,
                 freeze_backbone: bool = True):
        super().__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.res_block1 = _ResidualBlock(feature_dim)
        self.res_block2 = _ResidualBlock(feature_dim)  # 항상 학습

        if freeze_backbone:
            for p in self.shared_mlp.parameters():
                p.requires_grad = False
            for p in self.res_block1.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 7) → (B, 7, N)
        x = x.transpose(1, 2)
        x = self.shared_mlp(x)    # (B, feature_dim, N)
        x = self.res_block1(x)    # (B, feature_dim, N)
        x = self.res_block2(x)    # (B, feature_dim, N)
        x = x.max(dim=2).values   # (B, feature_dim) global max pool
        return x
