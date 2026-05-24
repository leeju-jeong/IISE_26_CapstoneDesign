"""
ST-GCN++ Feature Extractor (standalone, PYSKL-compatible architecture)

Graph : COCO 17-joint skeleton, 3-partition spatial configuration
Input : (B, N*J, 3)  N=100 frames, J=17 joints, 3=[x, y, score]
Output: (B, feature_dim)

Architecture:
  - 10-block ST-GCN++ with adaptive graph convolution
  - Multi-scale temporal convolution (MSTCN: k=1,3,5 + max-pool branch)
  - Channel progression: 64×3, 128×4, 256×3
  - Global average pool → linear projection → feature_dim

Pretrained weights:
  If stgcn_ckpt points to a PYSKL checkpoint, pass load_pyskl=True.
  Otherwise the backbone is randomly initialized (fine-tune from scratch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# COCO 17-joint graph
# ──────────────────────────────────────────────────────────────
_N = 17

# Undirected skeleton edges
_COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (0, 5), (0, 6),                            # nose → shoulders
    (5, 6),                                    # shoulder bar
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12),                          # shoulders → hips
    (11, 12),                                  # hip bar
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]

_ROOT = 0  # nose — consistent with PYSKL's COCO graph


def _bfs_dist(edges: list[tuple[int, int]], root: int, n: int) -> list[int]:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = [-1] * n
    dist[root] = 0
    q = [root]
    while q:
        v = q.pop(0)
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _build_adj_3p() -> torch.Tensor:
    """
    3-partition adjacency matrix A of shape (3, 17, 17):
      A[0] — self-connections
      A[1] — centripetal edges (towards root)
      A[2] — centrifugal edges (away from root)
    Row-normalised per partition.
    """
    dist = _bfs_dist(_COCO_EDGES, _ROOT, _N)
    A = np.zeros((3, _N, _N), dtype=np.float32)
    for i in range(_N):
        A[0, i, i] = 1.0
    for i, j in _COCO_EDGES:
        di, dj = dist[i], dist[j]
        if di > dj:          # i farther → centripetal i→j, centrifugal j→i
            A[1, i, j] = 1.0
            A[2, j, i] = 1.0
        elif dj > di:
            A[1, j, i] = 1.0
            A[2, i, j] = 1.0
        else:                # lateral → both in centripetal
            A[1, i, j] = 1.0
            A[1, j, i] = 1.0
    # Row-normalise each partition
    for k in range(3):
        row_sum = A[k].sum(axis=1, keepdims=True)
        A[k] = np.where(row_sum > 0, A[k] / row_sum, 0.0)
    return torch.from_numpy(A)  # (3, 17, 17)


# ──────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────

class _SpatialGCN(nn.Module):
    """
    Adaptive spatial graph convolution with 3-partition adjacency.
    PA is initialised with A and updated during training (gcn_adaptive='init').
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        # A: (3, V, V)
        self.register_buffer("A_base", A)
        self.PA = nn.Parameter(A.clone())          # learnable, init = A
        self.W = nn.Conv2d(in_ch * 3, out_ch, 1)  # K=3 partitions
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        A = self.PA  # (3, V, V)
        # spatial aggregation per partition: einsum → (B, 3, C, T, V)
        x_agg = torch.einsum("bctv,kvw->bkctw", x, A)
        B, K, C, T, V = x_agg.shape
        x_agg = x_agg.reshape(B, K * C, T, V)    # (B, 3C, T, V)
        return self.bn(self.W(x_agg))             # (B, out_ch, T, V)


class _MSTCNBranch(nn.Module):
    """Single temporal branch: pointwise BN-ReLU → depthwise-style temporal conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, (kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MSTCN(nn.Module):
    """
    Multi-scale temporal convolution: 4 branches (k=1, k=3, k=5, max-pool).
    Output channels == input channels (each branch gets channels//4).
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        assert channels % 4 == 0
        ch = channels // 4
        self.b0 = _MSTCNBranch(channels, ch, 1, stride)
        self.b1 = _MSTCNBranch(channels, ch, 3, stride)
        self.b2 = _MSTCNBranch(channels, ch, 5, stride)
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(channels, ch, 1),
            nn.BatchNorm2d(ch),
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x),
                         self.pool_branch(x)], dim=1)
        return self.bn(out)


class _STGCNBlock(nn.Module):
    """One ST-GCN++ block: spatial GCN → MSTCN → residual."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 stride: int = 1, residual: bool = True):
        super().__init__()
        self.gcn = _SpatialGCN(in_ch, out_ch, A)
        self.relu = nn.ReLU(inplace=True)
        self.tcn = _MSTCN(out_ch, stride=stride)

        if not residual:
            self.res = lambda x: 0
        elif in_ch == out_ch and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.gcn(x))
        y = self.tcn(y)
        return self.relu(y + self.res(x))


# ──────────────────────────────────────────────────────────────
# Backbone
# ──────────────────────────────────────────────────────────────

_BLOCK_CFG = [
    # (out_ch, stride)
    (64,  1), (64,  1), (64,  1),
    (128, 2), (128, 1), (128, 1), (128, 1),
    (256, 2), (256, 1), (256, 1),
]


class _STGCNPlusPlusBackbone(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        A = _build_adj_3p()  # (3, 17, 17)
        self.data_bn = nn.BatchNorm1d(in_channels * _N)

        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, stride in _BLOCK_CFG:
            self.blocks.append(
                _STGCNBlock(in_ch, out_ch, A, stride=stride,
                            residual=(in_ch != 0))
            )
            in_ch = out_ch

        self.out_channels = _BLOCK_CFG[-1][0]  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape

        # data BN over (C*V) channels per time step
        x_bn = x.permute(0, 2, 1, 3).reshape(B * T, C * V)  # (B*T, C*V) — not ideal
        # Standard PYSKL approach: BN on (B, C*V, T)
        x_bn = x.permute(0, 1, 3, 2).reshape(B, C * V, T)   # (B, C*V, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.reshape(B, C, V, T).permute(0, 1, 3, 2)    # (B, C, T, V)

        for block in self.blocks:
            x = block(x)

        # Global average pool over T and V
        return x.mean(dim=(2, 3))  # (B, 256)


# ──────────────────────────────────────────────────────────────
# Public extractor
# ──────────────────────────────────────────────────────────────

class STGCNExtractor(nn.Module):
    def __init__(self, feature_dim: int = 256, n_frames: int = 100,
                 n_joints: int = 17, ckpt_path: str = "",
                 freeze: bool = True):
        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints

        self.backbone = _STGCNPlusPlusBackbone(in_channels=3)

        if ckpt_path and Path(ckpt_path).exists():
            _load_pyskl_weights(self.backbone, ckpt_path)
        elif ckpt_path:
            print(f"[WARN] ST-GCN++ ckpt not found: {ckpt_path}  — random init")

        if freeze:
            # Freeze all blocks except the last two (fine-tune last stage)
            freeze_up_to = len(self.backbone.blocks) - 2
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_up_to:
                    for p in block.parameters():
                        p.requires_grad = False

        backbone_dim = self.backbone.out_channels  # 256
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        ) if feature_dim != backbone_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N*J, 3)
        returns: (B, feature_dim)
        """
        B = x.shape[0]
        # (B, N*J, 3) → (B, N, J, 3) → (B, 3, N, J) = (B, C, T, V)
        x = x.reshape(B, self.n_frames, self.n_joints, 3)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 3, T, V)

        feat = self.backbone(x)   # (B, 256)
        return self.proj(feat)    # (B, feature_dim)


# ──────────────────────────────────────────────────────────────
# PYSKL weight loader (best-effort key mapping)
# ──────────────────────────────────────────────────────────────

def _load_pyskl_weights(backbone: _STGCNPlusPlusBackbone,
                        ckpt_path: str) -> None:
    """
    Try to load PYSKL STGCNPlusPlus checkpoint into our backbone.
    PYSKL checkpoints store weights under 'backbone.*' keys inside
    the recognizer state dict.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # Strip 'backbone.' prefix if present
    mapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            mapped[k[len("backbone."):]] = v
        elif not k.startswith("cls_head."):
            mapped[k] = v

    result = backbone.load_state_dict(mapped, strict=False)
    missing = [k for k in result.missing_keys if "PA" not in k]  # PA is ours
    if missing:
        print(f"[WARN] Missing keys when loading ST-GCN++ ckpt: {missing[:5]}...")
    print(f"[INFO] Loaded ST-GCN++ weights from {ckpt_path}")
