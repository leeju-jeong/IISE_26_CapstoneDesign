"""
MotionBERT (DSTformer) feature extractor.

① clip (B, T, 11, 7) → x,y + valid_mask
② MP11 → H36M17 (하체: coord=0, conf=0, valid_mask=0)
③ bbox norm (valid upper-body)
④ MotionBERT input (B, T, 17, 2)  [x, y]
   → backbone 호환을 위해 내부적으로 (x, y, valid_mask) 3채널 사용
⑤ frozen backbone → x ∈ R^512
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from functools import partial

_MB_ROOT = Path(__file__).resolve().parents[2] / "MotionBERT"
if str(_MB_ROOT) not in sys.path:
    sys.path.insert(0, str(_MB_ROOT))

from lib.model.DSTformer import DSTformer  # noqa: E402

_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4
_LSHO, _RSHO, _LELB, _RELB, _LWR, _RWR = 5, 6, 7, 8, 9, 10

_H36M_NOSE, _H36M_HEAD, _H36M_NECK, _H36M_BELLY = 9, 10, 8, 7
_H36M_LSHO, _H36M_LELB, _H36M_LWR = 11, 12, 13
_H36M_RSHO, _H36M_RELB, _H36M_RWR = 14, 15, 16

_UPPER_H36M = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
_LOWER_H36M = (0, 1, 2, 3, 4, 5, 6)


def mediapipe11_to_h36m(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    clip: (T, 11, 7) or (B, T, 11, 7)
    returns:
      xy   : (..., 17, 2)
      mask : (..., 17)  valid_mask (0/1)
    """
    single = clip.ndim == 3
    if single:
        clip = clip[np.newaxis]

    B, T, _, _ = clip.shape
    xy = np.zeros((B, T, 17, 2), dtype=np.float32)
    mask = np.zeros((B, T, 17), dtype=np.float32)

    x = clip[..., 0]
    y = clip[..., 1]
    c = clip[..., 3]

    xy[:, :, _H36M_NOSE, 0] = x[:, :, _NOSE]
    xy[:, :, _H36M_NOSE, 1] = y[:, :, _NOSE]
    mask[:, :, _H36M_NOSE] = (c[:, :, _NOSE] > 0).astype(np.float32)

    xy[:, :, _H36M_HEAD, 0] = (x[:, :, _LEYE] + x[:, :, _REYE]) * 0.5
    xy[:, :, _H36M_HEAD, 1] = (y[:, :, _LEYE] + y[:, :, _REYE]) * 0.5
    mask[:, :, _H36M_HEAD] = np.minimum(c[:, :, _LEYE], c[:, :, _REYE])

    xy[:, :, _H36M_NECK, 0] = (x[:, :, _LSHO] + x[:, :, _RSHO]) * 0.5
    xy[:, :, _H36M_NECK, 1] = (y[:, :, _LSHO] + y[:, :, _RSHO]) * 0.5
    mask[:, :, _H36M_NECK] = np.minimum(c[:, :, _LSHO], c[:, :, _RSHO])

    for src, dst in [
        (_LSHO, _H36M_LSHO), (_LELB, _H36M_LELB), (_LWR, _H36M_LWR),
        (_RSHO, _H36M_RSHO), (_RELB, _H36M_RELB), (_RWR, _H36M_RWR),
    ]:
        xy[:, :, dst, 0] = x[:, :, src]
        xy[:, :, dst, 1] = y[:, :, src]
        mask[:, :, dst] = (c[:, :, src] > 0).astype(np.float32)

    xy[:, :, _H36M_BELLY] = xy[:, :, _H36M_NECK]
    mask[:, :, _H36M_BELLY] = mask[:, :, _H36M_NECK] * 0.5

    # lower body: coord=0, valid_mask=0 (already zero)

    if single:
        return xy[0], mask[0]
    return xy, mask


def bbox_normalize_upper(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, 17, 2) + mask → bbox normalized xy in [-1, 1]."""
    result = xy.copy()
    T = xy.shape[0]
    for t in range(T):
        valid = mask[t, list(_UPPER_H36M)] > 0
        if valid.sum() < 2:
            continue
        pts = xy[t, list(_UPPER_H36M)][valid]
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        scale = max(xmax - xmin, ymax - ymin)
        if scale < 1e-8:
            continue
        xs = (xmin + xmax - scale) / 2
        ys = (ymin + ymax - scale) / 2
        result[t, :, 0] = (xy[t, :, 0] - xs) / scale
        result[t, :, 1] = (xy[t, :, 1] - ys) / scale
        result[t, :, :2] = (result[t, :, :2] - 0.5) * 2
        result[t, :, :2] = np.clip(result[t, :, :2], -1, 1)
    return result


def clip_to_motionbert_input(clip: np.ndarray) -> np.ndarray:
    """
    (T, 11, 7) → (T, 17, 2) MotionBERT xy input.
  mask는 별도 반환하지 않고 backbone 내부에서 3번째 채널로 사용.
    """
    xy, mask = mediapipe11_to_h36m(clip)
    xy = bbox_normalize_upper(xy, mask)
    return xy


def xy_mask_to_backbone_tensor(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, 17, 2) + (T, 17) → (T, 17, 3) for pretrained DSTformer."""
    return np.concatenate([xy, mask[..., None]], axis=-1).astype(np.float32)


def _load_motionbert_ckpt(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_pos", ckpt)
    mapped = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"[WARN] MotionBERT missing keys: {missing[:5]}...")
    if unexpected:
        print(f"[WARN] MotionBERT unexpected keys: {unexpected[:5]}...")
    print(f"[INFO] Loaded MotionBERT from {ckpt_path}")


class MotionBERTExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        n_frames: int = 50,
        n_joints: int = 11,
        ckpt_path: str = "",
        freeze: bool = True,
        maxlen: int = 243,
        pooling: str = "mean",
        dim_feat: int = 256,
        mlp_ratio: int = 4,
        depth: int = 5,
        num_heads: int = 8,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.feature_dim = feature_dim
        self.pooling = pooling

        self.backbone = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=dim_feat,
            dim_rep=feature_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_joints=17,
            maxlen=maxlen,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            att_fuse=True,
        )

        if ckpt_path and Path(ckpt_path).exists():
            _load_motionbert_ckpt(self.backbone, ckpt_path)
        elif ckpt_path:
            print(f"[WARN] MotionBERT ckpt not found: {ckpt_path}")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def preprocess_clip(self, clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T, 11, 7) → xy (T,17,2), mask (T,17)."""
        xy, mask = mediapipe11_to_h36m(clip)
        xy = bbox_normalize_upper(xy, mask)
        return xy, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 11, 7)
        returns: (B, 512)
        """
        B = x.shape[0]
        if x.dim() == 2:
            x = x.reshape(B, self.n_frames, self.n_joints, 7)

        batch_inp = []
        for i in range(B):
            xy, mask = self.preprocess_clip(x[i].detach().cpu().numpy())
            batch_inp.append(xy_mask_to_backbone_tensor(xy, mask))
        inp = torch.from_numpy(np.stack(batch_inp)).to(x.device)

        if self.backbone.training and all(not p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()

        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            rep = self.backbone.get_representation(inp)

        if self.pooling == "mean":
            return rep.mean(dim=(1, 2))
        return rep.reshape(B, -1, self.feature_dim).mean(dim=1)


def build_motionbert_from_cfg(cfg: dict) -> MotionBERTExtractor:
    """configs/default.yaml model 섹션 → MotionBERTExtractor."""
    m = cfg["model"]
    return MotionBERTExtractor(
        feature_dim=m["feature_dim"],
        n_frames=m["n_frames"],
        n_joints=m["n_joints"],
        ckpt_path=m["motionbert_ckpt"],
        freeze=m.get("freeze_backbone", True),
        maxlen=m.get("motionbert_maxlen", 243),
        pooling=m.get("pooling", "mean"),
        dim_feat=m.get("motionbert_dim_feat", 256),
        mlp_ratio=m.get("motionbert_mlp_ratio", 4),
        depth=m.get("motionbert_depth", 5),
        num_heads=m.get("motionbert_num_heads", 8),
    )
