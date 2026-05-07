"""
MotionBERT 기반 Feature Extractor (ICCV 2023, pretrained on Human3.6M + NTU60)

Input : (B, N*J, 7)  N=100 프레임, J=11 MediaPipe 상체 관절
Output: (B, feature_dim)

내부 처리:
  1. (B, N, 11, 2+1) 로 reshape → x, y, confidence 추출
  2. MediaPipe 11관절 → H36M 17관절 매핑 (하체 없는 관절은 0 처리)
  3. crop_scale 정규화 → [-1, 1] 범위
  4. MotionBERT DSTformer → (B, N, 17, 512)
  5. mean pool (프레임+관절) → (B, 512)
  6. 선형 투영 → (B, feature_dim)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# MotionBERT 레포 경로
_MB_REPO = Path(__file__).parent.parent.parent / "MotionBERT"
if str(_MB_REPO) not in sys.path:
    sys.path.insert(0, str(_MB_REPO))

from lib.model.DSTformer import DSTformer  # noqa: E402

DEFAULT_CKPT = str(
    Path(__file__).parent.parent.parent / "models" / "motionbert" / "MB_release.bin"
)

# ──────────────────────────────────────────────────────────
# H36M 17관절 인덱스 정의
# ──────────────────────────────────────────────────────────
# 우리 MediaPipe local 인덱스 (J=11):
#   0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
#   5:l_sh  6:r_sh   7:l_el   8:r_el   9:l_wr  10:r_wr

# H36M 17관절:
#   0:Hip  1:RHip  2:RKnee  3:RAnkle
#   4:LHip 5:LKnee 6:LAnkle 7:Spine  8:Thorax
#   9:Nose 10:Head 11:LSh  12:LEl  13:LWr
#   14:RSh  15:REl  16:RWr


def _mediapipe_to_h36m(coords_xy: torch.Tensor,
                       coords_conf: torch.Tensor) -> torch.Tensor:
    """
    coords_xy   : (B, N, 11, 2)   MediaPipe x, y
    coords_conf : (B, N, 11, 1)   confidence

    Returns: (B, N, 17, 3)  H36M 포맷 (x, y, confidence)
             하체 관절은 (0, 0, 0) 으로 채움
    """
    B, N = coords_xy.shape[:2]
    device = coords_xy.device
    h36m = torch.zeros(B, N, 17, 3, device=device)

    sh_mid = (coords_xy[:, :, 5, :] + coords_xy[:, :, 6, :]) / 2  # (B,N,2)
    sh_mid_conf = (coords_conf[:, :, 5] + coords_conf[:, :, 6]) / 2  # (B,N,1)
    head = coords_xy[:, :, 1:5, :].mean(dim=2)  # 눈+귀 평균 (B,N,2)
    head_conf = coords_conf[:, :, 1:5].mean(dim=2)

    def _fill(h36m_idx, xy, conf):
        h36m[:, :, h36m_idx, :2] = xy
        h36m[:, :, h36m_idx, 2:] = conf

    # 상체 관절 매핑
    _fill(9,  coords_xy[:, :, 0],  coords_conf[:, :, 0])   # Nose
    _fill(10, head,                 head_conf)               # Head
    _fill(11, coords_xy[:, :, 5],  coords_conf[:, :, 5])   # LShoulder
    _fill(12, coords_xy[:, :, 7],  coords_conf[:, :, 7])   # LElbow
    _fill(13, coords_xy[:, :, 9],  coords_conf[:, :, 9])   # LWrist
    _fill(14, coords_xy[:, :, 6],  coords_conf[:, :, 6])   # RShoulder
    _fill(15, coords_xy[:, :, 8],  coords_conf[:, :, 8])   # RElbow
    _fill(16, coords_xy[:, :, 10], coords_conf[:, :, 10])  # RWrist

    # 하체 없음 → 어깨 중점으로 채우고 confidence=0 (모델이 무시)
    _fill(8, sh_mid, torch.zeros_like(sh_mid_conf))   # Thorax
    _fill(7, sh_mid, torch.zeros_like(sh_mid_conf))   # Spine
    _fill(0, sh_mid, torch.zeros_like(sh_mid_conf))   # Hip

    return h36m  # (B, N, 17, 3)


def _crop_scale(motion: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    motion: (B, N, 17, 3)  — conf > 0.1 관절만으로 bounding box 정규화
    Returns (B, N, 17, 3) 에서 x,y ∈ [-1, 1] 근사
    """
    B = motion.shape[0]
    out = motion.clone()
    for b in range(B):
        valid = motion[b, :, :, 2] > 0.1   # (N, 17) bool
        pts = motion[b][valid][:, :2]       # (M, 2)
        if pts.shape[0] < 4:
            continue
        xmin, ymin = pts.min(dim=0).values
        xmax, ymax = pts.max(dim=0).values
        ratio = ((xmax - xmin) * (ymax - ymin) + eps).sqrt()
        out[b, :, :, 0] = (motion[b, :, :, 0] - xmin) / ratio
        out[b, :, :, 1] = (motion[b, :, :, 1] - ymin) / ratio
        out[b, :, :, :2][~valid] = 0.0
    return out


class MotionBERTExtractor(nn.Module):
    def __init__(self, feature_dim: int = 512, n_frames: int = 100,
                 n_joints: int = 11, ckpt_path: str = DEFAULT_CKPT,
                 freeze: bool = True):
        super().__init__()
        self.n_frames = n_frames
        self.n_joints = n_joints

        self.backbone = DSTformer(
            dim_in=3, dim_out=3, dim_feat=512, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=2,
            num_joints=17, maxlen=243, att_fuse=True,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k.replace("module.", ""): v for k, v in ckpt["model_pos"].items()}
        self.backbone.load_state_dict(state, strict=True)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 512 → feature_dim 투영 (학습 대상)
        self.proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        ) if feature_dim != 512 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N*J, 7)  — dataset이 넘겨주는 형식
        returns: (B, feature_dim)
        """
        B = x.shape[0]
        clips = x.reshape(B, self.n_frames, self.n_joints, 7)

        xy   = clips[:, :, :, :2]           # (B, N, J, 2)
        conf = clips[:, :, :, 3:4]          # (B, N, J, 1)

        h36m = _mediapipe_to_h36m(xy, conf)      # (B, N, 17, 3)
        h36m = _crop_scale(h36m)                  # (B, N, 17, 3)

        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.backbone.parameters()
        )):
            feat = self.backbone.get_representation(h36m)  # (B, N, 17, 512)

        feat = feat.mean(dim=(1, 2))   # (B, 512) — 프레임+관절 mean pool
        return self.proj(feat)          # (B, feature_dim)
