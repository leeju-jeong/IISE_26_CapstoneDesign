"""
Baseline Training — MotionBERT frozen + μ/Σ 계산만

실행:
  python src/train_baseline.py --data_root data/pilot --config configs/baseline.yaml

완료 후 저장:
  checkpoints/baseline/stats.npz  (μ, σ for Mahalanobis)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StudyDataset
from models import MotionBERTExtractor
from utils import compute_stats


def train(cfg: dict, data_root: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # MotionBERT (frozen)
    backbone = MotionBERTExtractor(
        checkpoint_path=cfg["model"]["motionbert_checkpoint"],
        freeze=cfg["model"]["freeze_backbone"],
    ).to(device)
    backbone.eval()

    # 정상 클립만 로드
    train_ds = StudyDataset(data_root, cfg, split="train", normal_only=True)
    if len(train_ds) == 0:
        print("[ERROR] 학습 데이터가 없습니다. data_root 경로와 splits.yaml을 확인하세요.")
        return

    loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # MotionBERT가 frozen이므로 학습 없이 feature만 추출
    print("[INFO] 정상 클립에서 feature 추출 중...")
    all_features = []
    with torch.no_grad():
        for clips, _, _ in tqdm(loader, desc="Feature extraction"):
            # clips: (B, T, J, 3)
            clips = clips.to(device)
            x = backbone(clips)           # (B, 512)
            all_features.append(x.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)  # (N, 512)
    print(f"[INFO] 총 {all_features.shape[0]}개 클립 feature 추출 완료")

    mu, std = compute_stats(all_features)
    np.savez(ckpt_dir / "stats.npz", mu=mu, std=std)
    print(f"[DONE] stats.npz 저장 → {ckpt_dir}/stats.npz  (mu: {mu.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.data_root)


if __name__ == "__main__":
    main()
