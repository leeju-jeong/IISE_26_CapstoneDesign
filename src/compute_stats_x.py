"""
Task 1-8: 정상 clip → MotionBERT x → μ_x, σ_x

Adapter / CLIP 없음. 출력: checkpoints/task1/stats_x.npz
"""
import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from dataset import build_dataset
from models import build_motionbert_from_cfg
from utils import collect_backbone_features, compute_stats, save_stats_x


def compute_stats_x(cfg: dict, data_root: str) -> Path:
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / "stats_x.npz"

    backbone = build_motionbert_from_cfg(cfg).to(device)
    backbone.eval()

    ds = build_dataset(data_root, cfg, normal_only=True)
    if len(ds) == 0:
        raise RuntimeError(
            "clip이 0개입니다. data/skeletons/*.pkl 과 data/labels.csv 를 확인하세요."
        )

    loader = DataLoader(
        ds, batch_size=cfg["training"].get("batch_size", 32), shuffle=False,
    )

    print(f"[INFO] Extracting MotionBERT features from {len(ds)} normal clips...")
    all_x = collect_backbone_features(backbone, loader, device)
    mu_x, std_x = compute_stats(all_x)

    save_stats_x(out_path, mu_x, std_x, all_x=all_x)
    print(f"[DONE] N={all_x.shape[0]}, dim={all_x.shape[1]}")
    print(f"       mu_x/std_x → {out_path}")
    print(f"       std_x min={std_x.min():.6f}, max={std_x.max():.6f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Task1: compute mu_x, std_x")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--config", type=str, default="configs/task1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    compute_stats_x(cfg, args.data_root)


if __name__ == "__main__":
    main()
