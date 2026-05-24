"""
Task 1-10: 정상 분포 sanity check (OOD 없음)

입력: checkpoints/task1/stats_x.npz
출력: outputs/task1_score_hist.png, task1_tsne_normal_only.png,
      task1_features_summary.txt
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dataset import build_dataset
from models import build_motionbert_from_cfg
from plot_task1 import plot_tsne_by_sublabel, sublabels_from_dataset
from utils import (
    collect_backbone_features,
    exp_mahal_score,
    load_stats_x,
    standardized_l2,
)


def verify_stats_x(cfg: dict, data_root: str) -> None:
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    stats_path = ckpt_dir / "stats_x.npz"
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    stats = load_stats_x(stats_path)
    mu_x, std_x = stats["mu_x"], stats["std_x"]
    eps = cfg.get("inference", {}).get("stats_eps", 1e-8)
    gamma_x = cfg.get("inference", {}).get("gamma_x", 1.0)
    feat_dim = cfg["model"]["feature_dim"]

    ds = build_dataset(data_root, cfg, normal_only=True)

    if "all_x" in stats:
        all_x = stats["all_x"]
    else:
        backbone = build_motionbert_from_cfg(cfg).to(device)
        backbone.eval()
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        all_x = collect_backbone_features(backbone, loader, device)

    mahal = standardized_l2(all_x, mu_x, std_x, eps)
    score_x = exp_mahal_score(mahal, gamma_x, feat_dim)

    summary_path = out_dir / "task1_features_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"N clips: {all_x.shape[0]}\n")
        f.write(f"feature dim: {all_x.shape[1]}\n")
        f.write(f"std_x min: {std_x.min():.6f}\n")
        f.write(f"std_x max: {std_x.max():.6f}\n")
        f.write(f"mahal_x mean: {mahal.mean():.4f} std: {mahal.std():.4f}\n")
        f.write(f"mahal_x min: {mahal.min():.4f} max: {mahal.max():.4f}\n")
        f.write(f"score_x mean: {score_x.mean():.4f} std: {score_x.std():.4f}\n")
        f.write(f"score_x min: {score_x.min():.4f} max: {score_x.max():.4f}\n")
    print(f"[DONE] Summary → {summary_path}")

    _plot_hist(score_x, out_dir)
    plot_tsne_by_sublabel(
        all_x, sublabels_from_dataset(ds),
        out_dir / "task1_tsne_normal_only.png",
        title="Task1: t-SNE normal clips by sublabel (MotionBERT x)",
    )


def _plot_hist(score_x, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing, skip histogram")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(score_x, bins=30, color="steelblue", alpha=0.8)
    plt.xlabel("score_x (normal clips)")
    plt.ylabel("Count")
    plt.title("Task1: score_x distribution (normal only)")
    plt.tight_layout()
    path = out_dir / "task1_score_hist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DONE] Histogram → {path}")


def main():
    parser = argparse.ArgumentParser(description="Task1: verify stats_x")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--config", type=str, default="configs/task1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    verify_stats_x(cfg, args.data_root)


if __name__ == "__main__":
    main()
