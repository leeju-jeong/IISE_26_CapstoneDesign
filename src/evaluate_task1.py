"""
Task 1-9/10: MotionBERT-only OOD evaluation (no text / adapter).

입력: checkpoints/.../stats_x.npz
출력: outputs/task1_scores.csv, task1_ood_hist.png, task1_tsne.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
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


def evaluate_task1(cfg: dict, data_root: str) -> None:
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    stats_path = ckpt_dir / "stats_x.npz"
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    inf = cfg.get("inference", {})
    eps = inf.get("stats_eps", 1e-8)
    gamma_x = inf.get("gamma_x", 1.0)
    feat_dim = cfg["model"]["feature_dim"]

    stats = load_stats_x(stats_path)
    mu_x, std_x = stats["mu_x"], stats["std_x"]

    backbone = build_motionbert_from_cfg(cfg).to(device)
    backbone.eval()

    ds = build_dataset(data_root, cfg, normal_only=False)
    if len(ds) == 0:
        raise RuntimeError("clip이 0개입니다. data/people1 라벨·pkl을 확인하세요.")

    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_x = collect_backbone_features(backbone, loader, device)

    mahal = standardized_l2(all_x, mu_x, std_x, eps)
    score_x = exp_mahal_score(mahal, gamma_x, feat_dim)
    anomaly = 1.0 - score_x

    labels = []
    for i in range(len(ds)):
        _, binary, _ = ds[i]
        labels.append(int(binary))
    labels = np.array(labels)

    n_normal = int((labels == 0).sum())
    n_ood = int((labels == 1).sum())
    print(f"[INFO] clips={len(labels)} normal={n_normal} ood={n_ood}")

    rows = []
    for i, meta in enumerate(ds.meta):
        rows.append({
            "video_id": meta.get("video_id", ""),
            "clip_start_sec": meta.get("clip_start_sec", ""),
            "clip_end_sec": meta.get("clip_end_sec", ""),
            "label": meta.get("label", ds.labels[i]),
            "sublabel": meta.get("sublabel", ""),
            "mahal_x": float(mahal[i]),
            "score_x": float(score_x[i]),
            "anomaly_score": float(anomaly[i]),
            "binary_ood": int(labels[i]),
        })

    csv_path = out_dir / "task1_scores.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[DONE] Scores → {csv_path}")

    if n_ood > 0 and n_normal > 0:
        auroc = roc_auc_score(labels, anomaly)
        auprc = average_precision_score(labels, anomaly)
        print(f"[METRIC] AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        summary = out_dir / "task1_eval_summary.txt"
        with open(summary, "w") as f:
            f.write(f"clips: {len(labels)}\n")
            f.write(f"normal: {n_normal}\n")
            f.write(f"ood: {n_ood}\n")
            f.write(f"AUROC: {auroc:.4f}\n")
            f.write(f"AUPRC: {auprc:.4f}\n")
            f.write(f"score_x normal mean: {score_x[labels==0].mean():.4f}\n")
            f.write(f"score_x ood mean: {score_x[labels==1].mean():.4f}\n")
        print(f"[DONE] Summary → {summary}")
    else:
        print("[WARN] normal 또는 ood clip이 없어 AUROC 계산 생략")

    _plot_ood_hist(score_x, labels, out_dir)
    plot_tsne_by_sublabel(
        all_x, sublabels_from_dataset(ds),
        out_dir / "task1_tsne.png",
        title="Task1: t-SNE by sublabel (MotionBERT x)",
    )


def _plot_ood_hist(score_x, labels, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing, skip histogram")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(score_x[labels == 0], bins=30, alpha=0.6, label="Normal (1-5)", color="steelblue")
    if (labels == 1).any():
        plt.hist(score_x[labels == 1], bins=30, alpha=0.6, label="OOD (6)", color="crimson")
    plt.xlabel("score_x")
    plt.ylabel("Count")
    plt.title("Task1: score_x — normal vs OOD")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "task1_ood_hist.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[DONE] Histogram → {path}")


def main():
    parser = argparse.ArgumentParser(description="Task1 OOD evaluation (score_x only)")
    parser.add_argument("--data_root", type=str, default="data/people1")
    parser.add_argument("--config", type=str, default="configs/people1.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate_task1(cfg, args.data_root)


if __name__ == "__main__":
    main()
