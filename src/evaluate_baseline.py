"""
Baseline Evaluation — Mahalanobis OoD Score + AUROC + t-SNE

실행:
  python src/evaluate_baseline.py --data_root data/pilot --config configs/baseline.yaml

출력:
  - 콘솔: AUROC
  - outputs/baseline_tsne.png
  - outputs/baseline_scores.csv
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import StudyDataset
from models import MotionBERTExtractor
from utils import compute_normality_score, mahalanobis_diag


def evaluate(cfg: dict, data_root: str, ckpt_dir: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(ckpt_dir)

    backbone = MotionBERTExtractor(
        checkpoint_path=cfg["model"]["motionbert_checkpoint"],
        freeze=False,   # eval 시에는 freeze 파라미터 불필요, 그냥 eval()로 고정
    ).to(device)
    backbone.eval()

    stats = np.load(ckpt_dir / "stats.npz")
    mu, std = stats["mu"], stats["std"]

    test_ds = StudyDataset(data_root, cfg, split="test", normal_only=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []

    with torch.no_grad():
        for clips, binary_labels, _ in loader:
            clips = clips.to(device)             # (B, T, J, 3)
            x = backbone(clips)                  # (B, 512)
            x_np = x.cpu().numpy()

            mahal = mahalanobis_diag(x_np, mu, std)                 # (B,)
            # Baseline: Prompt Score 없음 → ood_score만 normality로 사용
            norm_score = np.exp(-cfg["inference"]["mahal_gamma"] * mahal)  # (B,)

            all_scores.extend(norm_score.tolist())
            all_labels.extend(binary_labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)  # 0=normal, 1=OOD

    auroc = roc_auc_score(all_labels, -all_scores)  # 높은 normality = normal
    print(f"[RESULT] AUROC = {auroc:.4f}")

    # t-SNE
    _plot_tsne(cfg, data_root, backbone, device)

    # CSV 저장
    _save_csv(all_scores, all_labels)

    return auroc


def _plot_tsne(cfg, data_root, backbone, device):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] sklearn/matplotlib 없음, t-SNE 생략")
        return

    test_ds = StudyDataset(data_root, cfg, split="test", normal_only=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for clips, binary, _ in loader:
            x = backbone(clips.to(device)).cpu().numpy()
            feats.append(x)
            labels.extend(binary.tolist())

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels) - 1))
    emb = tsne.fit_transform(feats)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                c="steelblue", label="Normal", alpha=0.6, s=20)
    plt.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                c="crimson", label="OOD", alpha=0.6, s=20)
    plt.legend()
    plt.title("t-SNE: Baseline Feature Space (Normal vs OOD)")
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_tsne.png", dpi=150)
    print(f"[DONE] t-SNE → outputs/baseline_tsne.png")
    plt.close()


def _save_csv(scores, labels):
    import csv
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "baseline_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_idx", "normality_score", "label"])
        for i, (s, l) in enumerate(zip(scores, labels)):
            writer.writerow([i, f"{s:.4f}", l])
    print(f"[DONE] Scores → outputs/baseline_scores.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.data_root, args.ckpt_dir)


if __name__ == "__main__":
    main()
