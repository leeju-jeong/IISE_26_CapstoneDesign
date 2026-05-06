"""
Evaluation script: AUROC + t-SNE

실행:
  python src/evaluate.py --data_root data/pilot --config configs/default.yaml

출력:
  - AUROC (normal vs OOD)
  - t-SNE 시각화 (outputs/tsne.png)
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import StudyDataset
from models import MLPAdapter, PointNetExtractor
from utils import (compute_normality_score, encode_text_prompts,
                   mahalanobis_diag)


def evaluate(cfg: dict, data_root: str, ckpt_dir: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(ckpt_dir)

    # ── Load models ──
    backbone = PointNetExtractor(
        in_dim=cfg["data"]["joint_dim"],
        feature_dim=cfg["model"]["feature_dim"],
        freeze_backbone=False,
    ).to(device)
    backbone.load_state_dict(torch.load(ckpt_dir / "backbone.pth", map_location=device))
    backbone.eval()

    adapter = MLPAdapter(
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["adapter_hidden"],
        clip_dim=cfg["model"]["clip_dim"],
    ).to(device)
    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pth", map_location=device))
    adapter.eval()

    stats = np.load(ckpt_dir / "stats.npz")
    mu, std = stats["mu"], stats["std"]

    # ── CLIP text embeddings ──
    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"]["clip_model"], pretrained=cfg["model"]["clip_pretrained"]
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"]["clip_model"])
    prompts = cfg["training"]["normal_prompts"]
    text_embeds = encode_text_prompts(prompts, clip_model, tokenizer, device)
    text_avg = text_embeds.mean(dim=0, keepdim=True)  # (1, 512)

    # ── Test dataset (normal + OOD) ──
    test_ds = StudyDataset(data_root, cfg, split="test", normal_only=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for points, binary_labels, _ in test_loader:
            points = points.to(device)
            x = backbone(points)                     # (B, feature_dim)
            z = adapter(x)                           # (B, 512)

            x_np = x.cpu().numpy()
            z_np = z.cpu().numpy()
            t_np = text_avg.cpu().numpy()            # (1, 512)

            mahal = mahalanobis_diag(x_np, mu, std)          # (B,)
            prompt_sim = (z_np @ t_np.T).squeeze(-1)          # (B,)
            scores = compute_normality_score(
                mahal, prompt_sim,
                cfg["inference"]["mahal_gamma"],
                cfg["model"]["feature_dim"],
            )
            all_scores.extend(scores.tolist())
            all_labels.extend(binary_labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)  # 0=normal, 1=OOD

    # AUROC: higher normality score → normal (so flip for sklearn)
    auroc = roc_auc_score(all_labels, -all_scores)
    print(f"[RESULT] AUROC = {auroc:.4f}")

    # ── t-SNE ──
    _plot_tsne(cfg, data_root, backbone, device, ckpt_dir)

    return auroc


def _plot_tsne(cfg, data_root, backbone, device, ckpt_dir):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] sklearn/matplotlib not installed, skipping t-SNE")
        return

    test_ds = StudyDataset(data_root, cfg, split="test", normal_only=False)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for points, binary, _ in loader:
            x = backbone(points.to(device)).cpu().numpy()
            feats.append(x)
            labels.extend(binary.tolist())

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb = tsne.fit_transform(feats)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                c="steelblue", label="Normal", alpha=0.6, s=20)
    plt.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                c="crimson", label="OOD", alpha=0.6, s=20)
    plt.legend()
    plt.title("t-SNE: Feature Space (Normal vs OOD)")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne.png", dpi=150)
    print(f"[DONE] t-SNE saved → {out_dir / 'tsne.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.data_root, args.ckpt_dir)


if __name__ == "__main__":
    main()
