"""
Evaluation: AUROC, AUPRC, t-SNE, score timeline
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import StudyDataset
from models import MLPAdapter, build_motionbert_from_cfg
from utils import (
    compute_clip_scores,
    encode_text_prompts,
    load_stats,
    temporal_smooth,
)


def _load_models(cfg, ckpt_dir, device):
    backbone = build_motionbert_from_cfg(cfg).to(device)
    backbone.eval()

    adapter = MLPAdapter(
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["adapter_hidden"],
        clip_dim=cfg["model"]["clip_dim"],
        dropout=cfg["model"].get("adapter_dropout", 0.1),
    ).to(device)
    adapter.load_state_dict(
        torch.load(ckpt_dir / "adapter.pth", map_location=device, weights_only=True),
    )
    adapter.eval()
    return backbone, adapter


def evaluate(cfg: dict, data_root: str, ckpt_dir: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(ckpt_dir)
    inf = cfg["inference"]
    feat_dim = cfg["model"]["feature_dim"]

    backbone, adapter = _load_models(cfg, ckpt_dir, device)
    stats = load_stats(ckpt_dir / "stats.npz")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"]["clip_model"], pretrained=cfg["model"]["clip_pretrained"],
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"]["clip_model"])
    text_embeds = encode_text_prompts(
        cfg["training"]["normal_prompts"], clip_model, tokenizer, device,
    )
    text_avg = F.normalize(text_embeds.mean(dim=0, keepdim=True), dim=-1).cpu().numpy()

    test_ds = StudyDataset(data_root, cfg, split="test", normal_only=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    all_x, all_z, all_labels = [], [], []

    with torch.no_grad():
        for clips, binary_labels, _ in test_loader:
            clips = clips.to(device)
            x = backbone(clips)
            z = adapter(x)
            all_x.append(x.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_labels.extend(binary_labels.tolist())

    all_x = np.concatenate(all_x, axis=0)
    all_z = np.concatenate(all_z, axis=0)
    all_labels = np.array(all_labels)

    scores = compute_clip_scores(
        all_x, all_z, text_avg,
        stats["mu_x"], stats["std_x"],
        stats["mu_z"], stats["std_z"],
        inf["gamma_x"], inf["gamma_z"],
        inf["fusion"], feat_dim, inf.get("stats_eps", 1e-8),
    )

    anomaly = scores["anomaly"]
    auroc = roc_auc_score(all_labels, anomaly)
    auprc = average_precision_score(all_labels, anomaly)
    print(f"[RESULT] fusion={inf['fusion']}")
    print(f"[RESULT] AUROC  = {auroc:.4f}")
    print(f"[RESULT] AUPRC  = {auprc:.4f}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    _plot_tsne(all_z, all_labels, out_dir)
    _plot_timeline(scores["normality"], all_labels, out_dir)
    _plot_score_hist(scores["normality"], all_labels, out_dir)

    return {"auroc": auroc, "auprc": auprc}


def _plot_tsne(feats, labels, out_dir):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] skip t-SNE")
        return

    n = len(feats)
    if n < 5:
        return
    perp = min(30, n - 1)
    emb = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(feats)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                c="steelblue", label="Normal", alpha=0.6, s=20)
    plt.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                c="crimson", label="OOD", alpha=0.6, s=20)
    plt.legend()
    plt.title("t-SNE: Adapter embedding (z)")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne.png", dpi=150)
    print(f"[DONE] t-SNE → {out_dir / 'tsne.png'}")
    plt.close()


def _plot_timeline(normality, labels, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    idx = np.arange(len(normality))
    plt.figure(figsize=(12, 4))
    plt.plot(idx, normality, color="steelblue", alpha=0.7, label="Normality")
    ood_idx = np.where(labels == 1)[0]
    if len(ood_idx):
        plt.scatter(ood_idx, normality[ood_idx], c="crimson", s=12, label="OOD clip", zorder=3)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Clip index")
    plt.ylabel("Normality score")
    plt.title("Score timeline (test clips)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_timeline.png", dpi=150)
    print(f"[DONE] Timeline → {out_dir / 'score_timeline.png'}")
    plt.close()


def _plot_score_hist(normality, labels, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(normality[labels == 0], bins=30, alpha=0.6, label="Normal", color="steelblue")
    plt.hist(normality[labels == 1], bins=30, alpha=0.6, label="OOD", color="crimson")
    plt.xlabel("Normality score")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Normality distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "score_plot.png", dpi=150)
    print(f"[DONE] Hist → {out_dir / 'score_plot.png'}")
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
