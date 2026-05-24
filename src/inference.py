"""
Inference: video → normality / anomaly scores

출력: outputs/scores.csv, outputs/score_timeline.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml

from dataset import build_clips
from models import MLPAdapter, build_motionbert_from_cfg
from pose_extractor import extract_skeleton
from utils import (
    compute_clip_scores,
    encode_text_prompts,
    load_stats,
    temporal_smooth,
)


def run_inference(video_path: str, cfg: dict, ckpt_dir: str) -> np.ndarray:
    device = cfg["training"]["device"]
    ckpt_dir = Path(ckpt_dir)
    inf = cfg["inference"]
    feat_dim = cfg["model"]["feature_dim"]

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

    print(f"[INFO] Extracting skeleton: {video_path}")
    skeleton, _, _ = extract_skeleton(video_path, cfg)
    if skeleton is None:
        print("[ERROR] Skeleton extraction failed.")
        return np.array([])

    clips = build_clips(skeleton, cfg)
    if not clips:
        print("[ERROR] No clips.")
        return np.array([])

    print(f"[INFO] {len(clips)} clips, fusion={inf['fusion']}")

    xs, zs = [], []
    with torch.no_grad():
        for clip in clips:
            t = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).to(device)
            x = backbone(t)
            z = adapter(x)
            xs.append(x.cpu().numpy())
            zs.append(z.cpu().numpy())

    all_x = np.concatenate(xs, axis=0)
    all_z = np.concatenate(zs, axis=0)

    scores = compute_clip_scores(
        all_x, all_z, text_avg,
        stats["mu_x"], stats["std_x"],
        stats["mu_z"], stats["std_z"],
        inf["gamma_x"], inf["gamma_z"],
        inf["fusion"], feat_dim, inf.get("stats_eps", 1e-8),
    )

    smooth_n = inf.get("smooth_clips", 3)
    norm_smooth = temporal_smooth(scores["normality"], window=smooth_n)
    anom_smooth = 1.0 - norm_smooth

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stride_sec = cfg["data"]["stride_sec"]
    times = [i * stride_sec for i in range(len(norm_smooth))]

    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "clip_idx", "start_sec",
            "score_x", "score_z", "score_text",
            "normality", "anomaly",
            "normality_smooth", "anomaly_smooth",
        ])
        for i, t in enumerate(times):
            w.writerow([
                i, t,
                f"{scores['score_x'][i]:.4f}",
                f"{scores['score_z'][i]:.4f}",
                f"{scores['score_text'][i]:.4f}",
                f"{scores['normality'][i]:.4f}",
                f"{scores['anomaly'][i]:.4f}",
                f"{norm_smooth[i]:.4f}",
                f"{anom_smooth[i]:.4f}",
            ])
    print(f"[DONE] CSV → {csv_path}")

    _plot_timeline(times, scores["normality"], norm_smooth, out_dir)
    return norm_smooth


def _plot_timeline(times, raw, smooth, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(times, raw, alpha=0.4, color="steelblue", label="Raw")
    plt.plot(times, smooth, color="steelblue", linewidth=2, label=f"Smooth ({len(smooth)} clips)")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Time (sec)")
    plt.ylabel("Normality score")
    plt.title("Study Normality Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_timeline.png", dpi=150)
    print(f"[DONE] Timeline → {out_dir / 'score_timeline.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_inference(args.video, cfg, args.ckpt_dir)


if __name__ == "__main__":
    main()
