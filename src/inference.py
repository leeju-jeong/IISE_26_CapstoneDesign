"""
End-to-end inference: video → Study Normality Score (per clip)

실행:
  python src/inference.py --video path/to/video.mp4 --config configs/default.yaml

출력:
  - 콘솔: clip별 normality score
  - outputs/scores.csv
  - outputs/score_plot.png
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import yaml

from models import MLPAdapter, PointNetExtractor
from pose_extractor import extract_skeleton
from dataset import build_clips
from utils import (compute_normality_score, encode_text_prompts,
                   mahalanobis_diag, temporal_smooth)


def run_inference(video_path: str, cfg: dict, ckpt_dir: str) -> np.ndarray:
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
    text_avg = text_embeds.mean(dim=0, keepdim=True).cpu().numpy()  # (1, 512)

    # ── Extract skeleton ──
    print(f"[INFO] Extracting skeleton from {video_path}...")
    skeleton, _, _ = extract_skeleton(video_path, cfg)
    if skeleton is None:
        print("[ERROR] Skeleton extraction failed.")
        return np.array([])

    # ── Build clips ──
    fps = cfg["data"]["fps"]
    clips = build_clips(
        skeleton, fps,
        cfg["data"]["window_sec"],
        cfg["data"]["stride_sec"],
        cfg["data"]["frame_step"],
    )
    if not clips:
        print("[ERROR] No clips generated.")
        return np.array([])

    print(f"[INFO] {len(clips)} clips generated")

    # ── Inference ──
    scores_raw = []
    with torch.no_grad():
        for clip in clips:
            N, J, D = clip.shape
            points = torch.tensor(clip.reshape(N * J, D),
                                  dtype=torch.float32).unsqueeze(0).to(device)  # (1, N*J, 7)
            x = backbone(points)
            z = adapter(x)

            x_np = x.cpu().numpy()
            z_np = z.cpu().numpy()

            mahal = mahalanobis_diag(x_np, mu, std)
            prompt_sim = (z_np @ text_avg.T).squeeze()
            score = compute_normality_score(
                mahal, np.array([prompt_sim]),
                cfg["inference"]["mahal_gamma"],
                cfg["model"]["feature_dim"],
            )
            scores_raw.append(float(score[0]))

    scores_raw = np.array(scores_raw)

    # ── Temporal smoothing ──
    smooth_window = max(1, cfg["inference"]["smooth_window_sec"] //
                        cfg["data"]["stride_sec"])
    scores_smooth = temporal_smooth(scores_raw, window=smooth_window)

    # ── Save outputs ──
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    stride_sec = cfg["data"]["stride_sec"]
    times = [i * stride_sec for i in range(len(scores_smooth))]

    import csv
    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_idx", "start_sec", "normality_score_raw", "normality_score_smooth"])
        for i, (t, sr, ss) in enumerate(zip(times, scores_raw, scores_smooth)):
            writer.writerow([i, t, f"{sr:.4f}", f"{ss:.4f}"])
    print(f"[DONE] Scores saved → {csv_path}")

    _plot_scores(times, scores_raw, scores_smooth, out_dir)

    return scores_smooth


def _plot_scores(times, raw, smooth, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plt.figure(figsize=(12, 4))
    plt.plot(times, raw, alpha=0.4, color="steelblue", label="Raw")
    plt.plot(times, smooth, color="steelblue", linewidth=2, label="Smoothed")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Time (sec)")
    plt.ylabel("Study Normality Score")
    plt.title("Study Normality Score over Time")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_plot.png", dpi=150)
    print(f"[DONE] Plot saved → {out_dir / 'score_plot.png'}")
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
