"""
Training script

실행:
  python src/train.py --data_root data/pilot --config configs/default.yaml

완료 후 저장:
  checkpoints/backbone.pth
  checkpoints/adapter.pth
  checkpoints/stats.npz  (μ, σ for Mahalanobis)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StudyDataset
from losses import contrastive_loss
from models import MLPAdapter, MotionBERTExtractor
from utils import compute_stats, encode_text_prompts


def train(cfg: dict, data_root: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── CLIP text encoder (frozen) ──
    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"]["clip_model"], pretrained=cfg["model"]["clip_pretrained"]
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"]["clip_model"])

    prompts = cfg["training"]["normal_prompts"]
    text_embeds = encode_text_prompts(prompts, clip_model, tokenizer, device)  # (P, 512)
    print(f"[CLIP] text embeddings: {text_embeds.shape}")

    # ── Models ──
    backbone = MotionBERTExtractor(
        feature_dim=cfg["model"]["feature_dim"],
        n_frames=cfg["model"]["n_frames"],
        n_joints=cfg["model"]["n_joints"],
        ckpt_path=cfg["model"]["motionbert_ckpt"],
        freeze=cfg["model"]["freeze_backbone"],
    ).to(device)

    adapter = MLPAdapter(
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["adapter_hidden"],
        clip_dim=cfg["model"]["clip_dim"],
    ).to(device)

    trainable = list(filter(lambda p: p.requires_grad, backbone.parameters()))
    trainable += list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Dataset ──
    train_ds = StudyDataset(data_root, cfg, split="train", normal_only=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )

    temp = cfg["training"]["temperature"]
    best_loss = float("inf")

    # ── Training loop ──
    for epoch in range(cfg["training"]["epochs"]):
        backbone.train()
        adapter.train()
        epoch_loss = 0.0

        for points, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            points = points.to(device)          # (B, N*J, 7)
            x = backbone(points)                # (B, feature_dim)
            z = adapter(x)                      # (B, 512) normalized
            loss = contrastive_loss(z, text_embeds, temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}] loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(backbone.state_dict(), ckpt_dir / "backbone.pth")
            torch.save(adapter.state_dict(), ckpt_dir / "adapter.pth")

    print(f"[DONE] Best loss: {best_loss:.4f} → saved to {ckpt_dir}")

    # ── Compute μ, σ using BEST checkpoint (not last epoch) ──
    print("[INFO] Computing Mahalanobis statistics from training data...")
    backbone.load_state_dict(torch.load(ckpt_dir / "backbone.pth", map_location=device))
    backbone.eval()
    all_features = []
    with torch.no_grad():
        for points, _, _ in DataLoader(train_ds, batch_size=64, shuffle=False):
            points = points.to(device)
            x = backbone(points)
            all_features.append(x.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)  # (N, feature_dim)
    mu, std = compute_stats(all_features)
    np.savez(ckpt_dir / "stats.npz", mu=mu, std=std)
    print(f"[DONE] stats.npz saved → mu: {mu.shape}, std: {std.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.data_root)


if __name__ == "__main__":
    main()
