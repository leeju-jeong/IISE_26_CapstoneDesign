"""
Training pipeline:
  [5] 정상 train → μ_x, σ_x
  [6] Adapter 학습 (align + preserve)
  [7] 정상 train → μ_z, σ_z
  저장: adapter.pth, stats.npz
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StudyDataset
from losses import adapter_training_loss
from models import MLPAdapter, build_motionbert_from_cfg
from utils import compute_stats, encode_text_prompts, save_stats


@torch.no_grad()
def collect_backbone_features(backbone, loader, device) -> np.ndarray:
    feats = []
    for clips, _, _ in loader:
        clips = clips.to(device)
        feats.append(backbone(clips).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def collect_adapter_embeddings(backbone, adapter, loader, device) -> np.ndarray:
    zs = []
    for clips, _, _ in loader:
        clips = clips.to(device)
        x = backbone(clips)
        z = adapter(x)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


def train(cfg: dict, data_root: str):
    device = cfg["training"]["device"]
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        cfg["model"]["clip_model"], pretrained=cfg["model"]["clip_pretrained"],
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"]["clip_model"])

    text_embeds = encode_text_prompts(
        cfg["training"]["normal_prompts"], clip_model, tokenizer, device,
    )
    text_avg = F.normalize(text_embeds.mean(dim=0, keepdim=True), dim=-1)

    ood_prompts = cfg["training"].get("ood_prompts") or []
    ood_embeds = None
    if ood_prompts:
        ood_embeds = encode_text_prompts(ood_prompts, clip_model, tokenizer, device)

    backbone = build_motionbert_from_cfg(cfg).to(device)
    backbone.eval()

    adapter = MLPAdapter(
        feature_dim=cfg["model"]["feature_dim"],
        hidden_dim=cfg["model"]["adapter_hidden"],
        clip_dim=cfg["model"]["clip_dim"],
        dropout=cfg["model"].get("adapter_dropout", 0.1),
    ).to(device)

    train_ds = StudyDataset(data_root, cfg, split="train", normal_only=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    stats_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    # [5] μ_x, σ_x
    print("[INFO] Computing μ_x, σ_x on normal train clips...")
    all_x = collect_backbone_features(backbone, stats_loader, device)
    mu_x, std_x = compute_stats(all_x)
    print(f"  x stats: {all_x.shape[0]} samples")

    preserve_w = cfg["training"].get("preserve_weight", 0.1)
    ood_w = cfg["training"].get("ood_loss_weight", 0.0)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_loss = float("inf")

    # [6] Adapter 학습
    for epoch in range(cfg["training"]["epochs"]):
        adapter.train()
        epoch_loss = 0.0

        for clips, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            clips = clips.to(device)
            with torch.no_grad():
                x = backbone(clips)
            z = adapter(x)
            loss, _ = adapter_training_loss(
                z, x, text_avg, preserve_w, ood_embeds, ood_w,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}] loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(adapter.state_dict(), ckpt_dir / "adapter.pth")

    print(f"[DONE] Best loss: {best_loss:.4f} → {ckpt_dir / 'adapter.pth'}")

    # [7] μ_z, σ_z
    print("[INFO] Computing μ_z, σ_z on normal train clips...")
    adapter.load_state_dict(
        torch.load(ckpt_dir / "adapter.pth", map_location=device, weights_only=True),
    )
    adapter.eval()
    all_z = collect_adapter_embeddings(backbone, adapter, stats_loader, device)
    mu_z, std_z = compute_stats(all_z)

    save_stats(ckpt_dir / "stats.npz", mu_x, std_x, mu_z, std_z)
    print(f"[DONE] stats.npz → mu_x/std_x/mu_z/std_z shape {mu_x.shape}")


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
