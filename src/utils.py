"""
통계(μ, σ), Mahalanobis L2, score fusion, temporal smoothing, CLIP encoding
"""
from __future__ import annotations

import numpy as np
import torch


def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    std = features.std(axis=0)
    return mu, std


def standardized_l2(
    x: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """||(x - μ) / (σ + eps)||₂  per sample."""
    return np.linalg.norm((x - mu) / (std + eps), axis=-1)


def exp_mahal_score(
    mahal: np.ndarray,
    gamma: float,
    feature_dim: int = 512,
) -> np.ndarray:
    return np.exp(-gamma / np.sqrt(feature_dim) * mahal)


def score_text(cos_sim: np.ndarray) -> np.ndarray:
    return (cos_sim + 1.0) / 2.0


def fuse_normality(
    score_x: np.ndarray,
    score_z: np.ndarray,
    score_text: np.ndarray,
    mode: str = "linear_40_40_20",
) -> np.ndarray:
    if mode == "linear_50_50":
        return 0.5 * score_x + 0.5 * score_z
    if mode == "linear_40_40_20":
        return 0.4 * score_x + 0.4 * score_z + 0.2 * score_text
    if mode == "product":
        return score_x * score_z * score_text
    raise ValueError(f"Unknown fusion mode: {mode}")


def compute_clip_scores(
    x: np.ndarray,
    z: np.ndarray,
    text_avg: np.ndarray,
    mu_x: np.ndarray,
    std_x: np.ndarray,
    mu_z: np.ndarray,
    std_z: np.ndarray,
    gamma_x: float,
    gamma_z: float,
    fusion: str,
    feature_dim: int = 512,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    x, z: (N, D)   text_avg: (1, D) or (D,)
    Returns dict with score_x, score_z, score_text, normality, anomaly.
    """
    if text_avg.ndim == 2:
        text_avg = text_avg.reshape(1, -1)

    mahal_x = standardized_l2(x, mu_x, std_x, eps)
    mahal_z = standardized_l2(z, mu_z, std_z, eps)
    score_x = exp_mahal_score(mahal_x, gamma_x, feature_dim)
    score_z = exp_mahal_score(mahal_z, gamma_z, feature_dim)

    cos_sim = (z @ text_avg.T).squeeze(-1)
    score_txt = score_text(cos_sim)
    normality = fuse_normality(score_x, score_z, score_txt, fusion)

    return {
        "mahal_x": mahal_x,
        "mahal_z": mahal_z,
        "score_x": score_x,
        "score_z": score_z,
        "score_text": score_txt,
        "normality": normality,
        "anomaly": 1.0 - normality,
    }


def load_stats(path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "mu_x": data["mu_x"],
        "std_x": data["std_x"],
        "mu_z": data["mu_z"],
        "std_z": data["std_z"],
    }


def save_stats(path, mu_x, std_x, mu_z, std_z) -> None:
    np.savez(path, mu_x=mu_x, std_x=std_x, mu_z=mu_z, std_z=std_z)


def temporal_smooth(scores: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or len(scores) < window:
        return scores.copy()
    kernel = np.ones(window) / window
    padded = np.pad(scores, (window - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(scores)]


def encode_text_prompts(prompts: list[str], model, tokenizer,
                        device: str = "cuda") -> torch.Tensor:
    import torch.nn.functional as F
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
    return F.normalize(text_embeds.float(), dim=-1)
