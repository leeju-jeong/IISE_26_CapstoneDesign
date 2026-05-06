"""
공통 유틸리티:
  - Mahalanobis distance (diagonal covariance)
  - Study Normality Score 계산
  - Temporal smoothing
"""
import numpy as np
import torch


# ──────────────────────────────────────────────
# Mahalanobis (diagonal covariance)
# ──────────────────────────────────────────────

def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    features: (N, D) normal clip features
    Returns (mu, std) each shape (D,)
    """
    mu = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    return mu, std


def mahalanobis_diag(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    x   : (N, D) or (D,)
    Returns: (N,) or scalar distances
    """
    return np.sqrt(((x - mu) ** 2 / (std ** 2)).sum(axis=-1))


# ──────────────────────────────────────────────
# Score fusion
# ──────────────────────────────────────────────

def compute_normality_score(mahal_dist: np.ndarray, prompt_sim: np.ndarray,
                             gamma: float = 1.0) -> np.ndarray:
    """
    mahal_dist  : (N,) Mahalanobis distances (larger = more OoD)
    prompt_sim  : (N,) cosine similarities with normal text ∈ [-1, 1]
    gamma       : decay rate for OoD component

    Returns: (N,) Study Normality Score ∈ [0, 1]
    """
    # OoD component: normal clip → small distance → score ≈ 1
    ood_score = np.exp(-gamma * mahal_dist)

    # Prompt component: normal clip → high cosine sim → score ≈ 1
    prompt_score = (prompt_sim + 1.0) / 2.0  # [-1,1] → [0,1]

    return ood_score * prompt_score


# ──────────────────────────────────────────────
# Temporal smoothing
# ──────────────────────────────────────────────

def temporal_smooth(scores: np.ndarray, window: int = 3) -> np.ndarray:
    """
    scores : (T,) per-clip normality scores
    window : number of clips to average (centered)
    Returns smoothed (T,) array
    """
    if window <= 1 or len(scores) < window:
        return scores.copy()
    pad = window // 2
    padded = np.pad(scores, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smoothed[: len(scores)]


# ──────────────────────────────────────────────
# CLIP text encoding
# ──────────────────────────────────────────────

def encode_text_prompts(prompts: list[str], model, tokenizer,
                        device: str = "cuda") -> torch.Tensor:
    """
    Returns (P, 512) L2-normalized text embeddings using open_clip.
    """
    import torch.nn.functional as F
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_embeds = model.encode_text(tokens)
    return F.normalize(text_embeds.float(), dim=-1)
