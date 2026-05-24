"""Task 1 visualization helpers."""
from __future__ import annotations

import numpy as np

# sublabel 1~5 + OOD(6)
SUBLABEL_COLORS = {
    "1": "#1f77b4",
    "2": "#ff7f0e",
    "3": "#2ca02c",
    "4": "#d62728",
    "5": "#9467bd",
    "6": "#e377c2",
}


def sublabels_from_dataset(ds) -> np.ndarray:
    return np.array([
        str(m.get("sublabel", "")).strip() if m.get("sublabel") not in (None, "")
        else ""
        for m in ds.meta
    ])


def plot_tsne_by_sublabel(
    feats: np.ndarray,
    sublabels: np.ndarray,
    out_path,
    title: str = "Task1: t-SNE (MotionBERT x)",
) -> None:
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] sklearn/matplotlib missing, skip t-SNE")
        return

    n = len(feats)
    if n < 5:
        print("[WARN] too few clips for t-SNE")
        return

    perp = min(30, n - 1)
    emb = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(feats)

    plt.figure(figsize=(9, 7))
    unique = sorted(set(sublabels), key=lambda x: (x == "", x == "6", x))

    for sl in unique:
        if sl == "":
            mask = sublabels == ""
            lbl = "unknown"
            color = "#aaaaaa"
            size, alpha, marker = 18, 0.4, "o"
        elif sl == "6":
            mask = sublabels == "6"
            lbl = "OOD (6)"
            color = "crimson"
            size, alpha, marker = 80, 0.95, "X"
        else:
            mask = sublabels == sl
            lbl = f"action {sl}"
            color = SUBLABEL_COLORS.get(sl, "#333333")
            size, alpha, marker = 22, 0.55, "o"

        if not mask.any():
            continue
        plt.scatter(
            emb[mask, 0], emb[mask, 1],
            c=color, label=lbl, alpha=alpha, s=size, marker=marker,
            edgecolors="k" if sl == "6" else "none", linewidths=0.5,
        )

    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[DONE] t-SNE → {out_path}")
