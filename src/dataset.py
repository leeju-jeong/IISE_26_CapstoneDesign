"""
skeleton pkl + labels.csv → clip 단위 Dataset

labels.csv:
  start_sec, end_sec, label

클립 (기본):
  window=5s, stride=3s, sampled_fps=10 → frame_step=3 @ 30fps
  clip shape: (50, 11, 7)

비교용:
  window=10s, stride=5s → (100, 11, 7)

라벨: pkl의 frame_indices로 labels.csv와 매칭
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def frame_step_from_cfg(cfg: dict) -> int:
    fps = cfg["data"]["fps"]
    sampled = cfg["data"].get("sampled_fps", 10)
    return max(1, int(round(fps / sampled)))


def n_frames_per_clip(cfg: dict) -> int:
    return cfg["data"]["window_sec"] * cfg["data"].get("sampled_fps", 10)


def build_clips_with_indices(
    skeleton: np.ndarray,
    frame_indices: np.ndarray,
    window_frames: int,
    stride_frames: int,
    frame_step: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    skeleton     : (T, J, 7)
    frame_indices: (T,) 원본 비디오 프레임 번호

    Returns list of (clip, clip_orig_indices)
      clip             : (N, J, 7)   N = window_frames // frame_step
      clip_orig_indices: (window_frames,)
    """
    T = skeleton.shape[0]
    result = []
    start = 0
    while start + window_frames <= T:
        clip = skeleton[start:start + window_frames:frame_step]
        orig_indices = frame_indices[start:start + window_frames]
        result.append((clip, orig_indices))
        start += stride_frames
    return result


def build_clips(skeleton: np.ndarray, cfg: dict) -> list[np.ndarray]:
    """inference용 (labels 없음)."""
    fps = cfg["data"]["fps"]
    window_frames = cfg["data"]["window_sec"] * fps
    stride_frames = cfg["data"]["stride_sec"] * fps
    frame_step = frame_step_from_cfg(cfg)
    T = skeleton.shape[0]
    clips = []
    start = 0
    while start + window_frames <= T:
        clips.append(skeleton[start:start + window_frames:frame_step])
        start += stride_frames
    return clips


def load_labels(labels_csv: str, fps: int) -> list[tuple[int, int, str]]:
    df = pd.read_csv(labels_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    segments = []
    for _, row in df.iterrows():
        start_f = int(float(row["start_sec"]) * fps)
        end_f = int(float(row["end_sec"]) * fps)
        label = str(row["label"]).strip()
        segments.append((start_f, end_f, label))
    return segments


def clip_label_from_orig_indices(
    orig_indices: np.ndarray,
    segments: list[tuple[int, int, str]],
    boundary_margin: int,
) -> Optional[str]:
    clip_start = int(orig_indices[0])
    clip_end = int(orig_indices[-1])

    for seg_start, seg_end, label in segments:
        if clip_start >= seg_start and clip_end <= seg_end:
            if (clip_start - seg_start < boundary_margin or
                    seg_end - clip_end < boundary_margin):
                return None
            return label
    return None


class StudyDataset(Dataset):
    def __init__(self, data_root: str, cfg: dict, split: str = "train",
                 normal_only: bool = False):
        self.cfg = cfg
        self.normal_only = normal_only

        fps = cfg["data"]["fps"]
        window_sec = cfg["data"]["window_sec"]
        stride_sec = cfg["data"]["stride_sec"]
        frame_step = frame_step_from_cfg(cfg)
        window_frames = window_sec * fps
        stride_frames = stride_sec * fps
        boundary_margin = int(cfg["data"].get("boundary_margin_sec", 5) * fps)

        data_root = Path(data_root)
        splits_path = data_root / "splits.yaml"

        if splits_path.exists():
            import yaml
            with open(splits_path) as f:
                splits = yaml.safe_load(f)
            subjects = splits.get(split, [])
        else:
            subjects = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

        self.clips = []
        self.labels = []

        for subj in subjects:
            subj_dir = data_root / subj
            if not subj_dir.is_dir():
                continue
            for pkl_path in sorted(subj_dir.glob("*_skeleton.pkl")):
                session_name = pkl_path.stem.replace("_skeleton", "")
                csv_path = subj_dir / f"{session_name}_labels.csv"

                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)

                skeleton = data["skeleton"]
                frame_indices = data.get("frame_indices")
                if frame_indices is None:
                    frame_indices = np.arange(skeleton.shape[0], dtype=np.int32)

                if csv_path.exists():
                    segments = load_labels(str(csv_path), fps)
                else:
                    total_orig = int(frame_indices[-1]) + 1
                    segments = [(0, total_orig, "normal")]

                clip_list = build_clips_with_indices(
                    skeleton, frame_indices,
                    window_frames, stride_frames, frame_step,
                )

                for clip, orig_indices in clip_list:
                    label = clip_label_from_orig_indices(
                        orig_indices, segments, boundary_margin,
                    )
                    if label is None:
                        continue
                    if normal_only and label != "normal":
                        continue
                    self.clips.append(clip)
                    self.labels.append(label)

        print(f"[Dataset] split={split}, clips={len(self.clips)}, "
              f"normal={self.labels.count('normal')}, "
              f"ood={sum(1 for l in self.labels if l != 'normal')}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]               # (N, J, 7)
        label = self.labels[idx]
        binary = 0 if label == "normal" else 1
        return torch.tensor(clip, dtype=torch.float32), binary, label
