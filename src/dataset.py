"""
skeleton pkl + labels.csv → clip 단위 Dataset

labels.csv 형식:
  start_sec, end_sec, label
  0, 600, normal
  610, 700, OOD_01

clip 단위:
  window = 10s, stride = 5s
  frame_step = 3 (매 3번째 프레임 샘플링)
  → 10s × 30fps / 3 = 100 frames per clip
  → clip shape: (100, 11, 7) → flatten → (1100, 7)
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_clips(skeleton: np.ndarray, fps: int, window_sec: int,
                stride_sec: int, frame_step: int) -> list[np.ndarray]:
    """
    (T, J, 7) skeleton → list of (N, J, 7) clips
    N = (window_sec * fps) // frame_step
    """
    window_frames = window_sec * fps
    stride_frames = stride_sec * fps
    T = skeleton.shape[0]
    clips = []
    start = 0
    while start + window_frames <= T:
        clip = skeleton[start:start + window_frames:frame_step]  # (N, J, 7)
        clips.append(clip)
        start += stride_frames
    return clips


def load_labels(labels_csv: str, fps: int) -> list[tuple[int, int, str]]:
    """Returns list of (start_frame, end_frame, label)."""
    df = pd.read_csv(labels_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    segments = []
    for _, row in df.iterrows():
        start_f = int(float(row["start_sec"]) * fps)
        end_f = int(float(row["end_sec"]) * fps)
        label = str(row["label"]).strip()
        segments.append((start_f, end_f, label))
    return segments


def frame_to_label(frame_idx: int, segments: list[tuple[int, int, str]],
                   window_frames: int, boundary_margin: int) -> Optional[str]:
    """
    클립의 start frame을 기준으로 라벨 결정.
    경계 ±margin 프레임 내 클립은 제외(None 반환).
    """
    clip_end = frame_idx + window_frames

    for seg_start, seg_end, label in segments:
        if frame_idx >= seg_start and clip_end <= seg_end:
            # clip이 완전히 세그먼트 안에 있는 경우
            # boundary_margin 확인
            if (frame_idx - seg_start < boundary_margin or
                    seg_end - clip_end < boundary_margin):
                return None
            return label

    return None  # 세그먼트에 걸쳐 있거나 해당 없음


class StudyDataset(Dataset):
    def __init__(self, data_root: str, cfg: dict, split: str = "train",
                 normal_only: bool = False):
        """
        data_root 구조:
          data_root/
            subject_01/
              session_A_skeleton.pkl
              session_A_labels.csv
            subject_02/
              ...
            splits.yaml  (train/val/test subject 분리)
        """
        self.cfg = cfg
        self.normal_only = normal_only

        fps = cfg["data"]["fps"]
        window_sec = cfg["data"]["window_sec"]
        stride_sec = cfg["data"]["stride_sec"]
        frame_step = cfg["data"]["frame_step"]
        window_frames = window_sec * fps
        boundary_margin = fps * 5  # ±5초 경계 클립 제외

        data_root = Path(data_root)
        splits_path = data_root / "splits.yaml"

        if splits_path.exists():
            import yaml
            with open(splits_path) as f:
                splits = yaml.safe_load(f)
            subjects = splits.get(split, [])
        else:
            # splits.yaml 없으면 전체 subject 사용
            subjects = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

        self.clips = []   # list of (T, J, 7) numpy arrays
        self.labels = []  # list of str: 'normal' or 'OOD_XX'

        for subj in subjects:
            subj_dir = data_root / subj
            if not subj_dir.is_dir():
                continue
            for pkl_path in sorted(subj_dir.glob("*_skeleton.pkl")):
                session_name = pkl_path.stem.replace("_skeleton", "")
                csv_path = subj_dir / f"{session_name}_labels.csv"

                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                skeleton = data["skeleton"]  # (T, J, 7)

                if csv_path.exists():
                    segments = load_labels(str(csv_path), fps)
                else:
                    # labels.csv 없으면 전체를 normal로 간주
                    segments = [(0, skeleton.shape[0], "normal")]

                clips = build_clips(skeleton, fps, window_sec, stride_sec, frame_step)
                start = 0
                for clip in clips:
                    label = frame_to_label(start, segments, window_frames, boundary_margin)
                    start += stride_sec * fps
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
        N, J, D = clip.shape
        points = clip.reshape(N * J, D)      # (N*J, 7) for PointNet
        label = self.labels[idx]
        binary = 0 if label == "normal" else 1
        return torch.tensor(points, dtype=torch.float32), binary, label
