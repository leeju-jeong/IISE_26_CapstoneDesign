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

라벨 매칭:
  pkl에 저장된 원본 frame_indices를 사용해 labels.csv와 매칭
  → skeleton의 유효 프레임 필터링과 labels.csv 프레임 공간이 일치함
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_clips_with_indices(
    skeleton: np.ndarray,
    frame_indices: np.ndarray,
    window_frames: int,
    stride_frames: int,
    frame_step: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    skeleton     : (T, J, 7)  valid frames only
    frame_indices: (T,)        original video frame indices for each valid frame

    Returns list of (clip, clip_orig_indices)
      clip             : (N, J, 7)   N = window_frames // frame_step
      clip_orig_indices: (window_frames,) original frame indices of the window
    """
    T = skeleton.shape[0]
    result = []
    start = 0
    while start + window_frames <= T:
        clip = skeleton[start:start + window_frames:frame_step]  # (N, J, 7)
        orig_indices = frame_indices[start:start + window_frames]  # (window_frames,)
        result.append((clip, orig_indices))
        start += stride_frames
    return result


def build_clips(skeleton: np.ndarray, fps: int, window_sec: int,
                stride_sec: int, frame_step: int) -> list[np.ndarray]:
    """inference용 단순 버전 (labels.csv 없이 사용)."""
    window_frames = window_sec * fps
    stride_frames = stride_sec * fps
    T = skeleton.shape[0]
    clips = []
    start = 0
    while start + window_frames <= T:
        clip = skeleton[start:start + window_frames:frame_step]
        clips.append(clip)
        start += stride_frames
    return clips


def load_labels(labels_csv: str, fps: int) -> list[tuple[int, int, str]]:
    """Returns list of (start_frame, end_frame, label) in original video frame space."""
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
    """
    orig_indices    : 클립의 원본 프레임 인덱스 배열 (window_frames,)
    boundary_margin : 경계에서 이 프레임 수 이내면 None 반환

    클립 전체가 하나의 세그먼트 안에 속할 때만 라벨 반환.
    """
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
        """
        data_root 구조:
          data_root/
            subject_01/
              session_A_skeleton.pkl    (skeleton + frame_indices 포함)
              session_A_labels.csv
            subject_02/
              ...
            splits.yaml  (train/val/test subject 분리, 없으면 전체 사용)
        """
        self.cfg = cfg
        self.normal_only = normal_only

        fps = cfg["data"]["fps"]
        window_sec = cfg["data"]["window_sec"]
        stride_sec = cfg["data"]["stride_sec"]
        frame_step = cfg["data"]["frame_step"]
        window_frames = window_sec * fps
        stride_frames = stride_sec * fps
        boundary_margin = fps * 5  # ±5초 경계 클립 제외

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

                skeleton = data["skeleton"]          # (T, J, 7)
                frame_indices = data.get("frame_indices")  # (T,) or None

                # frame_indices 없는 구버전 pkl 호환
                if frame_indices is None:
                    frame_indices = np.arange(skeleton.shape[0], dtype=np.int32)

                if csv_path.exists():
                    segments = load_labels(str(csv_path), fps)
                else:
                    # labels.csv 없으면 전체를 normal로 간주
                    total_orig_frames = int(frame_indices[-1]) + 1
                    segments = [(0, total_orig_frames, "normal")]

                clip_list = build_clips_with_indices(
                    skeleton, frame_indices, window_frames, stride_frames, frame_step
                )

                for clip, orig_indices in clip_list:
                    label = clip_label_from_orig_indices(
                        orig_indices, segments, boundary_margin
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
        N, J, D = clip.shape
        points = clip.reshape(N * J, D)      # (N*J, 7)
        label = self.labels[idx]
        binary = 0 if label == "normal" else 1
        return torch.tensor(points, dtype=torch.float32), binary, label
