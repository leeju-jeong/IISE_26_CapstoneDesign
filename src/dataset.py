"""
skeleton pkl + labels.csv → clip 단위 Dataset (Baseline용)

labels.csv 형식:
  start_sec, end_sec, label
  0, 600, normal
  610, 700, OOD_01

클립 단위:
  clip_frames = 64  (~2.1초 @ 30fps)
  stride_frames = 32 (50% overlap)
  → clip shape: (64, 11, 7) → (T, J, 7) 저장, __getitem__에서 (T, J, 3) 슬라이싱

라벨:
  normal (0): labels.csv의 "normal" 또는 action_id 1~5,7
  anomaly (1): "OOD_*" 또는 action_id 6
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_clips(skeleton: np.ndarray, clip_frames: int,
                stride_frames: int) -> list[np.ndarray]:
    """
    (T, J, 7) skeleton → list of (clip_frames, J, 7) clips
    """
    T = skeleton.shape[0]
    clips = []
    start = 0
    while start + clip_frames <= T:
        clips.append(skeleton[start:start + clip_frames])  # (clip_frames, J, 7)
        start += stride_frames
    return clips


NORMAL_ACTION_IDS = {1, 2, 3, 4, 5, 7}  # 6 = OOD


def load_labels(labels_csv: str, fps: int) -> list[tuple[int, int, str]]:
    """Returns list of (start_frame, end_frame, label).
    label_actions.py 출력(action 컬럼, 정수 1~7)과
    구형 포맷(label 컬럼, "normal"/"OOD_XX") 모두 지원.
    """
    df = pd.read_csv(labels_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # 컬럼명 자동 감지
    if "action" in df.columns:
        label_col = "action"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(f"labels.csv에 'action' 또는 'label' 컬럼이 없습니다: {labels_csv}")

    segments = []
    for _, row in df.iterrows():
        start_f = int(float(row["start_sec"]) * fps)
        end_f   = int(float(row["end_sec"])   * fps)
        raw = str(row[label_col]).strip()

        # 정수 action code (1~7) → "normal" / "OOD" 변환
        try:
            action_id = int(raw)
            label = "normal" if action_id in NORMAL_ACTION_IDS else "OOD"
        except ValueError:
            label = raw  # "normal", "OOD_01" 등 문자열 그대로 사용

        segments.append((start_f, end_f, label))
    return segments


def frame_to_label(frame_idx: int, segments: list[tuple[int, int, str]],
                   clip_frames: int, boundary_margin: int) -> Optional[str]:
    """
    클립 start frame 기준으로 라벨 결정.
    경계 ±margin 프레임 내 클립은 제외(None 반환).
    """
    clip_end = frame_idx + clip_frames
    for seg_start, seg_end, label in segments:
        if frame_idx >= seg_start and clip_end <= seg_end:
            if (frame_idx - seg_start < boundary_margin or
                    seg_end - clip_end < boundary_margin):
                return None
            return label
    return None


def _is_normal(label: str) -> bool:
    return label.lower() == "normal"


class StudyDataset(Dataset):
    def __init__(self, data_root: str, cfg: dict, split: str = "train",
                 normal_only: bool = False):
        """
        data_root/
          subject_01/
            session_A_skeleton.pkl
            session_A_labels.csv
          splits.yaml
        """
        self.cfg = cfg
        self.normal_only = normal_only

        fps = cfg["data"]["fps"]
        clip_frames = cfg["data"]["clip_frames"]
        stride_frames = cfg["data"]["stride_frames"]
        # 경계 마진: 2초 분량
        boundary_margin = fps * 2

        data_root = Path(data_root)
        splits_path = data_root / "splits.yaml"

        if splits_path.exists():
            import yaml
            with open(splits_path) as f:
                splits = yaml.safe_load(f)
            subjects = splits.get(split, [])
        else:
            subjects = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

        self.clips = []   # list of (clip_frames, J, 7)
        self.labels = []  # list of str: 'normal' or 'OOD_*'

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
                    segments = [(0, skeleton.shape[0], "normal")]

                clips = build_clips(skeleton, clip_frames, stride_frames)
                start = 0
                for clip in clips:
                    label = frame_to_label(start, segments, clip_frames, boundary_margin)
                    start += stride_frames
                    if label is None:
                        continue
                    if normal_only and not _is_normal(label):
                        continue
                    self.clips.append(clip)
                    self.labels.append(label)

        n_normal = sum(1 for l in self.labels if _is_normal(l))
        n_ood = len(self.labels) - n_normal
        print(f"[Dataset] split={split}, clips={len(self.clips)}, "
              f"normal={n_normal}, ood={n_ood}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]          # (T, J, 7)
        # MotionBERT 입력: (T, J, 3) — x, y, conf (7D 벡터의 0,1,3번째)
        xyz = clip[:, :, [0, 1, 3]]     # (T, J, 3)
        label = self.labels[idx]
        binary = 0 if _is_normal(label) else 1
        return torch.tensor(xyz, dtype=torch.float32), binary, label
