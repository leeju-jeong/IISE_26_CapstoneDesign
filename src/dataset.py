"""
skeleton pkl + labels.csv → clip 단위 Dataset

레이아웃:
  flat    — data/videos/, data/skeletons/{video_id}.pkl, data/labels.csv
  subject — data/subject_xx/*_skeleton.pkl (기존)
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def frame_step_from_cfg(cfg: dict) -> int:
    fps = cfg["data"]["fps"]
    sampled = cfg["data"].get("sampled_fps", 10)
    return max(1, int(round(fps / sampled)))


def stride_frames_from_cfg(cfg: dict) -> int:
    """stride_sec=0 → non-overlapping (stride = window)."""
    fps = cfg["data"]["fps"]
    window_frames = cfg["data"]["window_sec"] * fps
    stride_sec = cfg["data"]["stride_sec"]
    if stride_sec == 0:
        return window_frames
    return int(stride_sec * fps)


def n_frames_per_clip(cfg: dict) -> int:
    return cfg["data"]["window_sec"] * cfg["data"].get("sampled_fps", 10)


def build_clips_with_indices(
    skeleton: np.ndarray,
    frame_indices: np.ndarray,
    window_frames: int,
    stride_frames: int,
    frame_step: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
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
    fps = cfg["data"]["fps"]
    window_frames = cfg["data"]["window_sec"] * fps
    stride_frames = stride_frames_from_cfg(cfg)
    frame_step = frame_step_from_cfg(cfg)
    T = skeleton.shape[0]
    clips = []
    start = 0
    while start + window_frames <= T:
        clips.append(skeleton[start:start + window_frames:frame_step])
        start += stride_frames
    return clips


def load_labels_csv(labels_csv: str, fps: int,
                    video_id: Optional[str] = None) -> list[tuple[int, int, str, str]]:
    """Returns segments as (start_frame, end_frame, label, sublabel)."""
    df = pd.read_csv(labels_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    segments = []
    for _, row in df.iterrows():
        if video_id is not None and "video_id" in df.columns:
            if str(row["video_id"]).strip() != video_id:
                continue
        start_f = int(float(row["start_sec"]) * fps)
        end_f = int(float(row["end_sec"]) * fps)
        label = str(row["label"]).strip()
        if "sublabel" in df.columns and pd.notna(row["sublabel"]):
            sublabel = str(int(float(row["sublabel"])))
        else:
            sublabel = ""
        if label:
            segments.append((start_f, end_f, label, sublabel))
    return segments


def clip_annotation_from_orig_indices(
    orig_indices: np.ndarray,
    segments: list[tuple[int, int, str, str]],
    boundary_margin: int,
) -> Optional[tuple[str, str]]:
    clip_start = int(orig_indices[0])
    clip_end = int(orig_indices[-1])

    for seg_start, seg_end, label, sublabel in segments:
        if clip_start >= seg_start and clip_end <= seg_end:
            if boundary_margin > 0 and (
                clip_start - seg_start < boundary_margin
                or seg_end - clip_end < boundary_margin
            ):
                return None
            return label, sublabel
    return None


def clip_label_from_orig_indices(
    orig_indices: np.ndarray,
    segments: list[tuple[int, int, str, str]],
    boundary_margin: int,
) -> Optional[str]:
    ann = clip_annotation_from_orig_indices(orig_indices, segments, boundary_margin)
    return ann[0] if ann else None


def _is_normal(label: str) -> bool:
    return label.lower() in ("normal", "study", "studying")


class StudyDataset(Dataset):
    """subject 폴더 레이아웃 (기존)."""

    def __init__(self, data_root: str, cfg: dict, split: str = "train",
                 normal_only: bool = False):
        self.cfg = cfg
        self.normal_only = normal_only
        self.meta: list[dict] = []

        fps = cfg["data"]["fps"]
        window_frames = cfg["data"]["window_sec"] * fps
        stride_frames = stride_frames_from_cfg(cfg)
        frame_step = frame_step_from_cfg(cfg)
        boundary_margin = int(cfg["data"].get("boundary_margin_sec", 5) * fps)

        data_root = Path(data_root)
        splits_path = data_root / "splits.yaml"
        data_cfg = cfg.get("data", {})

        if splits_path.exists():
            import yaml
            with open(splits_path) as f:
                splits = yaml.safe_load(f)
            subject_names = splits.get(split, [])
            subject_dirs = [data_root / s for s in subject_names]
        elif data_cfg.get("subjects"):
            subject_dirs = [data_root / s for s in data_cfg["subjects"]]
        else:
            subject_dirs = [
                data_root / d.name
                for d in sorted(data_root.iterdir())
                if d.is_dir() and d.name not in ("videos", "skeletons")
            ]

        # data_root가 subject 폴더 자체인 경우 (예: data/people1/)
        if not subject_dirs and list(data_root.glob("*_skeleton.pkl")):
            subject_dirs = [data_root]

        self.clips: list[np.ndarray] = []
        self.labels: list[str] = []

        for subj_dir in subject_dirs:
            if not subj_dir.is_dir():
                continue
            for pkl_path in sorted(subj_dir.glob("*_skeleton.pkl")):
                session_name = pkl_path.stem.replace("_skeleton", "")
                csv_path = subj_dir / f"{session_name}_labels.csv"
                self._ingest_pkl(
                    pkl_path, csv_path, fps, window_frames, stride_frames,
                    frame_step, boundary_margin, video_id=session_name,
                )

        self._log_stats()

    def _ingest_pkl(self, pkl_path, csv_path, fps, window_frames, stride_frames,
                    frame_step, boundary_margin, video_id: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        skeleton = data["skeleton"]
        frame_indices = data.get("frame_indices")
        if frame_indices is None:
            frame_indices = np.arange(skeleton.shape[0], dtype=np.int32)

        if csv_path.exists():
            segments = load_labels_csv(str(csv_path), fps)
        else:
            total_orig = int(frame_indices[-1]) + 1
            segments = [(0, total_orig, "normal", "")]

        clip_list = build_clips_with_indices(
            skeleton, frame_indices, window_frames, stride_frames, frame_step,
        )

        for clip, orig_indices in clip_list:
            ann = clip_annotation_from_orig_indices(
                orig_indices, segments, boundary_margin,
            )
            if ann is None:
                continue
            label, sublabel = ann
            if self.normal_only and not _is_normal(label):
                continue
            start_sec = float(orig_indices[0]) / fps
            end_sec = float(orig_indices[-1]) / fps
            self.clips.append(clip)
            self.labels.append(label)
            self.meta.append({
                "video_id": video_id,
                "label": label,
                "sublabel": sublabel,
                "clip_start_sec": start_sec,
                "clip_end_sec": end_sec,
            })

    def _log_stats(self):
        print(f"[Dataset] clips={len(self.clips)}, "
              f"normal={sum(1 for l in self.labels if _is_normal(l))}, "
              f"ood={sum(1 for l in self.labels if not _is_normal(l))}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        binary = 0 if _is_normal(label) else 1
        return torch.tensor(clip, dtype=torch.float32), binary, label


class FlatDataset(Dataset):
    """
    Task 1 flat 레이아웃:
      data/skeletons/{video_id}.pkl + data/labels.csv
    """

    def __init__(self, data_root: str, cfg: dict, normal_only: bool = True):
        self.cfg = cfg
        self.normal_only = normal_only
        self.meta: list[dict] = []

        fps = cfg["data"]["fps"]
        window_frames = cfg["data"]["window_sec"] * fps
        stride_frames = stride_frames_from_cfg(cfg)
        frame_step = frame_step_from_cfg(cfg)
        boundary_margin = int(cfg["data"].get("boundary_margin_sec", 0) * fps)

        data_root = Path(data_root)
        skel_dir = data_root / "skeletons"
        labels_csv = data_root / "labels.csv"

        self.clips: list[np.ndarray] = []
        self.labels: list[str] = []

        if not skel_dir.is_dir():
            print(f"[FlatDataset] skeletons dir missing: {skel_dir}")
            return

        pkl_paths = sorted(skel_dir.glob("*.pkl"))
        if not pkl_paths:
            print(f"[FlatDataset] no skeleton pkl in {skel_dir}")
            return

        for pkl_path in pkl_paths:
            video_id = pkl_path.stem
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            skeleton = data["skeleton"]
            frame_indices = data.get("frame_indices")
            if frame_indices is None:
                frame_indices = np.arange(skeleton.shape[0], dtype=np.int32)

            if labels_csv.exists():
                segments = load_labels_csv(str(labels_csv), fps, video_id=video_id)
                if not segments:
                    segments = [(0, int(frame_indices[-1]) + 1, "normal", "")]
            else:
                segments = [(0, int(frame_indices[-1]) + 1, "normal", "")]

            clip_list = build_clips_with_indices(
                skeleton, frame_indices, window_frames, stride_frames, frame_step,
            )

            for clip, orig_indices in clip_list:
                ann = clip_annotation_from_orig_indices(
                    orig_indices, segments, boundary_margin,
                )
                if ann is None:
                    continue
                label, sublabel = ann
                if self.normal_only and not _is_normal(label):
                    continue
                start_sec = float(orig_indices[0]) / fps
                end_sec = float(orig_indices[-1]) / fps
                self.clips.append(clip)
                self.labels.append(label)
                self.meta.append({
                    "video_id": video_id,
                    "label": label,
                    "sublabel": sublabel,
                    "clip_start_sec": start_sec,
                    "clip_end_sec": end_sec,
                })

        print(f"[FlatDataset] videos={len(pkl_paths)}, clips={len(self.clips)}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        binary = 0 if _is_normal(label) else 1
        return torch.tensor(clip, dtype=torch.float32), binary, label


def build_dataset(data_root: str, cfg: dict, split: str = "train",
                  normal_only: bool = False) -> Dataset:
    layout = cfg.get("data", {}).get("layout", "subject")
    if layout == "flat":
        return FlatDataset(data_root, cfg, normal_only=normal_only)
    return StudyDataset(data_root, cfg, split=split, normal_only=normal_only)
