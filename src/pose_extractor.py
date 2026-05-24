"""
Video → skeleton pkl  (MediaPipe Tasks API, mediapipe >= 0.10.x)

Input : mp4
Output: skeleton.pkl
  skeleton      : (T, 11, 7)  — 모든 비디오 프레임 유지 (포즈 없으면 0 + 보간)
  frame_indices : (T,)        — 원본 프레임 번호
  7D = [x, y, time_norm, confidence, joint_idx_norm, centroid_x, centroid_y]

confidence < threshold 인 joint는 제거하지 않고, 시간축 선형 보간으로 채움.
"""
import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "pose_landmarker.task")

# MediaPipe 11 상체 관절 (configs data.joints 와 동일)
DEFAULT_JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]

JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]


def _interpolate_low_confidence(skeleton: np.ndarray,
                                conf_thresh: float) -> np.ndarray:
    """
    joint별로 confidence < thresh 인 프레임의 x,y를
    인접 유효 프레임 기준 선형 보간.
    skeleton: (T, J, 7)
    """
    out = skeleton.copy()
    T, J, _ = out.shape
    for j in range(J):
        conf = out[:, j, 3]
        valid = conf >= conf_thresh
        if valid.all():
            continue
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        for d in (0, 1):
            out[:, j, d] = np.interp(
                np.arange(T), idx, out[idx, j, d]
            )
        # 보간된 프레임은 threshold 근처 confidence 부여
        out[~valid, j, 3] = conf_thresh
    return out


def extract_skeleton(video_path: str, cfg: dict,
                     model_path: str = DEFAULT_MODEL) -> tuple:
    """
    Returns (skeleton, frame_indices, total_frames)
      skeleton      : (T, J, 7)
      frame_indices : (T,) — 원본 비디오 프레임 인덱스 (연속)
      total_frames  : int
    """
    joints = cfg["data"].get("joints", DEFAULT_JOINTS)
    min_conf = cfg["data"].get("min_confidence", 0.3)
    conf_thresh = cfg["data"].get("conf_threshold", 0.5)
    n_joints = len(joints)
    joint_idx_norms = np.arange(n_joints, dtype=np.float32) / max(n_joints - 1, 1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None, None, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_indices = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx / fps * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            time_norm = frame_idx / max(total_frames - 1, 1)
            orig_idx = frame_idx
            frame_idx += 1

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                coords = np.array(
                    [[lms[j].x, lms[j].y, lms[j].visibility] for j in joints],
                    dtype=np.float32,
                )
                centroid = coords[:, :2].mean(axis=0)
            else:
                coords = np.zeros((n_joints, 3), dtype=np.float32)
                centroid = np.zeros(2, dtype=np.float32)

            frame_data = np.column_stack([
                coords[:, 0],
                coords[:, 1],
                np.full(n_joints, time_norm, dtype=np.float32),
                coords[:, 2],
                joint_idx_norms,
                np.full(n_joints, centroid[0], dtype=np.float32),
                np.full(n_joints, centroid[1], dtype=np.float32),
            ])
            frames.append(frame_data)
            frame_indices.append(orig_idx)

    cap.release()

    if len(frames) == 0:
        print(f"[WARN] No frames read: {video_path}")
        return None, None, 0

    skeleton = np.stack(frames).astype(np.float32)
    skeleton = _interpolate_low_confidence(skeleton, conf_thresh)

    # 프레임 전체 평균 confidence가 너무 낮은 구간은 한 번 더 보간
    mean_conf = skeleton[:, :, 3].mean(axis=1)
    low_mask = mean_conf < min_conf
    if low_mask.any() and (~low_mask).any():
        valid_idx = np.where(~low_mask)[0]
        for j in range(n_joints):
            for d in (0, 1):
                skeleton[low_mask, j, d] = np.interp(
                    np.where(low_mask)[0], valid_idx,
                    skeleton[valid_idx, j, d],
                )
            skeleton[low_mask, j, 3] = min_conf

    return (
        skeleton,
        np.array(frame_indices, dtype=np.int32),
        total_frames,
    )


def process_session(video_path: str, output_dir: str, cfg: dict,
                    flat: bool = False) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if flat:
        out_path = output_dir / (video_path.stem + ".pkl")
    else:
        out_path = output_dir / (video_path.stem + "_skeleton.pkl")
    if out_path.exists():
        print(f"[SKIP] Already exists: {out_path}")
        return

    print(f"[INFO] Processing: {video_path.name}")
    skeleton, frame_indices, total_frames = extract_skeleton(str(video_path), cfg)
    if skeleton is None:
        return

    with open(out_path, "wb") as f:
        pickle.dump({
            "video_id": video_path.stem,
            "skeleton": skeleton,
            "frame_indices": frame_indices,
            "source": str(video_path),
            "fps": cfg["data"].get("fps", 30),
            "joints": JOINT_NAMES if flat else None,
        }, f)

    valid_rate = (skeleton[:, :, 3].mean(axis=1) >= cfg["data"].get("min_confidence", 0.3)).mean() * 100
    print(f"[DONE] {skeleton.shape}  유효 프레임 비율: {valid_rate:.1f}%  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="mp4 경로 또는 디렉토리")
    parser.add_argument("output_dir", type=str, help="skeleton pkl 저장 경로")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    flat = cfg.get("data", {}).get("layout") == "flat"

    video_path = Path(args.video_path)
    if video_path.is_dir():
        for mp4 in sorted(video_path.glob("**/*.mp4")):
            process_session(str(mp4), args.output_dir, cfg, flat=flat)
    else:
        process_session(str(video_path), args.output_dir, cfg, flat=flat)


if __name__ == "__main__":
    main()
