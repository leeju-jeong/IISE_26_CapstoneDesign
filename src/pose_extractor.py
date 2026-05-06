"""
Video → skeleton pkl
Input : mp4 파일 경로
Output: (T, J, 7) numpy array, 0~1 정규화
        7D = [x, y, time_norm, confidence, joint_idx_norm, centroid_x, centroid_y]
"""
import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml


UPPER_BODY_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]  # 11 joints


def extract_skeleton(video_path: str, cfg: dict) -> np.ndarray | None:
    """
    Returns (T, J, 7) array where T = valid frames, J = 11 joints.
    Returns None if video cannot be opened.
    """
    joints = cfg["data"]["joints"]
    min_conf = cfg["data"].get("min_confidence", 0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks is None:
            frame_idx += 1
            continue

        lms = result.pose_landmarks.landmark
        coords = np.array([[lms[j].x, lms[j].y, lms[j].visibility] for j in joints])  # (J, 3)

        mean_conf = coords[:, 2].mean()
        if mean_conf < min_conf:
            frame_idx += 1
            continue

        # centroid of the joint set (x, y)
        centroid = coords[:, :2].mean(axis=0)  # (2,)

        time_norm = frame_idx / max(total_frames - 1, 1)

        joint_vectors = []
        for j_local, j_global in enumerate(joints):
            x, y, conf = coords[j_local]
            joint_idx_norm = j_local / (len(joints) - 1)
            vec = np.array([x, y, time_norm, conf, joint_idx_norm, centroid[0], centroid[1]])
            joint_vectors.append(vec)

        frames.append(np.stack(joint_vectors))  # (J, 7)
        frame_idx += 1

    cap.release()
    pose.close()

    if len(frames) == 0:
        print(f"[WARN] No valid frames extracted from {video_path}")
        return None

    skeleton = np.stack(frames)  # (T, J, 7)
    skeleton = _normalize(skeleton)
    return skeleton


def _normalize(skeleton: np.ndarray) -> np.ndarray:
    """Per-channel min-max normalization to [0, 1] across all frames."""
    T, J, D = skeleton.shape
    flat = skeleton.reshape(-1, D)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    rng = np.where(mx - mn > 1e-8, mx - mn, 1.0)
    return (skeleton - mn) / rng


def process_session(video_path: str, output_dir: str, cfg: dict) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (video_path.stem + "_skeleton.pkl")
    if out_path.exists():
        print(f"[SKIP] Already exists: {out_path}")
        return

    print(f"[INFO] Processing: {video_path.name}")
    skeleton = extract_skeleton(str(video_path), cfg)
    if skeleton is None:
        return

    with open(out_path, "wb") as f:
        pickle.dump({"skeleton": skeleton, "source": str(video_path)}, f)

    print(f"[DONE] Saved {skeleton.shape} → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="mp4 파일 경로 (or directory)")
    parser.add_argument("output_dir", type=str, help="skeleton pkl 저장 경로")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    video_path = Path(args.video_path)
    if video_path.is_dir():
        for mp4 in sorted(video_path.glob("**/*.mp4")):
            process_session(str(mp4), args.output_dir, cfg)
    else:
        process_session(str(video_path), args.output_dir, cfg)


if __name__ == "__main__":
    main()
