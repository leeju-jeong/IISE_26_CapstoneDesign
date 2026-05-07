"""
Video → skeleton pkl  (MediaPipe Tasks API, mediapipe >= 0.10.x)
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

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "pose_landmarker.task")


def extract_skeleton(video_path: str, cfg: dict,
                     model_path: str = DEFAULT_MODEL) -> tuple:
    """
    Returns (skeleton, frame_indices, total_frames) where
      skeleton      : (T, J, 7) float32, 0~1 정규화
      frame_indices : (T,) int32, 원본 비디오 프레임 번호
      total_frames  : int, 원본 비디오 총 프레임 수
    Returns (None, None, 0) on failure.
    """
    joints = cfg["data"]["joints"]
    min_conf = cfg["data"].get("min_confidence", 0.5)
    n_joints = len(joints)
    joint_idx_norms = np.arange(n_joints) / (n_joints - 1)  # 사전 계산

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
    valid_frame_indices = []

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
            frame_idx += 1  # 매 프레임마다 단일 증가

            if not result.pose_landmarks:
                continue

            lms = result.pose_landmarks[0]
            coords = np.array([[lms[j].x, lms[j].y, lms[j].visibility]
                                for j in joints])  # (J, 3)

            if coords[:, 2].mean() < min_conf:
                continue

            centroid = coords[:, :2].mean(axis=0)
            time_norm = (frame_idx - 1) / max(total_frames - 1, 1)

            # 벡터화: (J, 7) 한 번에 구성
            frame_data = np.column_stack([
                coords[:, 0],                        # x
                coords[:, 1],                        # y
                np.full(n_joints, time_norm),        # time_norm
                coords[:, 2],                        # confidence
                joint_idx_norms,                     # joint_idx_norm
                np.full(n_joints, centroid[0]),      # centroid_x
                np.full(n_joints, centroid[1]),      # centroid_y
            ])
            frames.append(frame_data)
            valid_frame_indices.append(frame_idx - 1)

    cap.release()

    if len(frames) == 0:
        print(f"[WARN] No valid frames: {video_path}")
        return None, None, 0

    skeleton = np.stack(frames).astype(np.float32)  # (T, J, 7)
    skeleton = _normalize(skeleton)
    return skeleton, np.array(valid_frame_indices, dtype=np.int32), total_frames


def _normalize(skeleton: np.ndarray) -> np.ndarray:
    flat = skeleton.reshape(-1, skeleton.shape[-1])
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
    skeleton, frame_indices, total_frames = extract_skeleton(str(video_path), cfg)
    if skeleton is None:
        return

    with open(out_path, "wb") as f:
        pickle.dump({
            "skeleton": skeleton,
            "frame_indices": frame_indices,
            "source": str(video_path),
        }, f)

    valid_rate = len(frame_indices) / max(total_frames, 1) * 100
    print(f"[DONE] {skeleton.shape}  유효 프레임: {valid_rate:.1f}%  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="mp4 경로 또는 디렉토리")
    parser.add_argument("output_dir", type=str, help="skeleton pkl 저장 경로")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
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
