"""
skeleton.pkl + labels.csv → 행동별 clip 폴더 분류 + 훈련셋 샘플링

실행:
  python tools/generate_clips.py \\
      --skeleton data/raw/subject_01_skeleton.pkl \\
      --labels   data/raw/subject_01_labels.csv \\
      --output   data/clips/subject_01 \\
      --n_train  30        # 행동당 최대 30개 클립 (훈련셋)

출력 구조:
  data/clips/subject_01/
    action_1/clip_0001.npy   # shape: (64, 11, 3)  x,y,conf
    action_1/clip_0002.npy
    ...
    action_6/clip_0001.npy   # anomaly (inference only)
    train_manifest.csv       # 행동 1-5,7에서 랜덤 샘플링된 훈련 클립 목록
    test_manifest.csv        # 전체 클립 목록 (행동별 라벨 포함)

labels.csv 형식 (label_actions.py 출력):
  start_sec, end_sec, action
  0.000, 12.460, 1
  12.460, 15.460, 7
  ...

클립 파라미터:
  clip_frames = 64  (~2.1초 @ 30fps)
  stride      = 32  (50% overlap)

훈련셋 샘플링 (--n_train N):
  - 행동 1~5, 7 에서만 샘플링 (6번 제외)
  - 행동별로 최대 N개 랜덤 선택
  - 인접 클립 최소 2개 간격 유지 (같은 행동 세그먼트 내에서 연속 선택 방지)
"""
import argparse
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

CLIP_FRAMES  = 64
STRIDE       = CLIP_FRAMES            # 겹침 없음 — 각 구간 내에서 독립 클립만 생성
TRAIN_ACTIONS = {1, 2, 3, 4, 5, 7}   # 6번(딴짓)은 훈련 제외


# ── Clip 생성 ──────────────────────────────────────────────────────────────────

def extract_clips(skeleton: np.ndarray, start_frame: int, end_frame: int) -> list[np.ndarray]:
    """
    skeleton (T, J, 7) 에서 [start_frame, end_frame) 구간을
    64프레임 슬라이딩 윈도우로 클립 리스트 반환.
    각 클립: (64, 11, 3) — x, y, conf (7D에서 인덱스 0,1,3)
    """
    segment = skeleton[start_frame:end_frame]  # (L, J, 7)
    L = segment.shape[0]
    clips = []
    pos = 0
    while pos + CLIP_FRAMES <= L:
        clip = segment[pos:pos + CLIP_FRAMES, :, [0, 1, 3]]  # (64, J, 3)
        clips.append(clip.astype(np.float32))
        pos += STRIDE
    return clips


# ── 훈련셋 샘플링 ─────────────────────────────────────────────────────────────

def sample_with_spacing(clip_paths: list[Path], n: int, min_gap: int = 2) -> list[Path]:
    """
    clip_paths: 한 행동의 전체 클립 경로 리스트 (정렬된 순서)
    n         : 최대 샘플 수
    min_gap   : 연속 선택 방지 최소 간격 (클립 단위)
    → 최대 n개를 랜덤하게 선택하되, 선택된 클립 인덱스 간격이 min_gap 이상
    """
    if len(clip_paths) <= n:
        return list(clip_paths)

    indices = list(range(len(clip_paths)))
    random.shuffle(indices)

    selected = []
    selected_set = set()

    for idx in indices:
        # 이미 선택된 인덱스와 min_gap 이상 떨어져 있는지 확인
        too_close = any(abs(idx - s) < min_gap for s in selected_set)
        if not too_close:
            selected.append(idx)
            selected_set.add(idx)
        if len(selected) >= n:
            break

    return [clip_paths[i] for i in sorted(selected)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton", required=True, help="skeleton.pkl 경로")
    parser.add_argument("--labels",   required=True, help="labels.csv 경로 (action_actions.py 출력)")
    parser.add_argument("--output",   required=True, help="클립 저장 폴더")
    parser.add_argument("--fps",      type=float, default=30.0)
    parser.add_argument("--n_train",  type=int, default=None,
                        help="행동당 최대 훈련 클립 수 (미지정 시 전체 사용)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── skeleton 로드 ─────────────────────────────────────────────────────────
    with open(args.skeleton, "rb") as f:
        data = pickle.load(f)
    skeleton = data["skeleton"]  # (T, J, 7)
    T, J, D = skeleton.shape
    print(f"[INFO] skeleton: {T} frames, {J} joints, {D}D")

    # ── labels 로드 ───────────────────────────────────────────────────────────
    segments = []  # (start_frame, end_frame, action_id)
    with open(args.labels, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_f = int(float(row["start_sec"]) * args.fps)
            end_f   = int(float(row["end_sec"])   * args.fps)
            action  = int(row["action"])
            start_f = max(0, min(start_f, T))
            end_f   = max(0, min(end_f,   T))
            if end_f - start_f >= CLIP_FRAMES:
                segments.append((start_f, end_f, action))

    if not segments:
        print("[ERROR] 유효한 구간이 없습니다.")
        return

    print(f"[INFO] {len(segments)}개 구간 로드")

    # ── 클립 생성 및 저장 ─────────────────────────────────────────────────────
    # action → list of saved paths
    clips_by_action: dict[int, list[Path]] = defaultdict(list)

    for start_f, end_f, action in segments:
        action_dir = out_dir / f"action_{action}"
        action_dir.mkdir(exist_ok=True)

        clips = extract_clips(skeleton, start_f, end_f)
        for clip in clips:
            idx = len(clips_by_action[action])
            path = action_dir / f"clip_{idx:05d}.npy"
            np.save(str(path), clip)
            clips_by_action[action].append(path)

    print("\n[클립 생성 완료]")
    total_train_pool = 0
    for a in sorted(clips_by_action):
        n = len(clips_by_action[a])
        tag = " (inference only)" if a == 6 else " (train pool)"
        if a in TRAIN_ACTIONS:
            total_train_pool += n
        print(f"  action_{a} ({_name(a)}): {n}개{tag}")

    # ── test_manifest.csv — 전체 클립 ─────────────────────────────────────────
    test_rows = []
    for a in sorted(clips_by_action):
        for p in clips_by_action[a]:
            test_rows.append((str(p.relative_to(out_dir)), a, 0 if a != 6 else 1))

    test_csv = out_dir / "test_manifest.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "action", "label_binary"])
        writer.writerows(test_rows)
    print(f"\n[test_manifest] {len(test_rows)}개 클립 → {test_csv}")

    # ── train_manifest.csv — 행동 1~5,7 샘플링 ────────────────────────────────
    train_rows = []
    for a in sorted(TRAIN_ACTIONS):
        if a not in clips_by_action:
            continue
        pool = sorted(clips_by_action[a])  # 정렬하여 순서 고정
        if args.n_train:
            selected = sample_with_spacing(pool, args.n_train, min_gap=2)
        else:
            selected = pool
        for p in selected:
            train_rows.append((str(p.relative_to(out_dir)), a, 0))

    # 행 셔플 (행동 편향 방지)
    random.shuffle(train_rows)

    train_csv = out_dir / "train_manifest.csv"
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "action", "label_binary"])
        writer.writerows(train_rows)

    print(f"[train_manifest] {len(train_rows)}개 클립 → {train_csv}")
    if args.n_train:
        print(f"  (행동당 최대 {args.n_train}개 샘플링, 인접 간격 ≥2 유지)")

    print("\n[완료]")


def _name(a: int) -> str:
    return {1:"Typing",2:"Watching",3:"Writing",4:"Problem",5:"Pad",
            6:"OFF-TASK",7:"Transition"}.get(a, "?")


if __name__ == "__main__":
    main()
