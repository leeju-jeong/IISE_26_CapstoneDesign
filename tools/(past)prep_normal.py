"""
정상 영상 → 폴더 구조 + labels.csv 자동 생성

Usage:
  python tools/prep_normal.py --video C:/Users/.../session.mp4 --subject 01
  python tools/prep_normal.py --video C:/Users/.../session.mp4 --subject 02
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2


def get_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',     required=True, help='촬영한 mp4 파일 경로')
    parser.add_argument('--subject',   required=True, help='참가자 번호 (예: 01)')
    parser.add_argument('--session',   default='A',   help='세션명 (기본: A)')
    parser.add_argument('--data_root', default='data/pilot')
    args = parser.parse_args()

    src = Path(args.video)
    if not src.exists():
        print(f"[ERROR] 파일 없음: {src}")
        return

    subj_dir = Path(args.data_root) / f"subject_{args.subject}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    dst = subj_dir / f"session_{args.session}.mp4"
    labels_path = subj_dir / f"session_{args.session}_labels.csv"

    # 영상 복사
    print(f"[복사] {src.name} → {dst}")
    shutil.copy2(src, dst)

    # 길이 측정
    duration = get_duration(str(dst))
    print(f"[INFO] 영상 길이: {duration:.1f}초 ({duration/60:.1f}분)")

    # labels.csv (전체 normal)
    with open(labels_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['start_sec', 'end_sec', 'label'])
        writer.writerow([0.0, round(duration, 2), 'normal'])
    print(f"[DONE] labels.csv → {labels_path}")
    print(f"[완료] subject_{args.subject} / session_{args.session} 준비 완료")


if __name__ == '__main__':
    main()
