"""
영상 재생하면서 OOD 구간 라벨링 툴 (test 영상용)

Usage:
  python tools/label_video.py --video C:/Users/.../test.mp4 --subject 03

조작 키:
  SPACE       재생 / 일시정지
  ← →         5초 뒤/앞으로
  1~5         OOD 시작/종료 토글
    1 = OOD_01 스마트폰 보기
    2 = OOD_02 엎드리기
    3 = OOD_03 자리 비움
    4 = OOD_04 두리번거리기
    5 = OOD_05 고개 돌리기
  Z           마지막 이벤트 취소 (undo)
  S           저장하고 종료
  Q           저장 없이 종료
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


OOD_KEY_MAP = {49: 'OOD_01', 50: 'OOD_02', 51: 'OOD_03', 52: 'OOD_04', 53: 'OOD_05'}
OOD_NAMES   = {
    'OOD_01': '스마트폰 보기',
    'OOD_02': '엎드리기',
    'OOD_03': '자리 비움',
    'OOD_04': '두리번거리기',
    'OOD_05': '고개 돌리기',
}
WINDOW = "라벨링 툴 — Study Normality Score"


def build_segments(events: list, total: float) -> list:
    if not events:
        return [(0.0, round(total, 2), 'normal')]
    segments = []
    cursor, cur_label = 0.0, 'normal'
    for t, etype, label in events:
        if t - cursor > 0.1:
            segments.append((round(cursor, 2), round(t, 2), cur_label))
        cursor = t
        cur_label = label if etype == 'ood_start' else 'normal'
    if total - cursor > 0.1:
        segments.append((round(cursor, 2), round(total, 2), cur_label))
    return segments


def draw_ui(frame, cur_sec, total_sec, paused, events, current_ood, w_out, h_out):
    d = cv2.resize(frame, (w_out, h_out))
    h, w = d.shape[:2]

    # ── 진행바 ──────────────────────────────────────────────────────────────
    bar_y, bar_h = h - 28, 10
    bar_x0, bar_x1 = 10, w - 10
    bar_w = bar_x1 - bar_x0
    cv2.rectangle(d, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (60, 60, 60), -1)

    # normal / OOD 색상으로 구간 표시
    segs = build_segments(events, cur_sec)
    for s, e, lbl in segs:
        sx = bar_x0 + int((s / total_sec) * bar_w)
        ex = bar_x0 + int((e / total_sec) * bar_w)
        color = (0, 0, 200) if lbl != 'normal' else (0, 180, 0)
        cv2.rectangle(d, (sx, bar_y), (ex, bar_y + bar_h), color, -1)

    # 현재 위치 마커
    cx = bar_x0 + int((cur_sec / total_sec) * bar_w)
    cv2.rectangle(d, (cx - 2, bar_y - 4), (cx + 2, bar_y + bar_h + 4), (255, 255, 255), -1)

    # ── 상단 정보바 ──────────────────────────────────────────────────────────
    overlay = d.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, d, 0.4, 0, d)

    mm, ss = divmod(int(cur_sec), 60)
    tm, ts = divmod(int(total_sec), 60)
    cv2.putText(d, f"{'II' if paused else '▶'}  {mm:02d}:{ss:02d} / {tm:02d}:{ts:02d}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # OOD 상태
    if current_ood:
        status = f"● {current_ood}: {OOD_NAMES[current_ood]}"
        cv2.putText(d, status, (250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
    else:
        cv2.putText(d, "● NORMAL", (250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # 이벤트 수
    cv2.putText(d, f"이벤트 {len(events)}개  |  Z:되돌리기  S:저장  Q:종료",
                (w - 420, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # ── 하단 키 안내 ─────────────────────────────────────────────────────────
    overlay2 = d.copy()
    cv2.rectangle(overlay2, (0, h - 52), (w, h - 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.55, d, 0.45, 0, d)
    cv2.putText(d, "SPACE:재생/정지  ←→:5초이동  1:스마트폰  2:엎드리기  3:자리비움  4:두리번  5:고개돌리기",
                (8, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    return d


def print_segments(segments):
    print("\n[현재 세그먼트]")
    for s, e, lbl in segments:
        tag = "⚡" if lbl != 'normal' else "  "
        print(f"  {tag} {s:7.1f}s ~ {e:7.1f}s  {lbl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',     required=True)
    parser.add_argument('--subject',   required=True)
    parser.add_argument('--session',   default='A')
    parser.add_argument('--data_root', default='data/pilot')
    args = parser.parse_args()

    src = Path(args.video)
    if not src.exists():
        print(f"[ERROR] 파일 없음: {src}")
        return

    # 영상 복사
    subj_dir = Path(args.data_root) / f"subject_{args.subject}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    dst = subj_dir / f"session_{args.session}.mp4"
    labels_path = subj_dir / f"session_{args.session}_labels.csv"

    if not dst.exists():
        print(f"[복사] {src.name} → {dst}")
        shutil.copy2(src, dst)

    cap = cv2.VideoCapture(str(dst))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_f / fps
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 크기 (960 너비로 맞춤)
    scale = min(960 / vid_w, 600 / vid_h)
    w_out = int(vid_w * scale)
    h_out = int(vid_h * scale)

    print(f"[영상] {vid_w}×{vid_h} @ {fps:.1f}fps  총 {total_sec:.1f}s")
    print(f"[조작] SPACE=재생/정지  ←→=5초이동  1~5=OOD토글  Z=되돌리기  S=저장  Q=종료\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w_out, h_out + 30)

    events: list = []       # [(time_sec, 'ood_start'|'ood_end', label)]
    current_ood = None
    paused = True           # 처음엔 일시정지 상태로 시작
    cur_frame = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_f - 1)
            else:
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, frame = cap.read()
            if not ret:
                break

        cur_sec = cur_frame / fps
        display = draw_ui(frame, cur_sec, total_sec, paused,
                          events, current_ood, w_out, h_out)
        cv2.imshow(WINDOW, display)

        wait_ms = 1 if not paused else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        # ── 재생 제어 ──────────────────────────────────────────────────────
        if key == 32:  # SPACE
            paused = not paused

        elif key == 81 or key == 2:  # ← (왼쪽 화살표)
            cur_frame = max(0, cur_frame - int(fps * 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        elif key == 83 or key == 3:  # → (오른쪽 화살표)
            cur_frame = min(total_f - 1, cur_frame + int(fps * 5))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        # ── OOD 토글 ───────────────────────────────────────────────────────
        elif key in OOD_KEY_MAP:
            ood_label = OOD_KEY_MAP[key]
            now = round(cur_sec, 2)

            if current_ood is None:
                events.append((now, 'ood_start', ood_label))
                current_ood = ood_label
                print(f"[{now:.1f}s] ▶ OOD 시작: {ood_label} ({OOD_NAMES[ood_label]})")

            elif current_ood == ood_label:
                events.append((now, 'ood_end', ood_label))
                print(f"[{now:.1f}s] ■ OOD 종료: {ood_label}")
                current_ood = None

            else:
                events.append((now, 'ood_end', current_ood))
                events.append((now, 'ood_start', ood_label))
                print(f"[{now:.1f}s] ↔ 전환: {current_ood} → {ood_label}")
                current_ood = ood_label

        # ── undo ───────────────────────────────────────────────────────────
        elif key in (ord('z'), ord('Z')):
            if events:
                removed = events.pop()
                print(f"[UNDO] 제거: {removed}")
                # OOD 상태 재계산
                current_ood = None
                for _, etype, lbl in events:
                    if etype == 'ood_start':
                        current_ood = lbl
                    elif etype == 'ood_end':
                        current_ood = None

        # ── 저장 ───────────────────────────────────────────────────────────
        elif key in (ord('s'), ord('S')):
            if current_ood:
                events.append((round(cur_sec, 2), 'ood_end', current_ood))
                current_ood = None
            segments = build_segments(events, total_sec)
            with open(labels_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start_sec', 'end_sec', 'label'])
                for seg in segments:
                    writer.writerow(seg)
            print_segments(segments)
            print(f"\n[DONE] labels.csv 저장 → {labels_path}")
            break

        # ── 종료 ───────────────────────────────────────────────────────────
        elif key in (ord('q'), ord('Q')):
            print("[종료] 저장 안 함")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
