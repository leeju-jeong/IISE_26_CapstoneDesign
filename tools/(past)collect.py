"""
Study Normality Score — 데이터 수집 도구

Usage:
  python tools/collect.py --subject 01 --session A
  python tools/collect.py --subject 01 --session C
  python tools/collect.py --subject 01 --session A --duration 60   # 테스트 단축

Session C OOD 키 (같은 키 다시 → 종료):
  1 → OOD_01  스마트폰 보기
  2 → OOD_02  엎드리기
  3 → OOD_03  자리 비움
  4 → OOD_04  두리번거리기
  5 → OOD_05  고개 돌리기
  Q → 녹화 종료
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Study Normality Score — Data Collector"

OOD_KEY_MAP = {
    49: 'OOD_01',  # '1'
    50: 'OOD_02',  # '2'
    51: 'OOD_03',  # '3'
    52: 'OOD_04',  # '4'
    53: 'OOD_05',  # '5'
}
OOD_NAMES = {
    'OOD_01': '스마트폰 보기',
    'OOD_02': '엎드리기',
    'OOD_03': '자리 비움',
    'OOD_04': '두리번거리기',
    'OOD_05': '고개 돌리기',
}
SESSION_DURATIONS = {'A': 600, 'B': 300, 'C': 600, 'D': 300}


# ── Camera ────────────────────────────────────────────────────────────────────

def setup_camera(camera_idx: int):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {camera_idx} 열기 실패. --camera 옵션으로 인덱스 변경 시도.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 실제 해상도 확인 (첫 프레임 기준)
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 1 else 30.0   # 0으로 보고되는 경우 fallback
    print(f"[Camera] {w}×{h} @ {fps:.1f}fps  (인덱스={camera_idx})")
    return cap, w, h, fps


# ── UI drawing ────────────────────────────────────────────────────────────────

def _bar(img, y0, y1, alpha=0.55):
    """Semi-transparent dark bar at y0:y1."""
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    roi = img[y0:y1, :]
    dark = np.zeros_like(roi)
    img[y0:y1, :] = cv2.addWeighted(roi, 1 - alpha, dark, alpha, 0)


def draw_ui(frame: np.ndarray, elapsed: float, total: float,
            session: str, subject: str,
            current_ood: str | None, ood_start: float | None) -> np.ndarray:
    d = frame.copy()
    h, w = d.shape[:2]

    # border: green = normal, red = OOD active
    border_color = (0, 0, 220) if current_ood else (0, 200, 0)
    cv2.rectangle(d, (0, 0), (w - 1, h - 1), border_color, 6)

    # ── top bar ──────────────────────────────────────────────────────────────
    _bar(d, 0, 75)

    mm, ss = divmod(int(elapsed), 60)
    rm, rs = divmod(int(max(0, total - elapsed)), 60)
    timer = f"{mm:02d}:{ss:02d}  |  남은 {rm:02d}:{rs:02d}"
    cv2.putText(d, timer, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(d, f"Subject {subject}  Session {session}",
                (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # REC blink
    if int(elapsed * 2) % 2 == 0:
        cv2.circle(d, (w - 28, 22), 9, (0, 0, 255), -1)
    cv2.putText(d, "REC", (w - 68, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── bottom bar ───────────────────────────────────────────────────────────
    _bar(d, h - 85, h)

    if session == 'C':
        if current_ood:
            dur = elapsed - (ood_start or elapsed)
            label_text = f"● {current_ood}: {OOD_NAMES[current_ood]}  [{dur:.0f}s]  (같은 키 → 종료)"
            cv2.putText(d, label_text, (10, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        else:
            cv2.putText(d, "● NORMAL",
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2)

        cv2.putText(d, "1:스마트폰  2:엎드리기  3:자리비움  4:두리번  5:고개돌리기  Q:종료",
                    (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    else:
        cv2.putText(d, "NORMAL SESSION  |  Q: 조기 종료",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2)

    return d


# ── Phase helpers ─────────────────────────────────────────────────────────────

def preview_phase(cap, subject, session):
    """카메라 위치 확인. SPACE → 시작, ESC → 취소."""
    print("[PREVIEW] 상체 전체가 화면에 들어오는지 확인 후 SPACE를 누르세요.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        d = frame.copy()
        h, w = d.shape[:2]
        _bar(d, 0, 95)
        cv2.putText(d, f"Subject {subject}  Session {session}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(d, "상체 전체(머리~손목)가 화면에 들어오는지 확인하세요",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 230), 2)
        _bar(d, h - 50, h)
        cv2.putText(d, "SPACE: 녹화 시작  |  ESC: 취소",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.imshow(WINDOW, d)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:   # SPACE
            return True
        if key == 27:   # ESC
            return False


def countdown(cap, count: int = 3):
    for i in range(count, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            d = frame.copy()
            h, w = d.shape[:2]
            cv2.putText(d, str(i),
                        (w // 2 - 45, h // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 230, 230), 10)
            cv2.putText(d, "준비하세요...",
                        (w // 2 - 120, h // 2 + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow(WINDOW, d)
            cv2.waitKey(1)


# ── Segment logic ─────────────────────────────────────────────────────────────

def build_segments(events: list, total_duration: float) -> list:
    """
    events: [(time, 'ood_start'|'ood_end', label), ...]
    → [(start_sec, end_sec, label), ...]
    """
    if not events:
        return [(0.0, round(total_duration, 2), 'normal')]

    segments = []
    cursor = 0.0
    cur_label = 'normal'

    for t, etype, label in events:
        if t - cursor > 0.1:
            segments.append((round(cursor, 2), round(t, 2), cur_label))
        cursor = t
        cur_label = label if etype == 'ood_start' else 'normal'

    if total_duration - cursor > 0.1:
        segments.append((round(cursor, 2), round(total_duration, 2), cur_label))

    return segments


# ── Main recording loop ───────────────────────────────────────────────────────

def record_session(cap, writer, total_sec: float, session: str, subject: str):
    events = []
    current_ood = None
    ood_start = None
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start_time
        writer.write(frame)

        display = draw_ui(frame, elapsed, total_sec, session, subject, current_ood, ood_start)
        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF

        # ── 종료 조건 ─────────────────────────────────────────────────────
        if key in (ord('q'), ord('Q')) or elapsed >= total_sec:
            reason = "사용자 종료" if key in (ord('q'), ord('Q')) else "시간 완료"
            # 열려 있는 OOD 닫기
            if current_ood:
                events.append((round(elapsed, 2), 'ood_end', current_ood))
                print(f"[{elapsed:.1f}s] OOD 자동 종료: {current_ood}")
            print(f"[INFO] 녹화 종료 ({reason}) @ {elapsed:.1f}s")
            break

        # ── Session C: OOD 토글 ───────────────────────────────────────────
        if session == 'C' and key in OOD_KEY_MAP:
            ood_label = OOD_KEY_MAP[key]
            now = round(elapsed, 2)

            if current_ood is None:
                events.append((now, 'ood_start', ood_label))
                current_ood = ood_label
                ood_start = elapsed
                print(f"[{now:.1f}s] ▶ OOD 시작: {ood_label} ({OOD_NAMES[ood_label]})")

            elif current_ood == ood_label:
                events.append((now, 'ood_end', ood_label))
                print(f"[{now:.1f}s] ■ OOD 종료: {ood_label}  (지속 {elapsed - ood_start:.1f}s)")
                current_ood = None
                ood_start = None

            else:
                # 다른 OOD로 전환
                events.append((now, 'ood_end', current_ood))
                events.append((now, 'ood_start', ood_label))
                print(f"[{now:.1f}s] ↔ OOD 전환: {current_ood} → {ood_label}")
                current_ood = ood_label
                ood_start = elapsed

    final_elapsed = time.time() - start_time
    segments = build_segments(events, final_elapsed)
    return segments, final_elapsed


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Study Normality Score 데이터 수집')
    parser.add_argument('--subject',   required=True, help='참가자 번호 (예: 01)')
    parser.add_argument('--session',   required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--data_root', default='data/pilot', help='데이터 저장 루트')
    parser.add_argument('--camera',    type=int, default=0,  help='카메라 인덱스')
    parser.add_argument('--duration',  type=int, default=None,
                        help='녹화 시간(초). 기본: A=600 B=300 C=600 D=300')
    parser.add_argument('--notes',     default='', help='메모')
    args = parser.parse_args()

    session  = args.session.upper()
    total_sec = args.duration or SESSION_DURATIONS[session]

    subj_dir    = Path(args.data_root) / f"subject_{args.subject}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    video_path  = subj_dir / f"session_{session}.mp4"
    labels_path = subj_dir / f"session_{session}_labels.csv"
    meta_path   = subj_dir / f"session_{session}_meta.json"

    if video_path.exists():
        ans = input(f"[WARN] {video_path.name} 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ")
        if ans.strip().lower() != 'y':
            print("취소")
            return

    print(f"\n{'='*55}")
    print(f"  Subject : {args.subject}")
    print(f"  Session : {session}  ({total_sec//60}분)")
    print(f"  저장 경로: {subj_dir.resolve()}")
    if session == 'C':
        print("  OOD 키: 1=스마트폰 2=엎드리기 3=자리비움 4=두리번 5=고개돌리기")
    print(f"{'='*55}\n")

    cap, w, h, fps = setup_camera(args.camera)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print("[ERROR] VideoWriter 초기화 실패.")
        cap.release()
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 960), min(h, 540))

    try:
        if not preview_phase(cap, args.subject, session):
            print("취소")
            return

        countdown(cap)

        segments, actual_duration = record_session(
            cap, writer, total_sec, session, args.subject
        )

        # Sessions A/B/D: 전체가 normal
        if session != 'C':
            segments = [(0.0, round(actual_duration, 2), 'normal')]

        # ── labels.csv ────────────────────────────────────────────────────
        with open(labels_path, 'w', newline='', encoding='utf-8') as f:
            w_ = csv.writer(f)
            w_.writerow(['start_sec', 'end_sec', 'label'])
            for seg in segments:
                w_.writerow(seg)
        print(f"[DONE] labels.csv  → {labels_path}")

        # ── meta.json ─────────────────────────────────────────────────────
        meta = {
            'subject': args.subject,
            'session': session,
            'camera_idx': args.camera,
            'resolution': [w, h],
            'fps': fps,
            'duration_sec': round(actual_duration, 2),
            'notes': args.notes,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[DONE] meta.json   → {meta_path}")

        # ── 요약 출력 ─────────────────────────────────────────────────────
        print(f"\n[완료] {video_path.name}  ({actual_duration:.1f}s)")
        normal_sec = sum(e - s for s, e, l in segments if l == 'normal')
        ood_sec    = sum(e - s for s, e, l in segments if l != 'normal')
        print(f"       normal: {normal_sec:.1f}s  |  OOD: {ood_sec:.1f}s  |  세그먼트: {len(segments)}개")
        for seg in segments:
            print(f"         {seg[0]:7.1f}s ~ {seg[1]:7.1f}s  {seg[2]}")

    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
