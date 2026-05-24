"""
행동 어노테이션 툴 — 숫자키 toggle로 1~7번 행동 구간 기록

실행:
  python tools/label_actions.py --video data/people1/recording.mp4

조작법:
  1~7   : 행동 toggle
            · 처음 누름  → 해당 행동 시작
            · 같은 키 다시 → 해당 행동 종료
            · 다른 번호 누름 → 현재 행동 종료 + 새 행동 즉시 시작
  SPACE : 일시정지 / 재개
  A / D : 5초 뒤로 / 앞으로
  Z     : 마지막 확정 구간 취소
  S     : 저장 후 종료
  Q     : 저장 없이 종료

행동 코드:
  1: 타이핑          (Training: 정상)
  2: 강의시청        (Training: 정상)
  3: 필기            (Training: 정상)
  4: 문제풀기        (Training: 정상)
  5: 패드보기        (Training: 정상)
  6: 딴짓/오프태스크 (Inference only — Training 제외)
  7: 전환/연결       (Training: 정상, 갭 자동 삽입)

출력 CSV:
  start_sec, end_sec, action
  0.000, 12.460, 1
  12.460, 15.460, 7   ← 갭 자동 삽입
  15.460, 45.000, 2

출력 파일 위치: --video 영상과 같은 폴더, {영상이름}_labels.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ACTION_NAMES = {
    1: "Typing", 2: "Watching", 3: "Writing",
    4: "Problem", 5: "Pad", 6: "OFF-TASK", 7: "Transition",
}
ACTION_COLORS = {
    1: (60, 200, 60),
    2: (200, 140, 40),
    3: (40, 200, 200),
    4: (200, 60, 200),
    5: (100, 200, 255),
    6: (30, 30, 220),
    7: (130, 130, 130),
}
WINDOW = "Label Actions  [1-7 toggle | SPACE | A/D:seek | Z:undo | S:save | Q:quit]"


class Labeler:
    def __init__(self):
        self.segments: list[tuple[float, float, int]] = []
        self.active_action: int | None = None
        self.active_start: float = 0.0

    def press(self, action: int, t: float) -> str | None:
        MIN_DUR = 0.3

        if self.active_action is None:
            self.active_action = action
            self.active_start = t
            return f"[{t:.2f}s] ▶ Action {action} ({ACTION_NAMES[action]}) 시작"

        elif self.active_action == action:
            dur = t - self.active_start
            if dur < MIN_DUR:
                self.active_action = None
                return f"[{t:.2f}s] ⚠ {dur:.2f}s 구간 너무 짧아 무시"
            self.segments.append((self.active_start, t, action))
            msg = f"[{t:.2f}s] ■ Action {action} 종료  {self.active_start:.2f}→{t:.2f}s ({dur:.1f}s)"
            self.active_action = None
            return msg

        else:
            dur = t - self.active_start
            msgs = []
            if dur >= MIN_DUR:
                self.segments.append((self.active_start, t, self.active_action))
                msgs.append(f"[{t:.2f}s] ■ Action {self.active_action} 자동 종료 ({dur:.1f}s)")
            self.active_action = action
            self.active_start = t
            msgs.append(f"[{t:.2f}s] ▶ Action {action} ({ACTION_NAMES[action]}) 시작")
            return "\n".join(msgs)

    def undo(self, t: float) -> str:
        if self.active_action is not None:
            msg = f"[UNDO] 진행 중 Action {self.active_action} 취소 (시작: {self.active_start:.2f}s)"
            self.active_action = None
            return msg
        elif self.segments:
            s, e, a = self.segments.pop()
            return f"[UNDO] 구간 제거: Action {a}  {s:.2f}~{e:.2f}s"
        return "[UNDO] 취소할 구간 없음"

    def finalize(self, t: float):
        if self.active_action is not None and t - self.active_start >= 0.3:
            self.segments.append((self.active_start, t, self.active_action))
            self.active_action = None

    def fill_gaps(self, total_sec: float, gap_min: float = 0.15) -> list:
        segs = sorted(self.segments, key=lambda x: x[0])
        filled = []
        prev_end = 0.0
        for start, end, action in segs:
            if start > prev_end + gap_min:
                filled.append((prev_end, start, 7))
            filled.append((start, end, action))
            prev_end = end
        if total_sec - prev_end > gap_min:
            filled.append((prev_end, total_sec, 7))
        return filled


def draw_ui(frame, cur_sec, total_sec, labeler: Labeler, paused: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    d = frame.copy()

    bar_y = h - 20
    bx0, bx1 = 8, w - 8
    bw = bx1 - bx0
    cv2.rectangle(d, (bx0, bar_y - 7), (bx1, bar_y + 7), (40, 40, 40), -1)

    for s, e, a in labeler.segments:
        x0 = bx0 + int(s / total_sec * bw)
        x1 = bx0 + int(e / total_sec * bw)
        cv2.rectangle(d, (x0, bar_y - 7), (x1, bar_y + 7), ACTION_COLORS[a], -1)

    if labeler.active_action is not None:
        x0 = bx0 + int(labeler.active_start / total_sec * bw)
        x1 = bx0 + int(cur_sec / total_sec * bw)
        col = ACTION_COLORS[labeler.active_action]
        cv2.rectangle(d, (x0, bar_y - 7), (x1, bar_y + 7), col, -1)
        if int(cur_sec * 4) % 2 == 0:
            cv2.rectangle(d, (x1 - 2, bar_y - 10), (x1 + 2, bar_y + 10), (255, 255, 255), -1)

    cx = bx0 + int(cur_sec / total_sec * bw)
    cv2.line(d, (cx, bar_y - 11), (cx, bar_y + 11), (255, 255, 255), 2)

    cv2.rectangle(d, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(d, 0.65, frame, 0.35, 0, d)

    def fmt(s):
        m = int(s) // 60
        return f"{m:02d}:{s - m * 60:05.2f}"

    cv2.putText(d, f"{fmt(cur_sec)} / {fmt(total_sec)}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if paused:
        cv2.putText(d, "PAUSED", (w - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    if labeler.active_action is not None:
        a = labeler.active_action
        dur = cur_sec - labeler.active_start
        cv2.putText(d, f"[{a}] {ACTION_NAMES[a]}  {labeler.active_start:.2f}s → {cur_sec:.2f}s  ({dur:.1f}s)",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ACTION_COLORS[a], 2)
    else:
        cv2.putText(d, "Press 1-7 to start  |  same key to end",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)

    n = len(labeler.segments)
    cv2.putText(d, f"Segments: {n}", (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    if labeler.segments:
        s, e, a = labeler.segments[-1]
        cv2.putText(d, f"Last: [{a}]{ACTION_NAMES[a]} {s:.1f}~{e:.1f}s",
                    (180, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACTION_COLORS[a], 1)

    return d


def save_csv(segments: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "action"])
        for s, e, a in segments:
            writer.writerow([f"{s:.3f}", f"{e:.3f}", a])

    total = sum(e - s for s, e, a in segments)
    print(f"\n[SAVED] {path}")
    print(f"  구간 수: {len(segments)}개  /  총 커버: {total:.1f}s")

    from collections import defaultdict
    dur_by_action: dict = defaultdict(float)
    for s, e, a in segments:
        dur_by_action[a] += e - s
    for a in sorted(dur_by_action):
        dur = dur_by_action[a]
        tag = " ← anomaly (inference only)" if a == 6 else ""
        print(f"  [{a}] {ACTION_NAMES[a]:15s}  {dur:5.1f}s{tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="라벨링할 영상 (.mp4)")
    parser.add_argument("--output", default=None, help="출력 CSV 경로 (기본: 영상 폴더)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] 파일 없음: {video_path}")
        sys.exit(1)

    output_csv = args.output or str(video_path.parent / (video_path.stem + "_labels.csv"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_f / fps
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(960 / vid_w, 600 / vid_h)
    w_out, h_out = int(vid_w * scale), int(vid_h * scale)

    print(f"[INFO] {video_path.name}  {vid_w}×{vid_h} @ {fps:.1f}fps  {total_sec:.1f}s")
    print("조작: 1~7 toggle | SPACE 정지 | A/D 5초이동 | Z 취소 | S 저장 | Q 종료\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w_out, h_out + 10)

    labeler = Labeler()
    paused = True
    cur_f = 0
    ret, frame = cap.read()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_f - 1)
                ret, frame = cap.read()
            else:
                cur_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_f)
            ret, frame = cap.read()
            if not ret:
                break

        cur_sec = cur_f / fps
        disp = cv2.resize(frame, (w_out, h_out))
        disp = draw_ui(disp, cur_sec, total_sec, labeler, paused)
        cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key == 32:
            paused = not paused
        elif key in (81, ord('a'), ord('A')):
            cur_f = max(0, cur_f - int(fps * 5))
        elif key in (83, ord('d'), ord('D')):
            cur_f = min(total_f - 1, cur_f + int(fps * 5))
        elif chr(key) in "1234567":
            msg = labeler.press(int(chr(key)), cur_sec)
            if msg:
                print(msg)
        elif key in (ord('z'), ord('Z')):
            print(labeler.undo(cur_sec))
        elif key in (ord('s'), ord('S')):
            labeler.finalize(cur_sec)
            filled = labeler.fill_gaps(total_sec)
            save_csv(filled, output_csv)
            break
        elif key in (ord('q'), ord('Q')):
            print("[종료] 저장 안 함")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
