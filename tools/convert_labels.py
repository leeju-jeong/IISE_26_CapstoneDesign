"""
label_actions.py 출력 → v1 dataset 형식 변환

label_actions.py 출력 (our format):
  start_sec, end_sec, action
  0.000, 12.460, 1
  12.460, 15.460, 7
  350.000, 430.000, 6

v1 dataset 기대 형식:
  start_sec, end_sec, label
  0.000, 12.460, normal
  12.460, 15.460, normal
  350.000, 430.000, OOD_01

실행:
  python tools/convert_labels.py --input data/people1/recording_labels.csv
  → data/people1/recording_labels_v1.csv 생성

매핑:
  action 1~5, 7 → normal
  action 6      → OOD_01  (딴짓/오프태스크)
"""
import argparse
import csv
from pathlib import Path

NORMAL_ACTIONS = {1, 2, 3, 4, 5, 7}


def convert(input_path: str, output_path: str):
    rows = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = row["start_sec"].strip()
            end   = row["end_sec"].strip()
            action = int(row["action"].strip())
            label = "normal" if action in NORMAL_ACTIONS else "OOD_01"
            rows.append((start, end, label))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "label"])
        writer.writerows(rows)

    n_normal = sum(1 for _, _, l in rows if l == "normal")
    n_ood    = len(rows) - n_normal
    print(f"[DONE] {output_path}")
    print(f"  normal: {n_normal}개 구간  |  OOD_01: {n_ood}개 구간")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="label_actions.py 출력 CSV")
    parser.add_argument("--output", default=None,  help="변환 결과 저장 경로 (기본: 원본_v1.csv)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else \
                  input_path.parent / (input_path.stem + "_v1.csv")

    convert(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
