"""
splits.yaml 생성기 — 수집 완료 후 실행

Usage:
  python tools/make_splits.py --data_root data --train people1 --val people2 --test people2

결과: data/splits.yaml
"""
import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description='splits.yaml 생성')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--train', nargs='+', required=True, metavar='FOLDER',
                        help='train 폴더명 (예: people1 people2)')
    parser.add_argument('--val',   nargs='+', required=True, metavar='FOLDER')
    parser.add_argument('--test',  nargs='+', required=True, metavar='FOLDER')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    def to_dir(folders):
        dirs = []
        for name in folders:
            d = data_root / name
            if not d.is_dir():
                print(f"[WARN] {d} 폴더가 없습니다. 계속 진행.")
            dirs.append(name)
        return dirs

    splits = {
        'train': to_dir(args.train),
        'val':   to_dir(args.val),
        'test':  to_dir(args.test),
    }


    out_path = data_root / 'splits.yaml'
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(splits, f, allow_unicode=True, default_flow_style=False)

    print(f"[DONE] {out_path}")
    print(f"  train : {splits['train']}")
    print(f"  val   : {splits['val']}")
    print(f"  test  : {splits['test']}")


if __name__ == '__main__':
    main()
