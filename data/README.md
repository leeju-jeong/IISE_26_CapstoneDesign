# Task 1 데이터 준비 (flat 레이아웃)

## 디렉터리

```
data/
├── videos/          ← 정상 mp4 넣기
├── skeletons/       ← pose_extractor 출력 (자동 생성)
└── labels.csv       ← video_id별 normal 라벨
```

## labels.csv 형식

```csv
video_id,start_sec,end_sec,label
my_video_01,0,9999,normal
my_video_02,0,9999,normal
```

- `video_id` = mp4 파일명에서 `.mp4` 제외 (`my_video_01.mp4` → `my_video_01`)
- 이번 단계는 **normal만** (OOD 없음)

## 실행 순서 (데이터 준비 후)

```bash
conda activate /home/storage/leeju2/envs/study
cd /home/leeju2/Segformer

# 1) skeleton 추출
python src/pose_extractor.py data/videos/ data/skeletons/ --config configs/task1.yaml

# 2) μ_x, σ_x 계산
python src/compute_stats_x.py --data_root data/ --config configs/task1.yaml

# 3) sanity check
python src/verify_stats_x.py --data_root data/ --config configs/task1.yaml
```

## clip 설정 (configs/task1.yaml)

- window 2s, stride 0 (non-overlapping), 6fps → clip shape `(12, 11, 7)`
