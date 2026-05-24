# Study Normality Score

## 파이프라인 요약

| 단계 | 내용 |
|------|------|
| [1] Pose | MP4 → MediaPipe → `skeleton.pkl (T,11,7)` |
| [2] Clip | window 5s / stride 3s / 10fps → `(50,11,7)` |
| [3] MB 입력 | MP11→H36M17, bbox norm → `(B,T,17,2)` |
| [4] Feature | MotionBERT frozen → `x ∈ R^512` |
| [5] Stats x | 정상 train → `μ_x, σ_x` |
| [6] Adapter | `loss = align + 0.1×preserve` |
| [7] Stats z | 정상 train → `μ_z, σ_z` |
| [8–9] Score | `score_x`, `score_z`, `score_text` → fusion |
| [10] Eval | AUROC, AUPRC, timeline, t-SNE |

### Score fusion (`inference.fusion`)

- `linear_50_50`: `0.5·score_x + 0.5·score_z`
- `linear_40_40_20`: `0.4·score_x + 0.4·score_z + 0.2·score_text` (기본)
- `product`: `score_x × score_z × score_text`

`anomaly_score = 1 - normality_score`

## 실행

```bash
conda activate /home/storage/leeju2/envs/study

python src/pose_extractor.py video.mp4 data/subject_01/ --config configs/default.yaml
python src/train.py --data_root data/ --config configs/default.yaml
python src/evaluate.py --data_root data/ --ckpt_dir checkpoints
python src/inference.py --video video.mp4 --ckpt_dir checkpoints
```

### 10s 비교 실험

```bash
python src/train.py --data_root data/ --config configs/default_10s.yaml
```

## MotionBERT 가중치 (MB_lite)

**저장 위치:** `models/motionbert/lite_bert.bin`

## Task 1 — 정상 분포 (μ_x, σ_x)

데이터 준비: [data/README.md](data/README.md)

```bash
python src/pose_extractor.py data/videos/ data/skeletons/ --config configs/task1.yaml
python src/compute_stats_x.py --data_root data/ --config configs/task1.yaml
python src/verify_stats_x.py --data_root data/ --config configs/task1.yaml
```

출력: `checkpoints/task1/stats_x.npz`, `outputs/task1_*.png`

## 저장 파일 (Task 2)

- `checkpoints/adapter.pth`
- `checkpoints/stats.npz` — `mu_x, std_x, mu_z, std_z`
