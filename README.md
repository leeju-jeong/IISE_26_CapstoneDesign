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

```bash
# Hugging Face에서 받을 경우 (파일명은 자유)
wget "https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pretrain/MB_lite/latest_epoch.bin" \
  -O models/motionbert/lite_bert.bin
```

- `.gitignore`에 `models/motionbert/` 포함 → Git에는 올리지 않음
- 아키텍처: `MotionBERT/configs/pretrain/MB_lite.yaml` (`dim_feat=256`, `mlp_ratio=4`)
- Lite로 바꾼 뒤에는 `adapter.pth`, `stats.npz` **재학습** 필요

## 저장 파일

- `checkpoints/adapter.pth`
- `checkpoints/stats.npz` — `mu_x, std_x, mu_z, std_z`
