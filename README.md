# IISE_26_CapstoneDesign — Study Normality Score

## 파이프라인

```
영상 (MP4)
    │
    ▼
[pose_extractor.py]
MediaPipe PoseLandmarker (VIDEO 모드)
  - 상체 11개 관절 추출: nose, eyes, ears, shoulders, elbows, wrists
  - confidence < 0.5 프레임 제거
  - 관절별 7차원 벡터: [x, y, time_norm, confidence, joint_idx, centroid_x, centroid_y]
  - 저장: skeleton.pkl  →  shape: (T, 11, 7)
    │
    ▼
[dataset.py]
슬라이딩 윈도우 클립 분할
  - window = 10초, stride = 5초, 매 3프레임 샘플링
  - 클립 shape: (100 frames, 11 joints, 7 dims)
  - labels.csv 와 원본 frame_indices 기준으로 normal / OOD 라벨 매칭
    │
    ▼
[models/feature_extractor.py]  MotionBERTExtractor
  ① MediaPipe 11관절 → H36M 17관절 변환 (하체 관절은 confidence=0으로 채움)
  ② Bounding-box 정규화 → 좌표 범위 [-1, 1]
  ③ DSTformer (MotionBERT, ICCV 2023, pretrained on Human3.6M + NTU60)
  ④ 프레임 × 관절 차원 mean pooling
  출력: (B, 512)
    │
    ▼
[models/adapter.py]  MLPAdapter
  Linear(512 → 512) → ReLU → Linear(512 → 512) → L2 Normalize
  출력: (B, 512)  ← CLIP 공간에 정렬된 벡터
    │
    ▼
[train.py]  학습 (정상 데이터만 사용)
  - OpenCLIP ViT-B-32로 정상 공부 프롬프트 4개를 텍스트 임베딩
  - InfoNCE Loss: skeleton 벡터 ↔ 텍스트 임베딩 간 contrastive 학습
  - 학습 완료 후 훈련 특징 전체로 μ, σ 계산 (Mahalanobis용)
  - 저장: backbone.pth, adapter.pth, stats.npz
    │
    ▼
[inference.py / evaluate.py]  추론 및 평가
  Mahalanobis Distance:
    mahal = ‖(x − μ) / σ‖₂
  CLIP Cosine Similarity:
    cos_sim = z · text_avg
  Normality Score:
    score = exp(−γ/√512 × mahal) × (cos_sim + 1) / 2   ∈ [0, 1]
  Temporal Smoothing → 이동 평균

  출력: outputs/scores.csv, outputs/score_plot.png
  평가: AUROC (normal vs OOD), t-SNE 시각화
```

## 실행

```bash
# 1. 포즈 추출
python src/pose_extractor.py <video.mp4> <output_dir/>

# 2. 학습
python src/train.py --data_root data/ --config configs/default.yaml

# 3. 평가
python src/evaluate.py --data_root data/ --config configs/default.yaml

# 4. 추론
python src/inference.py --video <video.mp4> --config configs/default.yaml
```

## 환경 설정

```bash
bash setup.sh
conda activate study
```
