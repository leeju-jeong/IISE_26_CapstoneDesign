# IISE_26_CapstoneDesign — Study Normality Score

## 파이프라인

```
[Video Input]
    ↓ 720p 이상, 30fps, 카메라 정면 / 가로(landscape) 고정
[MediaPipe PoseLandmarker]
    ↓ 상체 11개 관절 추출
    머리 5개: 코, 눈(좌우), 귀(좌우)
    팔  6개: 어깨(좌우), 팔꿈치(좌우), 손목(좌우)
    confidence < 0.5 프레임 자동 제거
[7D 벡터 변환]
    ↓ v = [x, y, time_norm, confidence, joint_index, centroid_x, centroid_y]
    클립 단위 분할: window=10s, stride=5s, 3프레임마다 1샘플
    → 클립 shape: (100 frames, 11 joints, 7 dims)
[MotionBERT — DSTformer]  ← pretrained on Human3.6M + NTU60 (ICCV 2023)
    ↓ MediaPipe 11관절 → H36M 17관절 변환 후 입력
    ↓ Bounding-box 정규화 [-1, 1]
    ↓ 마지막 proj layer만 학습, backbone frozen
    x ∈ R^512
[MLP Adapter]  ← 학습 대상
    ↓ Contrastive loss로 학습 (정상 클립 + 텍스트 프롬프트 쌍)
    ↓ InfoNCE: clip-to-text + clip-to-clip
    f(x) ∈ R^512  (L2 normalized, CLIP 텍스트 임베딩 공간)
        ↙                              ↘
[OoD Score]                      [Prompt Score]
Mahalanobis(x, μ, σ)             cosine_sim(f(x), text_avg)
  = √Σ((x−μ)²/σ²)               text_avg: 정상 프롬프트 4개 평균
μ, σ: 정상 데이터 전체로 계산     ∈ [-1, 1] → [0, 1] 선형 변환
        ↘                              ↙
              [Score Fusion]
    score = exp(−γ/√512 × mahal) × (cos_sim + 1) / 2
    + Temporal Smoothing (이동 평균)
        ↓
[Study Normality Score] ∈ [0, 1]
    0: 이상 행동 (졸음, 스마트폰, 이석 등)
    1: 정상 학습 (집중, 필기, 화면 응시)
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
