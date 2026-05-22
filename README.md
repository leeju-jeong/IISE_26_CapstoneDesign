# Study Normality Score

> 자기주도 학습 장면에서 상체 스켈레톤과 텍스트 프롬프트 정렬을 통해  
> **Study Normality Score** (집중도 점수 0~1)를 산출하는 privacy-aware 시스템

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 참고 논문 | Sato et al., CVPR 2023 — Prompt-Guided Zero-Shot Anomaly Action Recognition |
| 입력 | 웹캠 영상 (720p, 30fps) |
| 포즈 추출 | MediaPipe — 상체 11개 관절 (코, 눈, 귀, 어깨, 팔꿈치, 손목) |
| 출력 | Study Normality Score ∈ [0, 1] (클립 단위, 10초 window) |
| Privacy | 스켈레톤만 사용 — 얼굴 외형·배경 없음 |

---

## 3개 파이프라인

### Pipeline 1 — Baseline (현재 구현: `hwann/baseline`)
> 텍스트 없이 **MotionBERT + Mahalanobis**만으로 OoD 탐지

```
영상 → MediaPipe → (T, 11, 7) skeleton
    → MotionBERT (frozen) → (512,) feature
    → 정상 클립들의 μ, σ 계산 (학습 = 분포 추정)
    → Mahalanobis distance → Normality Score = exp(-γ·dist)
```

- 목적: 기저 성능 측정 (텍스트 없이 얼마나 되는지)
- 예상 한계: 행동 간 semantic gap → t-SNE에서 클러스터 경계 모호

---

### Pipeline 2 — Full (예정)
> MotionBERT + **CLIP 텍스트 정렬** (Contrastive loss)

```
영상 → MediaPipe → skeleton
    → MotionBERT (마지막 레이어 fine-tune)
    → MLP Adapter → CLIP 공간 (512)
    ↙                        ↘
OoD Score                Prompt Score
Mahalanobis(x, μ, σ)    cosine_sim(f(x), CLIP("A student focusing on studying"))
    ↘                        ↙
         OoD × Prompt → Normality Score
```

- 학습: 정상 클립 + 텍스트 프롬프트 쌍으로 Contrastive loss
- 텍스트 프롬프트 4개:
  - "A student is sitting upright and focusing on studying."
  - "A learner is looking at the screen or desk material with sustained attention."
  - "A student is engaged in a normal self-study session."
  - "A person is reading or writing at a desk in a focused manner."

---

### Pipeline 3 — Cross-domain (예정)
> DAD Dataset(운전자 이상행동)으로 cross-domain 검증

- 프롬프트만 교체: "A driver is focusing on the road."
- 기여: "텍스트 프롬프트만 바꾸면 다른 도메인에도 적용 가능"

---

## 데이터셋

### 행동 코드
| 코드 | 행동 | Training | Inference |
|---|---|---|---|
| 1 | 타이핑 | ✅ 정상 | ✅ |
| 2 | 강의시청 | ✅ 정상 | ✅ |
| 3 | 필기 | ✅ 정상 | ✅ |
| 4 | 문제풀기 | ✅ 정상 | ✅ |
| 5 | 패드보기 | ✅ 정상 | ✅ |
| 6 | 딴짓 (스마트폰 등) | ❌ 제외 | ✅ (OOD) |
| 7 | 전환/연결 | ✅ 정상 | ✅ |

### 수집 방법
1. 영상 1개에 행동 1~7 **모두 포함**해서 촬영 (10분, 720p, 30fps)
2. `tools/label_actions.py`로 숫자키 toggle 라벨링
3. `src/pose_extractor.py`로 skeleton.pkl 추출
4. `tools/generate_clips.py`로 행동별 clip 폴더 분류

### 클립 파라미터
- **clip_frames = 64** (~2.1초 @ 30fps)
- **stride = 64** (겹침 없음, 구간 내 독립 클립)
- 출력 shape: `(64, 11, 3)` — (시간, 관절, [x, y, conf])
- 구간 경계 클립 버림 (64프레임 미만 tail 제외)

### 데이터 폴더 구조
```
data/
  raw/
    subject_01.mp4
    subject_01_labels.csv      ← label_actions.py 출력
  skeletons/
    subject_01_skeleton.pkl    ← (T, 11, 7) numpy array
  clips/
    subject_01/
      action_1/clip_00001.npy  ← (64, 11, 3)
      action_2/...
      action_6/...             ← OOD (inference only)
      train_manifest.csv
      test_manifest.csv
  splits.yaml                  ← subject 단위 train/val/test 분리
```

### Subject-independent split
- 같은 사람이 train/test에 동시에 들어가면 과대평가 → 반드시 분리
- Pilot: 2~3명 / 본수집: 10~15명
- 비율: train 8명 / val 2명 / test 2~3명

---

## 평가 지표
- **AUROC**: 정상 vs OOD 이진 분류 성능
- **t-SNE**: 정상/OOD feature 분포 분리 시각화
- **Privacy 실험** (Pipeline 3): RGB baseline 대비 성능 유지율

---

## 브랜치 구조
| 브랜치 | 내용 |
|---|---|
| `main` | full pipeline 코드 (Pipeline 2용) |
| `hwann/data-collection` | 구 데이터 수집 도구 |
| `hwann/baseline` | **Pipeline 1 데모** ← 현재 작업 브랜치 |

---

## 실행 순서 (Pipeline 1 Baseline)

```bash
# 1. 라벨링
python tools/label_actions.py --video data/raw/subject_01.mp4

# 2. 포즈 추출
python src/pose_extractor.py data/raw/subject_01.mp4 data/skeletons/

# 3. 클립 분류
python tools/generate_clips.py \
    --skeleton data/skeletons/subject_01_skeleton.pkl \
    --labels   data/raw/subject_01_labels.csv \
    --output   data/clips/subject_01 \
    --n_train  30

# 4. 분포 추정 (μ, σ)
python src/train_baseline.py --data_root data/clips/subject_01

# 5. 평가 (AUROC + t-SNE)
python src/evaluate_baseline.py --data_root data/clips/subject_01
```

> MotionBERT 사전 준비:
> ```bash
> git clone https://github.com/Walter0807/MotionBERT
> # MB_pretrain.bin → checkpoints/MB_pretrain.bin
> export PYTHONPATH=MotionBERT:$PYTHONPATH
> ```

---

## 관련 자료
- [Sato et al. CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Sato_Prompt-Guided_Zero-Shot_Anomaly_Action_Recognition_Using_Pretrained_Deep_Skeleton_Features_CVPR_2023_paper.pdf)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [DAD Dataset](https://github.com/okankop/Driver-Anomaly-Detection)
