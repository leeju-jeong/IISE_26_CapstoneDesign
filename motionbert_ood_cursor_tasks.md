# MotionBERT 기반 Skeleton OOD 탐지 프로젝트 진행 흐름

## 0. 지금 우리가 만들고 있는 것

이 프로젝트의 목표는 **공부 중인 사람의 상체 skeleton sequence가 정상 공부 패턴인지, 정상 분포에서 벗어난 OOD 상태인지 판단하는 시스템**을 만드는 것이다.

전체 아이디어는 다음과 같다.

```text
MP4 영상
→ MediaPipe로 상체 11관절 추출
→ sliding window로 skeleton clip 생성
→ MotionBERT로 clip-level motion feature 추출
→ 1번: MotionBERT 원본 feature 공간에서 정상 분포 계산
→ 2번: MLP Adapter를 학습한 뒤 adapter feature 공간에서 정상 분포 계산
→ 두 score를 결합해서 최종 normality / anomaly score 계산
```

최종적으로 보고 싶은 값은 다음이다.

```text
normality_score 높음 → 정상 공부 패턴에 가까움
normality_score 낮음 → OOD / 비집중 가능성 높음

anomaly_score = 1 - normality_score
```

---

# 전체 개발 흐름 요약

## Task 1. MotionBERT-only baseline 만들기

목표:

```text
Adapter 없이 MotionBERT feature만으로 정상 분포를 만들고,
test clip이 정상 분포에서 얼마나 벗어나는지 확인한다.
```

이 단계는 전체 프로젝트의 **기본 baseline**이다.

이 단계에서 확인해야 하는 질문:

```text
MotionBERT feature만으로 정상 공부와 비집중/OOD가 어느 정도 분리되는가?
```

---

## Task 2. MLP Adapter 추가하기

목표:

```text
MotionBERT feature 뒤에 MLP Adapter를 붙이고,
정상 공부 text prompt embedding과 가까워지도록 학습한다.
그 후 adapter feature 공간에서 두 번째 정상 분포를 만든다.
```

이 단계에서 확인해야 하는 질문:

```text
Adapter를 붙였을 때 OOD 탐지 성능이 좋아지는가?
Text prompt alignment가 실제로 도움이 되는가?
```

---

# Task 1. MotionBERT-only baseline

## Task 1의 최종 출력물

Task 1이 끝나면 아래 파일들이 생성되어야 한다.

```text
data/skeletons/*.pkl
outputs/task1_scores.csv
outputs/task1_score_plot.png
outputs/task1_tsne.png
checkpoints/stats_x.npz
```

각 파일의 의미는 다음과 같다.

```text
skeleton.pkl:
영상에서 추출한 상체 skeleton sequence

stats_x.npz:
MotionBERT 원본 feature x의 정상 분포
- mu_x
- sigma_x

task1_scores.csv:
각 clip별 score_x, anomaly_score 기록

task1_score_plot.png:
시간에 따른 score 변화 시각화

task1_tsne.png:
normal/OOD feature 분포 시각화
```

---

## Task 1-1. 프로젝트 폴더 구조 만들기

Cursor에서 먼저 아래 구조를 만든다.

```text
project/
│
├── data/
│   ├── videos/
│   ├── skeletons/
│   └── labels.csv
│
├── models/
│   ├── motionbert_extractor.py
│   └── scoring.py
│
├── scripts/
│   ├── pose_extractor.py
│   ├── build_dataset.py
│   ├── extract_motionbert_features.py
│   ├── compute_stats_x.py
│   ├── evaluate_task1.py
│   └── visualize_task1.py
│
├── checkpoints/
│
├── outputs/
│
└── configs/
    └── default.yaml
```

---

## Task 1-2. 입력 영상 준비

`data/videos/` 아래에 영상을 넣는다.

예시:

```text
data/videos/
├── normal_reading_01.mp4
├── normal_writing_01.mp4
├── normal_laptop_01.mp4
├── ood_phone_01.mp4
├── ood_drowsy_01.mp4
└── ood_away_01.mp4
```

처음에는 영상이 많지 않아도 된다.

최소 구성:

```text
normal 영상 2~3개
OOD 영상 2~3개
```

단, 학습/분포 계산에는 normal 영상만 사용한다.

---

## Task 1-3. labels.csv 작성

`data/labels.csv`를 만든다.

추천 형식:

```csv
video_id,start_sec,end_sec,label
normal_reading_01,0,120,normal
normal_writing_01,0,120,normal
normal_laptop_01,0,120,normal
ood_phone_01,0,60,ood
ood_drowsy_01,0,60,ood
ood_away_01,0,60,ood
```

처음에는 영상 전체를 normal 또는 ood로 라벨링해도 된다.

나중에는 특정 구간만 OOD로 지정할 수 있다.

예시:

```csv
video_id,start_sec,end_sec,label
study_mixed_01,0,30,normal
study_mixed_01,30,50,phone
study_mixed_01,50,80,normal
```

---

## Task 1-4. pose_extractor.py 구현

역할:

```text
MP4 영상
→ MediaPipe PoseLandmarker
→ 상체 11관절 추출
→ skeleton.pkl 저장
```

저장 shape:

```text
skeleton: (T, 11, 7)
```

상체 11관절:

```text
0  nose
1  left_eye
2  right_eye
3  left_ear
4  right_ear
5  left_shoulder
6  right_shoulder
7  left_elbow
8  right_elbow
9  left_wrist
10 right_wrist
```

각 관절의 7차원 vector:

```text
[x, y, time_norm, confidence, joint_idx, centroid_x, centroid_y]
```

주의점:

```text
confidence 낮은 프레임을 통째로 제거하지 말 것.
프레임은 유지하고, confidence 낮은 joint만 보간하거나 confidence 값으로 관리할 것.
```

저장 dict 예시:

```python
{
    "video_id": "normal_reading_01",
    "fps": 30,
    "frame_indices": [0, 1, 2, ...],
    "joints": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    ],
    "skeleton": skeleton_array  # shape: (T, 11, 7)
}
```

완료 조건:

```text
data/skeletons/normal_reading_01.pkl 생성
data/skeletons/normal_writing_01.pkl 생성
data/skeletons/ood_phone_01.pkl 생성
```

---

## Task 1-5. build_dataset.py 구현

역할:

```text
skeleton.pkl
→ sliding window clip 생성
→ clip별 label 매칭
```

기본 설정:

```yaml
window_sec: 5
stride_sec: 3
sampled_fps: 10
clip_frames: 50
```

즉:

```text
5초 window × 10fps = 50 frames
```

출력 clip shape:

```text
clip: (50, 11, 7)
```

clip 생성 예시:

```text
Clip 1: 0초 ~ 5초
Clip 2: 3초 ~ 8초
Clip 3: 6초 ~ 11초
Clip 4: 9초 ~ 14초
```

라벨 매칭 규칙:

```text
OOD overlap ratio >= 0.5 → ood
OOD overlap ratio <= 0.2 → normal
그 사이 → ambiguous로 제외
```

초기 구현에서는 더 단순하게 해도 된다.

```text
clip 구간 대부분이 normal label이면 normal
clip 구간 대부분이 ood label이면 ood
```

완료 조건:

```text
각 video에서 clip들이 생성됨
각 clip에 video_id, start_sec, end_sec, label이 붙음
```

---

## Task 1-6. MotionBERT 입력 변환 구현

MotionBERT는 입력으로 다음 형태를 기대한다.

```text
(B, T, 17, 3)
```

여기서:

```text
B = batch size
T = frame 수, 예: 50
17 = H36M format keypoints
3 = x, y, confidence
```

우리의 clip은 다음 형태다.

```text
(B, T, 11, 7)
```

따라서 변환이 필요하다.

변환 흐름:

```text
clip_7d: (B, T, 11, 7)
↓
[x, y, confidence]만 추출
upper_11: (B, T, 11, 3)
↓
H36M-like 17관절로 변환
motionbert_input: (B, T, 17, 3)
```

11관절에서 가져올 값:

```text
x = clip[..., 0]
y = clip[..., 1]
confidence = clip[..., 3]
```

H36M 17관절 매핑 초안:

```text
0  pelvis          = missing
1  right_hip       = missing
2  right_knee      = missing
3  right_ankle     = missing
4  left_hip        = missing
5  left_knee       = missing
6  left_ankle      = missing
7  spine           = midpoint(left_shoulder, right_shoulder)
8  thorax          = midpoint(left_shoulder, right_shoulder)
9  neck            = midpoint(shoulder_mid, nose)
10 head            = nose
11 left_shoulder   = left_shoulder
12 left_elbow      = left_elbow
13 left_wrist      = left_wrist
14 right_shoulder  = right_shoulder
15 right_elbow     = right_elbow
16 right_wrist     = right_wrist
```

missing joint 처리:

```text
x = 0
y = 0
confidence = 0
```

중요:

```text
하체 missing joint는 bbox normalization 계산에서 제외할 것.
```

---

## Task 1-7. MotionBERT feature 추출

사용할 기능:

```python
E = MotionBERT.get_representation(x)
```

입력:

```text
x: (B, T, 17, 3)
```

출력:

```text
E: (B, T, 17, 512)
```

우리는 clip-level feature 하나가 필요하므로 pooling한다.

추천:

```text
masked mean pooling over T and valid joints
```

출력:

```text
x_feat: (B, 512)
```

처음 구현에서는 단순 평균으로 시작해도 된다.

```python
x_feat = E.mean(dim=(1, 2))
```

하지만 최종적으로는 confidence > 0인 joint만 평균내는 masked mean pooling이 더 좋다.

완료 조건:

```text
각 clip마다 MotionBERT feature x_feat 생성
x_feat.shape = (512,)
```

---

## Task 1-8. 정상 분포 stats_x 계산

정상 clip만 사용한다.

```text
normal clips
↓
MotionBERT
↓
x_normal: (N, 512)
```

계산:

```python
mu_x = x_normal.mean(axis=0)
sigma_x = x_normal.std(axis=0)
```

저장:

```text
checkpoints/stats_x.npz
```

파일 내용:

```text
mu_x: (512,)
sigma_x: (512,)
```

주의:

```text
sigma_x가 0에 가까운 값은 eps를 더해서 나눗셈 안정화
```

---

## Task 1-9. Task 1 score 계산

추론 clip에 대해:

```python
mahal_x = ||(x_test - mu_x) / (sigma_x + eps)||_2
score_x = exp(-gamma_x / sqrt(512) * mahal_x)
```

의미:

```text
score_x 높음 → MotionBERT 원본 feature 공간에서 정상에 가까움
score_x 낮음 → 정상 분포에서 벗어남
```

OOD 평가에서는:

```python
anomaly_score = 1 - score_x
```

완료 조건:

```text
outputs/task1_scores.csv 생성
```

CSV 예시:

```csv
video_id,clip_start_sec,clip_end_sec,label,mahal_x,score_x,anomaly_score
normal_reading_01,0,5,normal,5.12,0.81,0.19
normal_reading_01,3,8,normal,5.44,0.79,0.21
ood_phone_01,0,5,ood,11.87,0.38,0.62
```

---

## Task 1-10. Task 1 평가

필수 평가:

```text
AUROC
AUPRC
normal/OOD score histogram
score timeline plot
t-SNE visualization
```

Task 1에서 확인할 것:

```text
MotionBERT-only score_x로 normal/OOD가 어느 정도 분리되는가?
```

Task 1 결과가 좋으면:

```text
MotionBERT feature 자체가 정상 공부와 비집중을 어느 정도 구분하고 있음.
```

Task 1 결과가 나쁘면:

```text
상체 11관절 → 17관절 변환 문제
MotionBERT domain mismatch 문제
정상 데이터 부족
OOD가 skeleton상 normal과 너무 유사함
```

을 점검한다.

---

# Task 2. MLP Adapter 추가

## Task 2의 최종 출력물

Task 2가 끝나면 아래 파일들이 생성되어야 한다.

```text
checkpoints/adapter.pth
checkpoints/stats_z.npz
checkpoints/text_avg.npy
outputs/task2_scores.csv
outputs/task2_score_plot.png
outputs/task2_tsne.png
```

---

## Task 2-1. OpenCLIP text embedding 생성

정상 공부 prompt를 만든다.

```text
normal_prompts:
- a student is studying
- a student is reading a book
- a student is writing notes
- a student is solving problems at a desk
- a student is looking at a laptop for studying
```

OpenCLIP text encoder로 embedding을 뽑는다.

```text
text_embs: (5, 512)
```

평균을 낸다.

```python
text_avg = mean(text_embs)
text_avg = normalize(text_avg)
```

저장:

```text
checkpoints/text_avg.npy
```

---

## Task 2-2. Adapter 모델 구현

파일:

```text
models/adapter.py
```

구조:

```text
Input: x_feat (B, 512)

Linear(512 → 512)
LayerNorm(512)
ReLU
Dropout(0.1)
Linear(512 → 512)
L2 Normalize

Output: z (B, 512)
```

역할:

```text
MotionBERT feature x를 CLIP text embedding과 비교 가능한 z로 변환한다.
```

---

## Task 2-3. Adapter 학습 데이터

학습에는 정상 clip만 사용한다.

```text
train data = normal clips only
```

OOD clip은 학습에 사용하지 않는다.

OOD clip은 평가에만 사용한다.

---

## Task 2-4. Adapter loss 설계

기본 loss:

```python
loss_align = 1 - cosine_similarity(z, text_avg)
```

의미:

```text
정상 skeleton embedding z가 정상 공부 text embedding과 가까워지도록 학습한다.
```

보존 loss:

```python
loss_preserve = 1 - cosine_similarity(z, normalize(x))
```

의미:

```text
Adapter가 MotionBERT 원본 feature 구조를 너무 망가뜨리지 않도록 한다.
```

최종 loss:

```python
loss_total = loss_align + 0.1 * loss_preserve
```

처음에는 이 loss만 사용한다.

---

## Task 2-5. 학습 시 augmentation

학습 시에만 약한 skeleton augmentation을 적용한다.

추천:

```text
coordinate jitter: std = 0.01
translation: ±0.03
scale: 0.9 ~ 1.1
joint dropout: p = 0.05 ~ 0.10
horizontal flip: optional
```

주의:

```text
평가/추론에는 augmentation 사용하지 않음.
stats 계산은 clean normal clip 기준으로 먼저 수행.
```

---

## Task 2-6. Adapter 학습 과정

학습 흐름:

```text
normal clip
↓
MotionBERT frozen
↓
x_feat: (B, 512)
↓
Adapter trainable
↓
z: (B, 512)
↓
loss_align + 0.1 loss_preserve
↓
adapter만 업데이트
```

freeze 대상:

```text
MotionBERT: freeze
OpenCLIP text encoder: freeze
```

학습 대상:

```text
MLP Adapter only
```

완료 조건:

```text
checkpoints/adapter.pth 저장
```

---

## Task 2-7. Adapter feature 정상 분포 stats_z 계산

Adapter 학습 후 clean normal clip을 다시 통과시킨다.

```text
normal clips
↓
MotionBERT
↓
Adapter
↓
z_normal: (N, 512)
```

계산:

```python
mu_z = z_normal.mean(axis=0)
sigma_z = z_normal.std(axis=0)
```

저장:

```text
checkpoints/stats_z.npz
```

내용:

```text
mu_z: (512,)
sigma_z: (512,)
```

---

## Task 2-8. Task 2 score 계산

추론 clip에 대해:

```text
clip
↓
MotionBERT
↓
x_test
↓
Adapter
↓
z_test
```

계산 1: MotionBERT-only score

```python
mahal_x = ||(x_test - mu_x) / (sigma_x + eps)||_2
score_x = exp(-gamma_x / sqrt(512) * mahal_x)
```

계산 2: Adapter-space score

```python
mahal_z = ||(z_test - mu_z) / (sigma_z + eps)||_2
score_z = exp(-gamma_z / sqrt(512) * mahal_z)
```

계산 3: Text similarity score

```python
cos_sim = dot(z_test, text_avg)
score_text = (cos_sim + 1) / 2
```

최종 score 후보:

```python
normality_score_1 = score_x

normality_score_2 = score_z

normality_score_fusion = 0.5 * score_x + 0.5 * score_z

normality_score_full = 0.4 * score_x + 0.4 * score_z + 0.2 * score_text

normality_score_product = score_x * score_z * score_text
```

OOD 평가:

```python
anomaly_score = 1 - normality_score
```

---

## Task 2-9. Task 2 평가

반드시 비교할 것:

```text
S1. score_x only
S2. score_z only
S3. score_text only
S4. 0.5 score_x + 0.5 score_z
S5. 0.4 score_x + 0.4 score_z + 0.2 score_text
S6. score_x × score_z × score_text
```

각각 AUROC를 계산한다.

확인 질문:

```text
Adapter를 붙인 score_z가 score_x보다 좋아졌는가?

score_text가 단독으로 의미 있는가?

fusion score가 단일 score보다 좋아졌는가?

product fusion이 너무 엄격해서 normal 오탐을 늘리지는 않는가?
```

---

# Task 1과 Task 2의 차이

| 구분 | Task 1 | Task 2 |
|---|---|---|
| 목적 | MotionBERT-only baseline | Adapter 추가 후 성능 개선 |
| 학습 | 없음 | Adapter만 학습 |
| 사용 feature | x = MotionBERT feature | z = Adapter(x) |
| 정상 분포 | μ_x, σ_x | μ_z, σ_z |
| text prompt | 사용 안 함 | normal prompt 사용 |
| loss | 없음 | cosine alignment + preserve |
| score | score_x | score_z, score_text, fusion |
| 핵심 질문 | MotionBERT만으로 OOD가 되나? | Adapter가 도움이 되나? |

---

# Cursor 작업 순서

## 먼저 Task 1만 끝내기

Cursor에서는 처음부터 Task 2까지 한 번에 만들지 말고, Task 1을 먼저 끝낸다.

### Cursor Prompt 1

```text
현재 프로젝트에 MotionBERT-only OOD baseline을 구현하려고 한다.
먼저 폴더 구조를 만들고, 다음 파일들을 생성해줘.

- scripts/pose_extractor.py
- scripts/build_dataset.py
- models/motionbert_extractor.py
- models/scoring.py
- scripts/compute_stats_x.py
- scripts/evaluate_task1.py
- configs/default.yaml

목표는 MP4 영상에서 MediaPipe 상체 11관절을 추출해 skeleton.pkl을 저장하고,
sliding window로 (50, 11, 7) clip을 만든 뒤,
MotionBERT 입력 형식인 (B, 50, 17, 3)으로 변환해서 feature x를 뽑는 것이다.

Task 1에서는 adapter를 구현하지 않는다.
정상 clip의 MotionBERT feature x로 mu_x, sigma_x를 계산하고,
test clip에 대해 mahal_x, score_x, anomaly_score를 계산하는 baseline만 만든다.
```

---

## Task 1이 돌아간 다음 Task 2 진행

### Cursor Prompt 2

```text
Task 1 MotionBERT-only baseline이 동작한다고 가정하고,
이제 Task 2로 MLP Adapter를 추가해줘.

추가할 파일:
- models/adapter.py
- scripts/build_text_embedding.py
- scripts/train_adapter.py
- scripts/compute_stats_z.py
- scripts/evaluate_task2.py

Adapter 구조:
Linear(512, 512) → LayerNorm → ReLU → Dropout(0.1) → Linear(512, 512) → L2 Normalize

학습:
MotionBERT는 freeze하고 Adapter만 학습한다.
OpenCLIP text encoder로 normal prompts embedding을 만들고 평균 text_avg를 사용한다.

Loss:
loss_align = 1 - cosine_similarity(z, text_avg)
loss_preserve = 1 - cosine_similarity(z, normalize(x))
loss_total = loss_align + 0.1 * loss_preserve

학습 후 normal clip의 z로 mu_z, sigma_z를 계산한다.

평가:
score_x, score_z, score_text, fusion score를 모두 계산하고 AUROC를 비교한다.
```

---

# 최종적으로 우리가 확인해야 할 것

## Task 1에서 확인

```text
MotionBERT 원본 feature만으로 normal/OOD가 구분되는가?
```

결과가 좋으면:

```text
MotionBERT feature가 공부 skeleton OOD에 어느 정도 의미 있음.
```

결과가 나쁘면:

```text
입력 변환, 정규화, 데이터 부족, 상체-only mismatch를 점검.
```

---

## Task 2에서 확인

```text
Adapter를 붙이면 score_z 또는 fusion score가 score_x보다 좋아지는가?
```

결과가 좋으면:

```text
Text prompt alignment가 도움이 됨.
```

결과가 나쁘면:

```text
Adapter가 feature를 뭉개고 있을 수 있음.
loss_preserve weight를 키우거나 adapter를 약하게 만들어야 함.
```

---

# 최종 목표

최종적으로 보고서에 쓸 수 있는 실험 구조는 다음이다.

```text
Task 1:
MotionBERT-only feature distribution 기반 OOD detection

Task 2:
MotionBERT + MLP Adapter + text prompt alignment 기반 OOD detection

Final:
score_x, score_z, score_text를 결합한 normality score로 normal/OOD 판단
```

최종 score 추천:

```python
normality_score = 0.5 * score_x + 0.5 * score_z
```

또는 text까지 포함:

```python
normality_score = 0.4 * score_x + 0.4 * score_z + 0.2 * score_text
```

OOD score:

```python
anomaly_score = 1 - normality_score
```

---

# 한 줄 요약

Task 1에서는 **MotionBERT 원본 feature만으로 정상 분포를 만들고 OOD를 확인**한다.  
Task 2에서는 **MLP Adapter를 추가해 정상 공부 text prompt와 feature를 정렬한 뒤, adapter feature 공간에서 두 번째 정상 분포를 만들고 score를 결합**한다.
