# Study Normality Score — Project Context

## 한 줄 정의

> 자기주도 학습 장면에서, 정상적인 공부 상태를 텍스트 프롬프트로 정의하고,
> 상체 스켈레톤과 CLIP 공간의 정렬을 통해 **Study Normality Score**를 산출하는 privacy-aware 시스템

---

## 참고 논문

**Sato et al., CVPR 2023**
"Prompt-Guided Zero-Shot Anomaly Action Recognition Using Pretrained Deep Skeleton Features"

### Sato 논문의 핵심 구조

1. 포즈 추출기 → 관절마다 **7D 벡터** 생성
   ```
   v = [x, y, time_index, confidence, joint_index, centroid_x, centroid_y]
   모든 값 0~1 정규화
   ```

2. **Feature Extractor F** (PointNet 기반, Kinetics-400 pretrained, 완전 frozen)
   ```
   V = {v1, ..., vJ} → shared MLP → Residual MLP blocks → MaxPool → x ∈ R^S
   ```

3. **Training** = 정상 데이터의 μ, Σ 계산만. DNN 업데이트 없음.

4. **OoD Score** = Mahalanobis(x, μ, Σ)

5. **Prompt Score** = cosine_sim(f(x), CLIP_text("violence"))
   - f: x와 텍스트 임베딩 차원을 맞추는 MLP (pretrained+frozen)

6. **Anomaly Score** = OoD Score × Prompt Score

### 우리 프로젝트와의 차이점

| 항목 | Sato | 우리 |
|---|---|---|
| 텍스트 정의 대상 | abnormal ("violence") | **normal** ("A student focusing on studying") |
| 출력 | Anomaly Score | **Study Normality Score** |
| 도메인 | 보안/폭력 탐지 | 자기주도학습 집중도 |
| Feature Extractor | 완전 frozen | **마지막 레이어 fine-tune** |
| 입력 | 전신 스켈레톤 | **상체 11개 관절만** |

---

## 확정된 파이프라인

```
[Video Input]
    ↓ 720p, 30fps, 카메라 거리 0.8~1.2m
[MediaPipe]
    ↓ 상체 11개 관절 추출
    머리 5개: 코, 눈(좌우), 귀(좌우)
    팔 6개: 어깨(좌우), 팔꿈치(좌우), 손목(좌우)
[7D 벡터 변환]
    ↓ v = [x, y, time, confidence, joint_index, centroid_x, centroid_y], 0~1 정규화
[Feature Extractor] ← PYSKL pretrained (ST-GCN or PoseC3D)
    ↓ 마지막 몇 레이어만 fine-tune 허용 (나머지 frozen)
    x ∈ R^S
[MLP Adapter] ← 학습 대상
    ↓ Contrastive loss로 학습 (정상 클립 + 텍스트 프롬프트 쌍)
    f(x) ∈ R^512 (CLIP text embedding 차원)
        ↙                    ↘
[OoD Score]              [Prompt Score]
Mahalanobis(x, μ, Σ)    cosine_sim(f(x), CLIP_text("A student focusing on studying"))
μ, Σ는 정상 데이터로 계산
        ↘                    ↙
    [Score Fusion] = OoD × Prompt
    + Temporal Smoothing (10초 window, stride 5초)
        ↓
[Study Normality Score] ∈ [0, 1]
```

---

## 핵심 설계 결정 및 이유

### 왜 MediaPipe인가
- Kinetics-400 pretrained PointNet은 전신 동작 기반 → 상체만 넣으면 품질 보장 없음
- NTU RGB+D 재학습은 캡스톤 스케일 초과 (114K 영상, GPU 수십 시간)
- MediaPipe = Google이 수억 장으로 학습한 상체 특화 추출기, 실시간, GPU 불필요

### 왜 완전 frozen이 아닌가
- Kinetics pretrain = 전신 격렬한 동작 위주
- 공부 도메인 = 미세한 상체 움직임 (고개 기울기, 어깨 방향, 손 위치)
- 마지막 레이어만 열면 도메인 갭을 메우면서 학습 데이터 부담 최소화

### 왜 Contrastive loss인가
- Sato의 핵심 기여: skeleton feature와 CLIP text embedding을 같은 공간으로 정렬
- 정렬이 없으면 Prompt Score가 의미 없어짐
- 정상 공부 클립 + 텍스트 프롬프트 쌍으로 학습 가능

### 왜 Normality Score인가 (Anomaly Score가 아닌)
- 공부 도메인에서 "이상"의 경계가 모호 (필기, 물 마시기, 잠깐 생각하기 = 정상)
- "정상 상태를 정의"하는 게 "이상 상태를 정의"하는 것보다 쉽고 정확
- 서비스 관점에서도 "집중도 점수"가 "이상행동 탐지"보다 자연스러움

### 왜 상체 11개 관절인가
- 앉아서 공부 → 하체 정보 의미 없음
- 집중/비집중의 핵심 신호: 고개 기울기, 어깨 방향, 팔/손 위치
- Privacy-aware: 하체, 배경, 얼굴 외형 제거 → 식별 가능성 감소

---

## 데이터셋 전략

### 수집 구조
```
[훈련용] 정상 공부 영상만 → μ, Σ 계산 + Adapter 학습
[평가용] 정상 + OOD 영상 → 라벨 있어야 AUROC 계산 가능
```

### 세션 구조 (참가자당 ~30분)
- **Session A** (10분): 정상 공부, 자유롭게 — label: `normal`
- **Session B** (5분): 정상 + 자연스러운 흔들림 (물 마시기 등) — label: `normal`
- **Session C** (10분): OOD 행동 삽입, 타임스탬프 기록 필수 — label: `normal` / `OOD_XX`
- **Session D** (5분): 환경 변화 (조명, 각도) — label: `normal`

### OOD 행동 5종
| 코드 | 행동 | 기준 |
|---|---|---|
| OOD_01 | 스마트폰 보기 | 30초 이상 |
| OOD_02 | 엎드리기 | 30초 이상 |
| OOD_03 | 자리 비움 | 상체 사라짐 |
| OOD_04 | 반복 두리번거리기 | 30초 이상 |
| OOD_05 | 고개 돌린 상태 지속 | 30초 이상 |

### 촬영 조건
- 장비: 노트북 웹캠 or 스마트폰 거치
- 해상도: 720p / 30fps
- 카메라: 눈높이 ±10cm, 정면 ±15°, 거리 0.8~1.2m
- 상체 전체(머리~손목) 프레임 내 필수

### 클립 단위
- window: 10초 (300 프레임 @ 30fps)
- stride: 5초
- 정상/OOD 경계 ±5초 클립 제외

### Split 방식
- **Subject-independent split** 필수 (같은 사람이 train/test 동시에 들어가면 과대평가)
- Pilot: 2~3명 → 본수집: 10~15명
- train: 8~10명 / val: 2명 / test: 2~3명

### 파일 구조
```
data/
  pilot/
    subject_01/
      session_A.mp4
      session_B.mp4
      session_C.mp4
      session_D.mp4
      labels.csv      # start_sec, end_sec, label
      meta.json       # 장비, 환경 조건
```

---

## 텍스트 프롬프트 후보 (정상 공부 정의)

```python
NORMAL_PROMPTS = [
    "A student is sitting upright and focusing on studying.",
    "A learner is looking at the screen or desk material with sustained attention.",
    "A student is engaged in a normal self-study session.",
    "A person is reading or writing at a desk in a focused manner.",
]
```

---

## 확장 가능한 도메인

| 도메인 | 이유 | 공개 데이터셋 |
|---|---|---|
| 온라인 시험 감독 | 세팅 동일, "정상" 더 엄격 | 없음 (직접 수집) |
| 운전자 이상행동 | 상체+앉은자세+정상명확 | **DAD Dataset** ← 논문 cross-domain 검증용 |
| 원격근무 집중도 | 프롬프트만 바꾸면 적용 가능 | 없음 |

DAD Dataset으로 cross-domain 실험 시 논문 contribution 강화 가능:
> "텍스트 프롬프트만 바꿔서 운전 도메인에도 적용 가능하다"

---

## 지금 당장 해야 할 일

### 1순위 (지금)
- [ ] **PYSKL 호환성 확인**: ST-GCN or PoseC3D에 상체 11개 관절 입력 → feature 추출 테스트
- [ ] **Pilot 수집**: 팀원 2~3명, 세션 구조대로, MediaPipe 관절 품질 확인

### 2순위 (다음)
- [ ] **Adapter 학습 코드 구현**: Contrastive loss, 정상 클립-텍스트 쌍 구성
- [ ] **본수집 (10~15명)**: 프로토콜 확정 후 진행

### 3순위 (나중)
- [ ] **Privacy 비교 실험**: RGB / Skeleton / Depth / noisy RGB 성능 비교
- [ ] **DAD Dataset cross-domain 실험**

---

## 평가 지표

- **AUROC**: 정상 vs OOD 구분 성능
- **t-SNE 시각화**: 정상/OOD feature 분포 분리 확인 (Pilot 후 바로 확인 가능)
- **Privacy 실험**: 모달리티별 AUROC 비교 (RGB baseline 대비 성능 유지율)

---

## 관련 링크 및 자료
- Sato et al. CVPR 2023 논문: 프로젝트 폴더 내 PDF
- PYSKL: https://github.com/kennymckormick/pyskl
- DAD Dataset: https://github.com/okankop/Driver-Anomaly-Detection
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker