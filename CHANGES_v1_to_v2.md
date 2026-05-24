# v1 → v2 변경사항

## 개요

v2는 v1 MotionBERT baseline 위에 **데이터 파이프라인 버그 수정 + 2초 window 실험 세팅**을 추가한 버전입니다.

---

## 주요 변경사항

### 1. `src/dataset.py` — 버그 수정 3종

#### 1-1. CSV 컬럼명 자동 대응
- v1: `label` 컬럼만 지원 → `annotation_trim_labels.csv`의 `action` 컬럼 읽기 실패
- v2: `label` → `action` 순서로 자동 탐지

```python
label_col = next((c for c in ["label", "action"] if c in df.columns), df.columns[-1])
```

#### 1-2. 숫자 라벨 정규화
- v1: `"3.0"` 같은 float 문자열을 그대로 사용 → `normal_action_ids`의 `"3"`과 불일치
- v2: float 라벨을 정수 문자열로 정규화 (`"3.0"` → `"3"`)

```python
try:
    label = str(int(float(raw)))
except ValueError:
    label = raw
```

#### 1-3. CSV 파일명 자동 탐색
- v1: `{session_name}_labels.csv` 고정 → `recording_trim_labels.csv` 없으면 전체를 normal로 처리
- v2: 해당 파일 없으면 디렉토리 내 `*_labels.csv` 자동 탐색 → `annotation_trim_labels.csv` 인식

#### 1-4. `normal_action_ids` 지원 추가
- v1: `_is_normal(label)` — "normal"/"study"/"studying" 문자열만 인식
- v2: config에 `normal_action_ids: [1,2,3,4,5]` 지정 시 해당 숫자 라벨을 정상으로 처리

---

### 2. `src/compute_stats_x.py` — 시간 순서 70/30 split 추가

- v1: 전체 정상 클립으로 μ, Σ 계산
- v2: config의 `train_ratio`(기본 1.0)에 따라 **시간 순서 기준 앞 N%만** 사용

```
전체 정상 클립 → 시간순 정렬 → 앞 70% → stats (μ, Σ)
                                나머지 30% + OOD → 평가
```

랜덤 split 대신 시간 순서를 쓰는 이유: 인접 클립이 train/val에 동시에 들어가는 것을 방지하고, 실제 사용 시나리오(앞 데이터로 분포 구축 → 이후 데이터 모니터링)에 부합.

---

### 3. `configs/people1_2s.yaml` — 신규 추가

| 항목 | v1 (`people1.yaml`) | v2 (`people1_2s.yaml`) |
|---|---|---|
| window_sec | 5초 | **2초** |
| sampled_fps | 5fps | **10fps** |
| n_frames | 25 | **20** |
| stride_sec | 0 (non-overlap) | 2 (non-overlap) |
| boundary_margin_sec | 0 | **2** |
| normal_action_ids | 없음 | **[1,2,3,4,5]** |
| train_ratio | 없음 | **0.7** |
| device | cuda:1 | **cuda:0** |

---

### 4. `configs/people1.yaml` — device 수정

- v1: `device: "cuda:1"` → 단일 GPU 환경에서 오류
- v2: `device: "cuda:0"`

---

### 5. `src/evaluate_task1.py` / `src/compute_stats_x.py` — yaml 인코딩 수정

- v1: `open(args.config)` — Windows CP949 환경에서 UTF-8 yaml 파일 읽기 실패
- v2: `open(args.config, encoding="utf-8")`

---

## Task 1 베이스라인 실험 결과 (v2 기준)

- **설정**: 2초 window, 10fps, boundary_margin 2초, 시간 순서 70/30 split
- **데이터**: people1 — 정상 226클립, OOD 8클립 (action 6: 스마트폰 보기, 두리번거리기)
- **stats 계산**: 정상 158클립 (앞 70%)
- **평가**: 전체 234클립

| 지표 | 값 |
|---|---|
| AUROC | 0.3966 |
| AUPRC | 0.0309 |

### t-SNE 분석

OOD 클립이 정상 feature 클러스터 내부에 위치함 → MotionBERT (Kinetics pretrain)가 공부 도메인의 미세한 상체 동작 차이를 충분히 포착하지 못함. 이는 예상된 결과로, MLP Adapter + CLIP text alignment (Task 2)의 필요성을 실험적으로 확인.

---

## 다음 단계 (v3 예정)

- [ ] MLP Adapter 학습 (Contrastive loss, 정상 클립 + CLIP 텍스트 프롬프트 정렬)
- [ ] Prompt Score = cosine_sim(f(x), CLIP_text("A student focusing on studying"))
- [ ] Score Fusion: OoD Score × Prompt Score = Study Normality Score
- [ ] 데이터 추가 수집 (10~15명 본수집)
