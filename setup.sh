#!/bin/bash
# ────────────────────────────────────────────────────────────────
# Study Normality Score — 환경 세팅 스크립트
# 사용법:
#   bash setup.sh                        # 기본: /home/storage/$USER/envs/study
#   bash setup.sh /path/to/custom/envs  # 경로 직접 지정
# ────────────────────────────────────────────────────────────────

set -e  # 에러 발생 시 즉시 중단

ENV_BASE="${1:-/home/storage/$USER/envs}"
ENV_PATH="$ENV_BASE/study"

echo "============================================"
echo " Study Normality Score — Environment Setup"
echo " ENV: $ENV_PATH"
echo "============================================"

# ── 1. Conda 확인 ──
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda가 설치되어 있지 않습니다."
    exit 1
fi

# ── 2. 환경 생성 ──
if conda env list | grep -q "$ENV_PATH"; then
    echo "[SKIP] 이미 환경이 존재합니다: $ENV_PATH"
else
    echo "[INFO] conda 환경 생성 중..."
    mkdir -p "$ENV_BASE"
    conda create -p "$ENV_PATH" python=3.10 -y
fi

# ── 3. PyTorch + 패키지 설치 ──
echo "[INFO] PyTorch (CUDA 12.1) 설치 중..."
conda run -p "$ENV_PATH" pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 -q

echo "[INFO] 나머지 패키지 설치 중..."
conda run -p "$ENV_PATH" pip install \
    mediapipe==0.10.35 \
    opencv-python \
    open_clip_torch \
    numpy \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    pyyaml \
    -q

# ── 4. 설치 확인 ──
echo ""
echo "[CHECK] 설치 확인..."
conda run -p "$ENV_PATH" python -c "
import torch, mediapipe, cv2, open_clip, numpy
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'  MediaPipe: {mediapipe.__version__}')
print(f'  OpenCV  : {cv2.__version__}')
print(f'  NumPy   : {numpy.__version__}')
print(f'  OpenCLIP: OK')
"

echo ""
echo "============================================"
echo " 완료!"
echo " 활성화: conda activate $ENV_PATH"
echo "============================================"
