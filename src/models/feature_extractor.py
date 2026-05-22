"""
MotionBERT feature extractor wrapper (Baseline용)

Input : (B, T, J_in, 3)  T=64프레임, J_in=11 상체 관절
Output: (B, 512)          T×J mean-pool 후 feature vector

사전 설정:
  1. MotionBERT 저장소 클론:
       git clone https://github.com/Walter0807/MotionBERT
     그리고 PYTHONPATH에 추가하거나, 아래 lib/ 폴더를 src/ 옆에 위치
  2. Pretrained 체크포인트 다운로드:
       https://github.com/Walter0807/MotionBERT → Releases → MB_pretrain.bin
     → checkpoints/MB_pretrain.bin 에 저장

관절 패딩:
  우리 11개 (상체) → COCO-17 앞 11개에 배치, 하체(11~16)는 0으로 패딩
  COCO-17: nose(0) l_eye(1) r_eye(2) l_ear(3) r_ear(4)
           l_sho(5) r_sho(6) l_elb(7) r_elb(8) l_wri(9) r_wri(10)
           l_hip(11) r_hip(12) l_kne(13) r_kne(14) l_ank(15) r_ank(16)
  우리 11개 MediaPipe → COCO 앞 11개에 순서대로 매핑
"""
import torch
import torch.nn as nn

N_COCO = 17  # MotionBERT pretrained joint count


class MotionBERTExtractor(nn.Module):
    def __init__(self, checkpoint_path: str, freeze: bool = True):
        super().__init__()
        try:
            from lib.model.DSTformer import DSTformer
        except ImportError:
            raise ImportError(
                "MotionBERT를 찾을 수 없습니다.\n"
                "1. git clone https://github.com/Walter0807/MotionBERT\n"
                "2. 프로젝트 루트에서: export PYTHONPATH=MotionBERT:$PYTHONPATH\n"
                "   또는 lib/ 폴더를 src/ 옆에 복사"
            )

        # DSTformer 하이퍼파라미터는 공식 pretrain 설정과 동일
        self.backbone = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            maxlen=243,
            num_joints=N_COCO,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # DataParallel로 저장된 경우 'module.' 접두사 제거
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MotionBERT] Missing keys: {len(missing)} — "
                  "체크포인트와 아키텍처 버전이 다를 수 있습니다.")

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J_in, 3)
        반환: (B, 512)
        """
        B, T, J_in, C = x.shape

        # 하체 패딩: (B, T, 17, 3)
        if J_in < N_COCO:
            pad = torch.zeros(B, T, N_COCO - J_in, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)

        # DSTformer: return_rep=True → (B, T, 17, dim_rep=512)
        feat = self.backbone(x, return_rep=True)  # (B, T, 17, 512)

        # 상체 관절만 사용하여 mean-pool → (B, 512)
        feat = feat[:, :, :J_in, :].mean(dim=(1, 2))
        return feat
