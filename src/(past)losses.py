"""
Contrastive loss: skeleton features ↔ CLIP text embeddings 정렬

학습 데이터가 전부 normal이므로:
  - 모든 clip feature z_i를 정상 텍스트 임베딩 t_avg 쪽으로 당김
  - InfoNCE with in-batch negatives (z_i vs z_j)

구체적으로:
  sim_matrix[i, k] = cosine_sim(z_i, t_k) / temperature
  → z_i는 모든 프롬프트 t_k에 가까워야 하므로 soft target = uniform over P prompts
  동시에 clip-to-clip: sim(z_i, z_j) / temp → i=j가 diagonal이 되도록
"""
import torch
import torch.nn.functional as F


def contrastive_loss(z: torch.Tensor, text_embeds: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    z           : (B, 512) skeleton features, L2 normalized
    text_embeds : (P, 512) CLIP text embeddings, L2 normalized
    Returns     : scalar loss
    """
    B = z.size(0)
    P = text_embeds.size(0)

    # --- clip-to-text alignment ---
    # z @ t.T → (B, P)  각 clip이 P개 프롬프트 중 하나를 target으로
    sim_ct = z @ text_embeds.T / temperature  # (B, P)

    # 모든 프롬프트가 동등하게 정상을 설명 → uniform soft target
    target_ct = torch.full((B, P), 1.0 / P, device=z.device)
    loss_ct = -(target_ct * F.log_softmax(sim_ct, dim=1)).sum(dim=1).mean()

    # --- clip-to-clip instance discrimination ---
    # 같은 배치 내 다른 clip은 서로 다른 instance → diagonal이 positive
    sim_cc = z @ z.T / temperature  # (B, B)
    # diagonal을 positive로, 나머지를 negative로
    labels_cc = torch.arange(B, device=z.device)
    loss_cc = F.cross_entropy(sim_cc, labels_cc)

    return 0.5 * loss_ct + 0.5 * loss_cc
