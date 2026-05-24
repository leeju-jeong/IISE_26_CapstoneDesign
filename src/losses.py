"""
Adapter loss:
  loss_align    = 1 - cos(z, text_avg)
  loss_preserve = 1 - cos(z, normalize(x))
  loss_total    = loss_align + preserve_weight * loss_preserve
"""
import torch
import torch.nn.functional as F


def adapter_training_loss(
    z: torch.Tensor,
    x: torch.Tensor,
    text_avg: torch.Tensor,
    preserve_weight: float = 0.1,
    ood_embeds: torch.Tensor | None = None,
    ood_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    x_norm = F.normalize(x, dim=-1)
    loss_align = (1.0 - (z * text_avg).sum(dim=-1)).mean()
    loss_preserve = (1.0 - (z * x_norm).sum(dim=-1)).mean()
    loss = loss_align + preserve_weight * loss_preserve

    if ood_embeds is not None and ood_embeds.numel() > 0 and ood_weight > 0:
        loss = loss + ood_weight * F.relu(z @ ood_embeds.T).mean()

    metrics = {
        "loss": loss.item(),
        "align": loss_align.item(),
        "preserve": loss_preserve.item(),
    }
    return loss, metrics
