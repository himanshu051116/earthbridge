from __future__ import annotations

import torch
import torch.nn.functional as F


def bidirectional_pair_loss(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style pair loss for aligned batches.

    The diagonal is treated as the exact cross-modal pair, so the two input
    batches must be ordered consistently.
    """

    if left_embeddings.shape != right_embeddings.shape:
        raise ValueError("left_embeddings and right_embeddings must have identical shapes")
    if left_embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    left = F.normalize(left_embeddings, dim=-1)
    right = F.normalize(right_embeddings, dim=-1)
    logits = (left @ right.T) / temperature
    labels = torch.arange(left.shape[0], device=left.device)

    left_to_right = F.cross_entropy(logits, labels)
    right_to_left = F.cross_entropy(logits.T, labels)
    return 0.5 * (left_to_right + right_to_left)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss for single-label batches."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be a 1D tensor matching the batch size")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    embeddings = F.normalize(embeddings, dim=-1)
    logits = (embeddings @ embeddings.T) / temperature
    batch_size = embeddings.shape[0]
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

    labels = labels.reshape(-1, 1)
    positive_mask = labels.eq(labels.T) & ~eye
    valid_anchor_mask = positive_mask.any(dim=1)

    if not valid_anchor_mask.any():
        return embeddings.sum() * 0.0

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * ~eye
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    mean_log_prob_positive = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(
        dim=1
    ).clamp_min(1)
    return -mean_log_prob_positive[valid_anchor_mask].mean()

