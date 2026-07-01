"""Loss functions for EarthBridge training."""

from earthbridge.losses.contrastive import (
    bidirectional_hard_negative_margin_loss,
    bidirectional_pair_loss,
    multilabel_supervised_contrastive_loss,
    supervised_contrastive_loss,
)

__all__ = [
    "bidirectional_hard_negative_margin_loss",
    "bidirectional_pair_loss",
    "multilabel_supervised_contrastive_loss",
    "supervised_contrastive_loss",
]
