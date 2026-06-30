"""Loss functions for EarthBridge training."""

from earthbridge.losses.contrastive import bidirectional_pair_loss, supervised_contrastive_loss

__all__ = ["bidirectional_pair_loss", "supervised_contrastive_loss"]

