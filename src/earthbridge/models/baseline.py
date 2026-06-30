from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from earthbridge.models.adapters import build_adapters
from earthbridge.models.backbone import build_backbone
from earthbridge.models.heads import ProjectionHead


class BaselineRetriever(nn.Module):
    """Single-head shared embedding baseline for same-modal and cross-modal retrieval."""

    def __init__(
        self,
        modality_channels: dict[str, int],
        backbone_name: str = "small_cnn",
        embedding_dim: int = 256,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        if not modality_channels:
            raise ValueError("modality_channels cannot be empty")

        self.embedding_dim = embedding_dim
        self.adapters = build_adapters(modality_channels)
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained_backbone)
        self.projection = ProjectionHead(feature_dim, embedding_dim)

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.adapters:
            raise ValueError(f"Unknown modality: {modality}")

        adapted = self.adapters[modality](x)
        features = self.backbone(adapted)
        descriptor = self.projection(features)
        return F.normalize(descriptor, dim=-1)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.encode(x, modality)

