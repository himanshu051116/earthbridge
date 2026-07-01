from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from earthbridge.models.adapters import build_adapters
from earthbridge.models.backbone import build_backbone
from earthbridge.models.heads import ProjectionHead

SMALL_CNN_ADAPTER_CHANNELS = 16
RGB_ADAPTER_CHANNELS = 3


class EarthBridgeDualHead(nn.Module):
    """Shared encoder with one cross-modal head and modality-specific same-modal heads."""

    def __init__(
        self,
        modality_channels: dict[str, int],
        backbone_name: str = "small_cnn",
        embedding_dim: int = 256,
        pretrained_backbone: bool = False,
        projection_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not modality_channels:
            raise ValueError("modality_channels cannot be empty")

        self.embedding_dim = embedding_dim
        adapter_channels = (
            SMALL_CNN_ADAPTER_CHANNELS
            if backbone_name.lower().strip() == "small_cnn"
            else RGB_ADAPTER_CHANNELS
        )
        self.adapters = build_adapters(modality_channels, output_channels=adapter_channels)
        self.shared_encoder, feature_dim = build_backbone(
            backbone_name,
            pretrained=pretrained_backbone,
            input_channels=adapter_channels,
        )
        self.cross_head = ProjectionHead(feature_dim, embedding_dim, dropout=projection_dropout)
        self.same_heads = nn.ModuleDict(
            {
                modality: ProjectionHead(feature_dim, embedding_dim, dropout=projection_dropout)
                for modality in modality_channels
            }
        )

    def encode_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.adapters:
            raise ValueError(f"Unknown modality: {modality}")
        return self.shared_encoder(self.adapters[modality](x))

    def encode_cross(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        descriptor = self.cross_head(self.encode_features(x, modality))
        return F.normalize(descriptor, dim=-1)

    def encode_same(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.same_heads:
            raise ValueError(f"Unknown modality: {modality}")
        descriptor = self.same_heads[modality](self.encode_features(x, modality))
        return F.normalize(descriptor, dim=-1)
