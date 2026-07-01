from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from earthbridge.models.adapters import InputAdapter
from earthbridge.models.backbone import build_backbone
from earthbridge.models.heads import ProjectionHead

SMALL_CNN_ADAPTER_CHANNELS = 16
RGB_ADAPTER_CHANNELS = 3


class SensorEncoder(nn.Module):
    """Modality-specific stem plus CNN features and global band statistics."""

    def __init__(
        self,
        input_channels: int,
        adapter_channels: int,
        backbone_name: str,
        pretrained_backbone: bool,
    ) -> None:
        super().__init__()
        self.adapter = InputAdapter(input_channels, output_channels=adapter_channels)
        self.backbone, backbone_dim = build_backbone(
            backbone_name,
            pretrained=pretrained_backbone,
            input_channels=adapter_channels,
        )
        self.sketch_size = 8
        sketch_pixels = self.sketch_size * self.sketch_size
        self.feature_dim = (
            backbone_dim
            + adapter_channels * 2
            + input_channels * 2
            + adapter_channels * sketch_pixels
            + input_channels * sketch_pixels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_means = x.mean(dim=(2, 3))
        raw_stds = x.std(dim=(2, 3), unbiased=False)
        raw_sketch = F.adaptive_avg_pool2d(
            x,
            output_size=(self.sketch_size, self.sketch_size),
        ).flatten(1)
        adapted = self.adapter(x)
        features = self.backbone(adapted)
        adapted_means = adapted.mean(dim=(2, 3))
        adapted_stds = adapted.std(dim=(2, 3), unbiased=False)
        adapted_sketch = F.adaptive_avg_pool2d(
            adapted,
            output_size=(self.sketch_size, self.sketch_size),
        ).flatten(1)
        return torch.cat(
            [
                features,
                adapted_means,
                adapted_stds,
                raw_means,
                raw_stds,
                adapted_sketch,
                raw_sketch,
            ],
            dim=1,
        )


class BaselineRetriever(nn.Module):
    """Single-head shared embedding baseline for same-modal and cross-modal retrieval."""

    def __init__(
        self,
        modality_channels: dict[str, int],
        backbone_name: str = "small_cnn",
        embedding_dim: int = 256,
        pretrained_backbone: bool = False,
        projection_dropout: float = 0.1,
        shared_backbone: bool = False,
    ) -> None:
        super().__init__()
        if not modality_channels:
            raise ValueError("modality_channels cannot be empty")

        self.embedding_dim = embedding_dim
        self.shared_backbone = shared_backbone
        adapter_channels = (
            SMALL_CNN_ADAPTER_CHANNELS
            if backbone_name.lower().strip() == "small_cnn"
            else RGB_ADAPTER_CHANNELS
        )
        if shared_backbone:
            self.adapters = nn.ModuleDict(
                {
                    modality: InputAdapter(channels, output_channels=adapter_channels)
                    for modality, channels in modality_channels.items()
                }
            )
            self.backbone, feature_dim = build_backbone(
                backbone_name,
                pretrained=pretrained_backbone,
                input_channels=adapter_channels,
            )
            self.projection = ProjectionHead(
                feature_dim,
                embedding_dim,
                dropout=projection_dropout,
            )
            self.encoders = nn.ModuleDict()
            self.projections = nn.ModuleDict()
        else:
            self.adapters = nn.ModuleDict()
            self.backbone = nn.Identity()
            self.projection = nn.Identity()
            self.encoders = nn.ModuleDict()
            self.projections = nn.ModuleDict()
            for modality, channels in modality_channels.items():
                encoder = SensorEncoder(
                    input_channels=channels,
                    adapter_channels=adapter_channels,
                    backbone_name=backbone_name,
                    pretrained_backbone=pretrained_backbone,
                )
                self.encoders[modality] = encoder
                self.projections[modality] = ProjectionHead(
                    encoder.feature_dim,
                    embedding_dim,
                    dropout=projection_dropout,
                )

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if self.shared_backbone:
            if modality not in self.adapters:
                raise ValueError(f"Unknown modality: {modality}")

            adapted = self.adapters[modality](x)
            features = self.backbone(adapted)
            descriptor = self.projection(features)
            return F.normalize(descriptor, dim=-1)

        if modality not in self.encoders:
            raise ValueError(f"Unknown modality: {modality}")

        features = self.encoders[modality](x)
        descriptor = self.projections[modality](features)
        return F.normalize(descriptor, dim=-1)

    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        return self.encode(x, modality)
