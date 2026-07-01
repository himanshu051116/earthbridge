from __future__ import annotations

import torch
from torch import nn


class SmallConvBackbone(nn.Module):
    """Offline-safe baseline backbone for tests and early CPU experiments."""

    feature_dim = 128

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, self.feature_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, self.feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_backbone(
    name: str = "small_cnn",
    pretrained: bool = False,
    input_channels: int = 3,
) -> tuple[nn.Module, int]:
    normalized = name.lower().strip()

    if normalized == "small_cnn":
        backbone = SmallConvBackbone(input_channels=input_channels)
        return backbone, backbone.feature_dim

    if normalized == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported backbone: {name}")
