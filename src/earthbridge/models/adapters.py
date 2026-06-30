from __future__ import annotations

import torch
from torch import nn


class InputAdapter(nn.Module):
    """Map modality-specific channel counts into a common input space."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.adapter = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("InputAdapter expects a BCHW tensor")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels for adapter, got {x.shape[1]}"
            )
        return self.adapter(x)


def build_adapters(modality_channels: dict[str, int]) -> nn.ModuleDict:
    return nn.ModuleDict(
        {modality: InputAdapter(channels) for modality, channels in modality_channels.items()}
    )
