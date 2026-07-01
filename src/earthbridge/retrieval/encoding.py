from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from earthbridge.data.dataset import (
    ManifestImageDataset,
    infer_modality_channels,
    single_item_collate,
)
from earthbridge.data.manifest import load_manifest
from earthbridge.models import BaselineRetriever, EarthBridgeDualHead


def build_model(
    model_type: str,
    modality_channels: dict[str, int],
    embedding_dim: int,
    backbone: str,
    projection_dropout: float = 0.1,
    shared_backbone: bool = False,
) -> torch.nn.Module:
    if model_type == "baseline":
        return BaselineRetriever(
            modality_channels=modality_channels,
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            projection_dropout=projection_dropout,
            shared_backbone=shared_backbone,
        )
    if model_type == "dual_head":
        return EarthBridgeDualHead(
            modality_channels=modality_channels,
            backbone_name=backbone,
            embedding_dim=embedding_dim,
            projection_dropout=projection_dropout,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_checkpoint_if_available(model: torch.nn.Module, checkpoint: str | Path | None) -> None:
    if not checkpoint:
        return

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)


def encode_manifest(
    manifest_path: str | Path,
    root_dir: str | Path = ".",
    image_size: int = 224,
    embedding_dim: int = 256,
    backbone: str = "small_cnn",
    model_type: str = "baseline",
    head: str = "cross",
    checkpoint: str | Path | None = None,
    modality_filter: str | None = None,
    device: str = "cpu",
    projection_dropout: float = 0.1,
    shared_backbone: bool = False,
) -> tuple[list[str], np.ndarray]:
    rows = load_manifest(manifest_path)
    if modality_filter:
        rows = [row for row in rows if row.get("modality") == modality_filter]
    if not rows:
        return [], np.empty((0, embedding_dim), dtype=np.float32)

    modality_channels = infer_modality_channels(rows)

    dataset = ManifestImageDataset(
        manifest_path=manifest_path,
        root_dir=root_dir,
        image_size=image_size,
        modality_channels=modality_channels,
        modality_filter=modality_filter,
    )
    model = build_model(
        model_type,
        modality_channels,
        embedding_dim,
        backbone,
        projection_dropout=projection_dropout,
        shared_backbone=shared_backbone,
    )
    load_checkpoint_if_available(model, checkpoint)
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_item_collate)
    ids: list[str] = []
    descriptors: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            modality = str(batch["modality"])
            sample_id = str(batch["sample_id"])

            if isinstance(model, BaselineRetriever):
                descriptor = model.encode(image, modality)
            elif head == "same":
                descriptor = model.encode_same(image, modality)
            else:
                descriptor = model.encode_cross(image, modality)

            ids.append(sample_id)
            descriptors.append(descriptor.cpu().numpy()[0])

    if not descriptors:
        return ids, np.empty((0, embedding_dim), dtype=np.float32)

    return ids, np.stack(descriptors).astype(np.float32)
