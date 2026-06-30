from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from earthbridge.data.dataset import infer_modality_channels
from earthbridge.data.manifest import load_manifest
from earthbridge.data.pairs import PairedImageDataset, paired_collate
from earthbridge.losses import bidirectional_pair_loss
from earthbridge.models import BaselineRetriever
from earthbridge.training.checkpointing import save_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: str
    root_dir: str = "."
    left_modality: str = "optical_rgb"
    right_modality: str = "sar"
    image_size: int = 224
    embedding_dim: int = 256
    backbone: str = "small_cnn"
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    device: str = "cpu"
    output_checkpoint: str = "artifacts/checkpoints/baseline_pair.pt"


def train_paired_baseline(config: TrainingConfig) -> dict[str, object]:
    rows = load_manifest(config.manifest_path)
    modality_channels = infer_modality_channels(rows)
    dataset = PairedImageDataset(
        manifest_path=config.manifest_path,
        left_modality=config.left_modality,
        right_modality=config.right_modality,
        root_dir=config.root_dir,
        image_size=config.image_size,
        modality_channels=modality_channels,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No aligned pairs found for {config.left_modality}->{config.right_modality}"
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=paired_collate,
    )
    model = BaselineRetriever(
        modality_channels=modality_channels,
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in loader:
            left_image = batch["left_image"].to(config.device)
            right_image = batch["right_image"].to(config.device)

            left_embeddings = model.encode(left_image, str(batch["left_modality"]))
            right_embeddings = model.encode(right_image, str(batch["right_modality"]))
            loss = bidirectional_pair_loss(
                left_embeddings,
                right_embeddings,
                temperature=config.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        average_loss = total_loss / max(batches, 1)
        history.append({"epoch": float(epoch), "train_pair_loss": average_loss})

    save_checkpoint(
        Path(config.output_checkpoint),
        model,
        optimizer,
        metadata={
            "config": asdict(config),
            "history": history,
            "modality_channels": modality_channels,
            "pair_count": len(dataset),
        },
    )

    return {
        "checkpoint": config.output_checkpoint,
        "pair_count": len(dataset),
        "history": history,
    }

