from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from earthbridge.data.dataset import infer_modality_channels
from earthbridge.data.manifest import load_manifest
from earthbridge.data.pairs import PairedImageDataset, paired_collate
from earthbridge.losses import (
    bidirectional_hard_negative_margin_loss,
    bidirectional_pair_loss,
    multilabel_supervised_contrastive_loss,
)
from earthbridge.models import BaselineRetriever
from earthbridge.training.checkpointing import save_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: str
    root_dir: str = "."
    left_modality: str = "optical_rgb"
    right_modality: str = "sar"
    validation_manifest_path: str = ""
    image_size: int = 224
    embedding_dim: int = 256
    backbone: str = "small_cnn"
    projection_dropout: float = 0.0
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    semantic_loss_weight: float = 0.1
    hard_negative_loss_weight: float = 0.2
    hard_negative_margin: float = 0.2
    seed: int | None = 42
    device: str = "cpu"
    output_checkpoint: str = "artifacts/checkpoints/baseline_pair.pt"


def seed_everything(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def assert_aligned_pair_ids(batch: dict[str, object]) -> None:
    if "left_pair_ids" not in batch or "right_pair_ids" not in batch:
        raise ValueError("Training batch is missing left/right pair IDs")

    left_pair_ids = list(batch.get("left_pair_ids", []))
    right_pair_ids = list(batch.get("right_pair_ids", []))
    if left_pair_ids != right_pair_ids:
        raise ValueError("Training batch has misaligned left/right pair IDs")


def encode_pair_dataset(
    model: BaselineRetriever,
    dataset: PairedImageDataset,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=paired_collate,
    )
    left_embeddings: list[torch.Tensor] = []
    right_embeddings: list[torch.Tensor] = []
    pair_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            assert_aligned_pair_ids(batch)
            left_image = batch["left_image"].to(device)
            right_image = batch["right_image"].to(device)
            left_embeddings.append(
                model.encode(left_image, str(batch["left_modality"])).detach().cpu()
            )
            right_embeddings.append(
                model.encode(right_image, str(batch["right_modality"])).detach().cpu()
            )
            pair_ids.extend(str(pair_id) for pair_id in batch["pair_ids"])

    if not left_embeddings:
        empty = torch.empty((0, model.embedding_dim), dtype=torch.float32)
        return empty, empty, []

    return torch.cat(left_embeddings), torch.cat(right_embeddings), pair_ids


def ranks_for_exact_pairs(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    chunk_size: int = 512,
) -> list[int]:
    if query_embeddings.shape != gallery_embeddings.shape:
        raise ValueError("query and gallery embeddings must have matching shapes")

    gallery = torch.nn.functional.normalize(gallery_embeddings.float(), dim=-1)
    queries = torch.nn.functional.normalize(query_embeddings.float(), dim=-1)
    ranks: list[int] = []

    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        scores = queries[start:end] @ gallery.T
        target_positions = torch.arange(start, end)
        target_scores = scores[
            torch.arange(end - start),
            target_positions,
        ].unsqueeze(1)
        ranks.extend((scores > target_scores).sum(dim=1).add(1).tolist())

    return [int(rank) for rank in ranks]


def metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "median_rank": 0.0,
        }

    total = len(ranks)
    return {
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / total,
        "median_rank": float(median(ranks)),
    }


def exact_pair_retrieval_metrics(
    model: BaselineRetriever,
    dataset: PairedImageDataset,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    left_embeddings, right_embeddings, pair_ids = encode_pair_dataset(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    left_to_right_ranks = ranks_for_exact_pairs(left_embeddings, right_embeddings)
    right_to_left_ranks = ranks_for_exact_pairs(right_embeddings, left_embeddings)

    left_to_right = metrics_from_ranks(left_to_right_ranks)
    right_to_left = metrics_from_ranks(right_to_left_ranks)
    mean_recall_at_10 = 0.5 * (
        left_to_right["recall_at_10"] + right_to_left["recall_at_10"]
    )
    mean_recall_at_1 = 0.5 * (
        left_to_right["recall_at_1"] + right_to_left["recall_at_1"]
    )

    return {
        "pair_count": len(pair_ids),
        "left_modality": dataset.left_modality,
        "right_modality": dataset.right_modality,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        f"{dataset.left_modality}_to_{dataset.right_modality}": left_to_right,
        f"{dataset.right_modality}_to_{dataset.left_modality}": right_to_left,
        "mean_recall_at_1": mean_recall_at_1,
        "mean_recall_at_10": mean_recall_at_10,
    }


def build_pair_dataset(
    manifest_path: str,
    config: TrainingConfig,
    modality_channels: dict[str, int],
) -> PairedImageDataset:
    return PairedImageDataset(
        manifest_path=manifest_path,
        left_modality=config.left_modality,
        right_modality=config.right_modality,
        root_dir=config.root_dir,
        image_size=config.image_size,
        modality_channels=modality_channels,
    )


def select_validation_dataset(
    config: TrainingConfig,
    train_dataset: PairedImageDataset,
    modality_channels: dict[str, int],
) -> PairedImageDataset:
    if config.validation_manifest_path:
        path = Path(config.validation_manifest_path)
        if path.exists():
            validation_dataset = build_pair_dataset(
                str(path),
                config,
                modality_channels,
            )
            if len(validation_dataset) > 0:
                return validation_dataset
    return train_dataset


def train_paired_baseline(config: TrainingConfig) -> dict[str, object]:
    data_generator = seed_everything(config.seed)
    rows = load_manifest(config.manifest_path)
    modality_channels = infer_modality_channels(rows)
    train_dataset = build_pair_dataset(config.manifest_path, config, modality_channels)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No aligned pairs found for {config.left_modality}->{config.right_modality}"
        )

    validation_dataset = select_validation_dataset(config, train_dataset, modality_channels)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=paired_collate,
        generator=data_generator,
    )
    model = BaselineRetriever(
        modality_channels=modality_channels,
        backbone_name=config.backbone,
        embedding_dim=config.embedding_dim,
        projection_dropout=config.projection_dropout,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_recall_at_10 = -1.0
    best_recall_at_1 = -1.0
    best_validation: dict[str, Any] | None = None
    checkpoint_path = Path(config.output_checkpoint)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in loader:
            assert_aligned_pair_ids(batch)
            left_image = batch["left_image"].to(config.device)
            right_image = batch["right_image"].to(config.device)

            left_embeddings = model.encode(left_image, str(batch["left_modality"]))
            right_embeddings = model.encode(right_image, str(batch["right_modality"]))
            loss = bidirectional_pair_loss(
                left_embeddings,
                right_embeddings,
                temperature=config.temperature,
            )
            if config.hard_negative_loss_weight > 0:
                hard_negative_loss = bidirectional_hard_negative_margin_loss(
                    left_embeddings,
                    right_embeddings,
                    margin=config.hard_negative_margin,
                )
                loss = loss + config.hard_negative_loss_weight * hard_negative_loss
            if config.semantic_loss_weight > 0:
                semantic_loss = multilabel_supervised_contrastive_loss(
                    torch.cat([left_embeddings, right_embeddings], dim=0),
                    [*batch["labels"], *batch["labels"]],
                    temperature=config.temperature,
                )
                loss = loss + config.semantic_loss_weight * semantic_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        average_loss = total_loss / max(batches, 1)
        validation_metrics = exact_pair_retrieval_metrics(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=config.device,
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_pair_loss": average_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        recall_at_10 = float(validation_metrics["mean_recall_at_10"])
        recall_at_1 = float(validation_metrics["mean_recall_at_1"])
        improved_recall_at_10 = recall_at_10 > best_recall_at_10
        tied_with_better_recall_at_1 = (
            recall_at_10 == best_recall_at_10 and recall_at_1 > best_recall_at_1
        )
        if improved_recall_at_10 or tied_with_better_recall_at_1:
            best_recall_at_10 = recall_at_10
            best_recall_at_1 = recall_at_1
            best_epoch = epoch
            best_validation = validation_metrics
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                metadata={
                    "config": asdict(config),
                    "history": history,
                    "modality_channels": modality_channels,
                    "pair_count": len(train_dataset),
                    "validation_pair_count": len(validation_dataset),
                    "best_epoch": best_epoch,
                    "best_mean_recall_at_10": best_recall_at_10,
                    "best_mean_recall_at_1": best_recall_at_1,
                    "best_validation": validation_metrics,
                },
            )

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a checkpoint")

    return {
        "checkpoint": config.output_checkpoint,
        "pair_count": len(train_dataset),
        "validation_pair_count": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_mean_recall_at_10": best_recall_at_10,
        "best_mean_recall_at_1": best_recall_at_1,
        "best_validation": best_validation or {},
        "history": history,
    }
