from __future__ import annotations

import random
from collections.abc import Iterable
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
    shared_backbone: bool = False
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    learnable_temperature: bool = False
    semantic_loss_weight: float = 0.0
    hard_negative_loss_weight: float = 0.0
    hard_negative_margin: float = 0.2
    stop_recall_at_1: float = 0.0
    stop_recall_at_10: float = 0.0
    require_validation_pair_alignment: bool = False
    diagnostic_sample_count: int = 0
    collapse_similarity_threshold: float = 0.999
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


def dataset_pair_ids(dataset: PairedImageDataset) -> list[str]:
    return [left_row.get("pair_id", "") for left_row, _right_row in dataset.pairs]


def pair_order_summary(
    train_dataset: PairedImageDataset,
    validation_dataset: PairedImageDataset,
) -> dict[str, object]:
    train_pair_ids = dataset_pair_ids(train_dataset)
    validation_pair_ids = dataset_pair_ids(validation_dataset)
    first_mismatch = ""
    for index, (train_pair_id, validation_pair_id) in enumerate(
        zip(train_pair_ids, validation_pair_ids, strict=False)
    ):
        if train_pair_id != validation_pair_id:
            first_mismatch = f"{index}: {train_pair_id} != {validation_pair_id}"
            break

    return {
        "train_pair_count": len(train_pair_ids),
        "validation_pair_count": len(validation_pair_ids),
        "train_unique_pair_count": len(set(train_pair_ids)),
        "validation_unique_pair_count": len(set(validation_pair_ids)),
        "intersection_pair_count": len(set(train_pair_ids) & set(validation_pair_ids)),
        "same_order": train_pair_ids == validation_pair_ids,
        "first_mismatch": first_mismatch,
        "first_train_pair_ids": train_pair_ids[:5],
        "first_validation_pair_ids": validation_pair_ids[:5],
    }


def assert_validation_pair_alignment(summary: dict[str, object]) -> None:
    if not summary["same_order"]:
        raise ValueError(
            "Training and validation pair IDs are not identical and aligned: "
            f"{summary['first_mismatch']}"
        )
    if summary["train_unique_pair_count"] != summary["train_pair_count"]:
        raise ValueError("Training overfit set contains duplicate pair IDs")


def current_temperature(
    config: TrainingConfig,
    logit_scale_parameter: torch.nn.Parameter | None,
    device: str,
) -> torch.Tensor:
    if logit_scale_parameter is None:
        return torch.tensor(config.temperature, dtype=torch.float32, device=device)

    logit_scale = logit_scale_parameter.exp().clamp(max=100.0)
    return logit_scale.reciprocal()


def temperature_diagnostics(temperature: torch.Tensor) -> dict[str, float]:
    value = float(temperature.detach().cpu())
    return {
        "effective_temperature": value,
        "logit_scale": 1.0 / value if value > 0 else 0.0,
    }


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().data.norm(2).cpu())
        total += norm * norm
    return total**0.5


def off_diagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] <= 1:
        return torch.empty(0, dtype=matrix.dtype, device=matrix.device)
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def near_identical_percentage(
    embeddings: torch.Tensor,
    threshold: float,
) -> float:
    if embeddings.shape[0] <= 1:
        return 0.0

    normalized = torch.nn.functional.normalize(embeddings.detach().float(), dim=-1)
    similarities = normalized @ normalized.T
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    near_identical = (similarities.masked_fill(~mask, -1.0) >= threshold).any(dim=1)
    return float(near_identical.float().mean().cpu())


def embedding_diagnostics(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    collapse_similarity_threshold: float,
) -> dict[str, float]:
    left = torch.nn.functional.normalize(left_embeddings.detach().float(), dim=-1)
    right = torch.nn.functional.normalize(right_embeddings.detach().float(), dim=-1)
    cross_similarities = left @ right.T
    positive_similarities = cross_similarities.diag()
    negative_similarities = off_diagonal_values(cross_similarities)

    return {
        "positive_pair_cosine_mean": float(positive_similarities.mean().cpu()),
        "negative_pair_cosine_mean": float(negative_similarities.mean().cpu())
        if negative_similarities.numel()
        else 0.0,
        "left_embedding_std_mean": float(left_embeddings.detach().std(dim=0).mean().cpu()),
        "right_embedding_std_mean": float(right_embeddings.detach().std(dim=0).mean().cpu()),
        "left_nearly_identical_embedding_percentage": near_identical_percentage(
            left_embeddings,
            collapse_similarity_threshold,
        ),
        "right_nearly_identical_embedding_percentage": near_identical_percentage(
            right_embeddings,
            collapse_similarity_threshold,
        ),
    }


def batch_band_statistics(images: torch.Tensor) -> list[dict[str, float]]:
    if images.ndim != 4:
        raise ValueError("Expected BCHW image tensor for band statistics")

    stats: list[dict[str, float]] = []
    for channel_index in range(images.shape[1]):
        channel = images[:, channel_index]
        stats.append(
            {
                "band": float(channel_index),
                "min": float(channel.min().detach().cpu()),
                "max": float(channel.max().detach().cpu()),
                "mean": float(channel.mean().detach().cpu()),
                "std": float(channel.std(unbiased=False).detach().cpu()),
            }
        )
    return stats


def dataset_image_statistics(
    dataset: PairedImageDataset,
    sample_count: int,
) -> dict[str, object]:
    if sample_count <= 0:
        return {}

    left_images: list[torch.Tensor] = []
    right_images: list[torch.Tensor] = []
    for index in range(min(sample_count, len(dataset))):
        item = dataset[index]
        left_images.append(torch.as_tensor(item["left_image"]))
        right_images.append(torch.as_tensor(item["right_image"]))

    if not left_images or not right_images:
        return {}

    return {
        "sample_count": len(left_images),
        "left_modality": dataset.left_modality,
        "right_modality": dataset.right_modality,
        "left_band_stats": batch_band_statistics(torch.stack(left_images)),
        "right_band_stats": batch_band_statistics(torch.stack(right_images)),
    }


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
        "first_pair_ids": pair_ids[:5],
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


def passes_exact_pair_gate(metrics: dict[str, Any], config: TrainingConfig) -> bool:
    if config.stop_recall_at_1 <= 0 and config.stop_recall_at_10 <= 0:
        return False

    left_key = f"{config.left_modality}_to_{config.right_modality}"
    right_key = f"{config.right_modality}_to_{config.left_modality}"
    left_metrics = metrics[left_key]
    right_metrics = metrics[right_key]
    return (
        left_metrics["recall_at_1"] >= config.stop_recall_at_1
        and right_metrics["recall_at_1"] >= config.stop_recall_at_1
        and left_metrics["recall_at_10"] >= config.stop_recall_at_10
        and right_metrics["recall_at_10"] >= config.stop_recall_at_10
    )


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
    alignment_summary = pair_order_summary(train_dataset, validation_dataset)
    if config.require_validation_pair_alignment:
        assert_validation_pair_alignment(alignment_summary)
    preprocessing_diagnostics = dataset_image_statistics(
        train_dataset,
        sample_count=config.diagnostic_sample_count,
    )
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
        shared_backbone=config.shared_backbone,
    ).to(config.device)
    logit_scale_parameter: torch.nn.Parameter | None = None
    if config.learnable_temperature:
        logit_scale_parameter = torch.nn.Parameter(
            torch.tensor(
                np.log(1.0 / config.temperature),
                dtype=torch.float32,
                device=config.device,
            )
        )
    trainable_parameters = list(model.parameters())
    if logit_scale_parameter is not None:
        trainable_parameters.append(logit_scale_parameter)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
        diagnostic_sums: dict[str, float] = {}
        first_batch_band_stats: dict[str, object] = {}

        for batch in loader:
            assert_aligned_pair_ids(batch)
            left_image = batch["left_image"].to(config.device)
            right_image = batch["right_image"].to(config.device)

            left_embeddings = model.encode(left_image, str(batch["left_modality"]))
            right_embeddings = model.encode(right_image, str(batch["right_modality"]))
            temperature = current_temperature(config, logit_scale_parameter, config.device)
            loss = bidirectional_pair_loss(
                left_embeddings,
                right_embeddings,
                temperature=temperature,
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
                    temperature=temperature,
                )
                loss = loss + config.semantic_loss_weight * semantic_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            batch_diagnostics = embedding_diagnostics(
                left_embeddings,
                right_embeddings,
                collapse_similarity_threshold=config.collapse_similarity_threshold,
            )
            batch_diagnostics.update(temperature_diagnostics(temperature))
            batch_diagnostics["gradient_norm"] = gradient_norm(trainable_parameters)
            for key, value in batch_diagnostics.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value

            if not first_batch_band_stats:
                first_batch_band_stats = {
                    "left_band_stats": batch_band_statistics(left_image.detach().cpu()),
                    "right_band_stats": batch_band_statistics(right_image.detach().cpu()),
                    "left_shape": list(left_image.shape),
                    "right_shape": list(right_image.shape),
                    "spatial_transform": "deterministic resize only; no augmentation",
                    "augmentations_enabled": False,
                }
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        average_loss = total_loss / max(batches, 1)
        epoch_diagnostics: dict[str, object] = {
            key: value / max(batches, 1) for key, value in diagnostic_sums.items()
        }
        epoch_diagnostics.update(first_batch_band_stats)
        validation_metrics = exact_pair_retrieval_metrics(
            model,
            validation_dataset,
            batch_size=config.batch_size,
            device=config.device,
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_pair_loss": average_loss,
            "diagnostics": epoch_diagnostics,
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
                    "pair_alignment": alignment_summary,
                    "preprocessing_diagnostics": preprocessing_diagnostics,
                    "best_epoch": best_epoch,
                    "best_mean_recall_at_10": best_recall_at_10,
                    "best_mean_recall_at_1": best_recall_at_1,
                    "best_validation": validation_metrics,
                },
            )

        if passes_exact_pair_gate(validation_metrics, config):
            break

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
        "pair_alignment": alignment_summary,
        "preprocessing_diagnostics": preprocessing_diagnostics,
        "history": history,
    }
