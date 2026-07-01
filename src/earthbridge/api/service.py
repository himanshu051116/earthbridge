from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from earthbridge.data.dataset import DEFAULT_CHANNELS, infer_modality_channels
from earthbridge.data.image_io import load_image_tensor_from_bytes, load_preview_png
from earthbridge.data.manifest import load_manifest
from earthbridge.models import BaselineRetriever, EarthBridgeDualHead
from earthbridge.retrieval.faiss_index import ExactFaissIndex


@dataclass
class RetrievalService:
    index: ExactFaissIndex | None
    name: str = "earthbridge-baseline"
    model: torch.nn.Module | None = None
    model_type: str = "baseline"
    image_size: int = 128
    embedding_dim: int = 256
    modality_channels: dict[str, int] | None = None
    gallery_records: dict[str, dict[str, str]] | None = None
    gallery_root: Path = Path(".")
    device: str = "cpu"

    @classmethod
    def from_paths(
        cls,
        index_path: str | Path,
        ids_path: str | Path,
        checkpoint_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        gallery_root: str | Path = ".",
        device: str = "cpu",
    ) -> RetrievalService:
        index_file = Path(index_path)
        ids_file = Path(ids_path)
        gallery_records = load_gallery_records(manifest_path) if manifest_path else {}
        if not index_file.exists() or not ids_file.exists():
            return cls(
                index=None,
                gallery_records=gallery_records,
                gallery_root=Path(gallery_root),
                device=device,
            )

        service = cls(
            index=ExactFaissIndex.load(index_file, ids_file),
            gallery_records=gallery_records,
            gallery_root=Path(gallery_root),
            device=device,
        )
        if checkpoint_path:
            service.load_model(checkpoint_path)
        return service

    @classmethod
    def from_environment(cls) -> RetrievalService:
        index_path = os.getenv("EARTHBRIDGE_INDEX_PATH", "artifacts/indexes/gallery.index")
        ids_path = os.getenv("EARTHBRIDGE_IDS_PATH", "artifacts/indexes/gallery_ids.json")
        checkpoint_path = os.getenv(
            "EARTHBRIDGE_CHECKPOINT_PATH",
            "artifacts/checkpoints/baseline_pair.pt",
        )
        manifest_path = os.getenv("EARTHBRIDGE_GALLERY_MANIFEST", "artifacts/manifests/test.csv")
        gallery_root = os.getenv("EARTHBRIDGE_GALLERY_ROOT", ".")
        device = os.getenv("EARTHBRIDGE_DEVICE", "cpu")
        return cls.from_paths(
            index_path=index_path,
            ids_path=ids_path,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            gallery_root=gallery_root,
            device=device,
        )

    @property
    def index_loaded(self) -> bool:
        return self.index is not None

    def model_info(self) -> dict[str, object]:
        if self.index is None:
            return {
                "name": self.name,
                "embedding_dim": None,
                "index_size": 0,
                "index_type": "not_loaded",
                "model_loaded": self.model is not None,
                "gallery_metadata_loaded": bool(self.gallery_records),
            }

        return {
            "name": self.name,
            "embedding_dim": int(self.index.index.d),
            "index_size": len(self.index.ids),
            "index_type": "faiss.IndexFlatIP",
            "model_loaded": self.model is not None,
            "gallery_metadata_loaded": bool(self.gallery_records),
        }

    def load_model(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            return

        payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
        state_dict = (
            payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        )

        self.model_type = str(config.get("model_type", "baseline"))
        self.image_size = int(config.get("image_size", self.image_size))
        self.embedding_dim = int(config.get("embedding_dim", self.embedding_dim))
        backbone = str(config.get("backbone", "small_cnn"))
        projection_dropout = float(config.get("projection_dropout", 0.1))
        self.modality_channels = metadata.get("modality_channels") or infer_channels_from_records(
            self.gallery_records or {}
        )
        if not self.modality_channels:
            self.modality_channels = dict(DEFAULT_CHANNELS)

        if self.model_type == "dual_head":
            model: torch.nn.Module = EarthBridgeDualHead(
                modality_channels=self.modality_channels,
                backbone_name=backbone,
                embedding_dim=self.embedding_dim,
                projection_dropout=projection_dropout,
            )
        else:
            model = BaselineRetriever(
                modality_channels=self.modality_channels,
                backbone_name=backbone,
                embedding_dim=self.embedding_dim,
                projection_dropout=projection_dropout,
            )

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def retrieve_descriptor(
        self,
        descriptor: list[float],
        top_k: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> dict[str, object]:
        if self.index is None:
            raise RuntimeError(
                "No index loaded. Create one with scripts/create_demo_index.py "
                "or scripts/build_indexes.py"
            )

        response = self.index.search(
            np.asarray(descriptor, dtype=np.float32),
            top_k=top_k,
            exclude_ids=set(exclude_ids or []),
        )
        results = self.format_results(response.results)
        return {
            "retrieval_type": "descriptor",
            "retrieval_time_ms": response.search_time_ms,
            "results": results,
        }

    def retrieve_image(
        self,
        filename: str,
        content: bytes,
        query_modality: str,
        target_modality: str,
        top_k: int = 10,
    ) -> dict[str, object]:
        if self.index is None:
            raise RuntimeError("No FAISS index loaded")
        if self.model is None:
            raise RuntimeError("No model checkpoint loaded")
        if not self.modality_channels or query_modality not in self.modality_channels:
            raise ValueError(f"Unknown query modality: {query_modality}")

        image = load_image_tensor_from_bytes(
            filename=filename,
            content=content,
            image_size=self.image_size,
            expected_channels=self.modality_channels[query_modality],
        ).unsqueeze(0)

        start = perf_counter()
        with torch.no_grad():
            image = image.to(self.device)
            if isinstance(self.model, BaselineRetriever):
                descriptor = self.model.encode(image, query_modality)
            elif query_modality == target_modality:
                descriptor = self.model.encode_same(image, query_modality)
            else:
                descriptor = self.model.encode_cross(image, query_modality)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        response = self.index.search(
            descriptor.cpu().numpy()[0],
            top_k=top_k + 50 if target_modality else top_k,
            overfetch=50,
        )
        elapsed_ms = (perf_counter() - start) * 1000
        retrieval_type = "same_modal" if query_modality == target_modality else "cross_modal"
        return {
            "retrieval_type": retrieval_type,
            "retrieval_time_ms": elapsed_ms,
            "results": self.format_results(
                response.results,
                target_modality=target_modality,
                limit=top_k,
            ),
        }

    def format_results(
        self,
        results: list[object],
        target_modality: str = "",
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        formatted: list[dict[str, object]] = []
        for result in results:
            record = (self.gallery_records or {}).get(result.sample_id, {})
            modality = record.get("modality", "")
            image_path = record.get("image_path", "")
            thumbnail = f"/gallery/{result.sample_id}" if image_path else ""
            if target_modality and modality and modality != target_modality:
                continue
            formatted.append(
                {
                    "rank": len(formatted) + 1,
                    "gallery_id": result.sample_id,
                    "similarity": result.score,
                    "modality": modality,
                    "labels": record.get("labels", ""),
                    "image_path": image_path,
                    "thumbnail": thumbnail,
                }
            )
            if limit is not None and len(formatted) >= limit:
                break
        return formatted

    def gallery_path(self, sample_id: str) -> Path | None:
        record = (self.gallery_records or {}).get(sample_id)
        if not record or not record.get("image_path"):
            return None
        path = self.gallery_root / record["image_path"]
        return path if path.exists() else None

    def gallery_preview(self, sample_id: str) -> bytes | None:
        path = self.gallery_path(sample_id)
        if path is None:
            return None
        return load_preview_png(path)


def load_gallery_records(manifest_path: str | Path | None) -> dict[str, dict[str, str]]:
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    return {row["sample_id"]: row for row in load_manifest(path)}


def infer_channels_from_records(records: dict[str, dict[str, str]]) -> dict[str, int]:
    if not records:
        return {}
    return infer_modality_channels(list(records.values()))
