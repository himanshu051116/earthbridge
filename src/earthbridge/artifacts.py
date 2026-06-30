from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_ARTIFACTS = [
    "checkpoints/baseline_pair.pt",
    "descriptors/gallery.npy",
    "descriptors/gallery_ids.json",
    "indexes/gallery.index",
    "indexes/gallery_ids.json",
    "manifests/test.csv",
    "reports/evaluation_summary.json",
    "reports/direction_metrics.csv",
    "reports/latency_summary.json",
]


@dataclass(frozen=True)
class ArtifactCheck:
    relative_path: str
    exists: bool
    size_bytes: int
    valid: bool
    error: str = ""


def _validate_json(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def _validate_numpy(path: Path) -> None:
    import numpy as np

    np.load(path)


def _validate_checkpoint(path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint must contain model_state_dict")


def _validate_faiss(path: Path) -> None:
    import faiss

    faiss.read_index(str(path))


def validate_artifact(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        _validate_json(path)
    elif suffix == ".npy":
        _validate_numpy(path)
    elif suffix == ".pt":
        _validate_checkpoint(path)
    elif suffix == ".index":
        _validate_faiss(path)


def verify_artifacts(
    artifact_root: str | Path = "artifacts",
    required_paths: list[str] | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root)
    required = required_paths or DEFAULT_REQUIRED_ARTIFACTS
    checks: list[ArtifactCheck] = []

    for relative in required:
        path = root / relative
        exists = path.exists()
        size_bytes = path.stat().st_size if exists and path.is_file() else 0
        valid = exists and size_bytes > 0
        error = ""

        if valid and deep:
            try:
                validate_artifact(path)
            except Exception as exc:
                valid = False
                error = str(exc)

        checks.append(
            ArtifactCheck(
                relative_path=relative,
                exists=exists,
                size_bytes=size_bytes,
                valid=valid,
                error=error,
            )
        )

    missing = [check.relative_path for check in checks if not check.exists]
    invalid = [check.relative_path for check in checks if check.exists and not check.valid]

    return {
        "artifact_root": str(root),
        "required_count": len(required),
        "present_count": sum(1 for check in checks if check.exists),
        "valid_count": sum(1 for check in checks if check.valid),
        "missing": missing,
        "invalid": invalid,
        "ok": not missing and not invalid,
        "checks": [asdict(check) for check in checks],
    }


def export_artifact_bundle(
    artifact_root: str | Path = "artifacts",
    output_zip: str | Path = "artifacts/earthbridge_export.zip",
    required_paths: list[str] | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_root)
    output = Path(output_zip)
    required = required_paths or DEFAULT_REQUIRED_ARTIFACTS
    report = verify_artifacts(root, required, deep=True)

    if not report["ok"] and not allow_missing:
        missing_text = ", ".join(report["missing"] + report["invalid"])
        raise FileNotFoundError(f"Cannot export incomplete artifact set: {missing_text}")

    output.parent.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in required:
            path = root / relative
            if path.exists() and path.is_file():
                archive.write(path, arcname=f"artifacts/{relative}")
                exported.append(relative)

        manifest = {
            "artifact_root": str(root),
            "exported": exported,
            "verification": report,
        }
        archive.writestr("artifact_manifest.json", json.dumps(manifest, indent=2))

    return {
        "output_zip": str(output),
        "exported_count": len(exported),
        "exported": exported,
        "verification": report,
    }
