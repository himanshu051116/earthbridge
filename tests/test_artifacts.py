import json
import zipfile

import numpy as np

from earthbridge.artifacts import export_artifact_bundle, verify_artifacts


def test_verify_artifacts_reports_missing_files(tmp_path):
    report = verify_artifacts(tmp_path, required_paths=["missing.npy"])

    assert not report["ok"]
    assert report["missing"] == ["missing.npy"]


def test_export_artifact_bundle_allows_partial_export(tmp_path):
    artifact_root = tmp_path / "artifacts"
    descriptor_dir = artifact_root / "descriptors"
    descriptor_dir.mkdir(parents=True)
    np.save(descriptor_dir / "gallery.npy", np.zeros((2, 4), dtype=np.float32))

    output_zip = tmp_path / "export.zip"
    result = export_artifact_bundle(
        artifact_root=artifact_root,
        output_zip=output_zip,
        required_paths=["descriptors/gallery.npy", "reports/missing.json"],
        allow_missing=True,
    )

    with zipfile.ZipFile(output_zip) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("artifact_manifest.json"))

    assert result["exported"] == ["descriptors/gallery.npy"]
    assert "artifacts/descriptors/gallery.npy" in names
    assert manifest["verification"]["missing"] == ["reports/missing.json"]

