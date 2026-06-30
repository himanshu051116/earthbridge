import sys
from pathlib import Path

from earthbridge.api.service import RetrievalService

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from create_demo_index import create_demo_artifacts  # noqa: E402


def test_create_demo_artifacts_produces_api_ready_bundle(tmp_path):
    artifacts = create_demo_artifacts(
        output_dir=tmp_path / "demo",
        pairs=1,
        image_size=16,
        embedding_dim=8,
        seed=123,
    )

    for path in artifacts.values():
        assert Path(path).exists()

    service = RetrievalService.from_paths(
        index_path=artifacts["index"],
        ids_path=artifacts["index_ids"],
        checkpoint_path=artifacts["checkpoint"],
        manifest_path=artifacts["manifest"],
        gallery_root=artifacts["gallery_root"],
    )

    query_path = Path(artifacts["gallery_root"]) / "images" / "pair_0000_s2.tif"
    response = service.retrieve_image(
        filename=query_path.name,
        content=query_path.read_bytes(),
        query_modality="multispectral",
        target_modality="sar",
        top_k=1,
    )

    assert service.index_loaded
    assert service.model is not None
    assert response["results"][0]["modality"] == "sar"
