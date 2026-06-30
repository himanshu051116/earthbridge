import numpy as np
import torch
from PIL import Image

from earthbridge.api.service import RetrievalService
from earthbridge.models import BaselineRetriever
from earthbridge.retrieval.faiss_index import ExactFaissIndex


def test_retrieval_service_loads_index_and_retrieves(tmp_path):
    index_path = tmp_path / "demo.index"
    ids_path = tmp_path / "demo_ids.json"
    ExactFaissIndex.build(
        ["A", "B"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ).save(index_path, ids_path)

    service = RetrievalService.from_paths(index_path, ids_path)
    response = service.retrieve_descriptor([1.0, 0.0], top_k=1)

    assert service.index_loaded
    assert response["results"][0]["gallery_id"] == "A"


def test_retrieval_service_reports_not_loaded_for_missing_index(tmp_path):
    service = RetrievalService.from_paths(tmp_path / "missing.index", tmp_path / "missing.json")

    assert not service.index_loaded
    assert service.model_info()["index_type"] == "not_loaded"


def test_retrieval_service_retrieves_from_uploaded_image(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    query_path = image_dir / "query.png"
    gallery_path = image_dir / "s1.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(query_path)
    Image.fromarray(np.full((16, 16), 64, dtype=np.uint8)).save(gallery_path)

    index_path = tmp_path / "gallery.index"
    ids_path = tmp_path / "gallery_ids.json"
    ExactFaissIndex.build(
        ["S1_A", "S2_A"],
        np.array([[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6], dtype=np.float32),
    ).save(index_path, ids_path)

    manifest_path = tmp_path / "test.csv"
    manifest_path.write_text(
        "sample_id,image_path,modality,pair_id,labels,channels\n"
        "S1_A,images/s1.png,sar,P1,Water,1\n"
        "S2_A,images/query.png,multispectral,P1,Water,3\n",
        encoding="utf-8",
    )

    model = BaselineRetriever(
        {"optical_rgb": 3, "sar": 1, "multispectral": 3},
        embedding_dim=8,
    )
    checkpoint_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "config": {
                    "model_type": "baseline",
                    "image_size": 16,
                    "embedding_dim": 8,
                    "backbone": "small_cnn",
                },
                "modality_channels": {"optical_rgb": 3, "sar": 1, "multispectral": 3},
            },
        },
        checkpoint_path,
    )

    service = RetrievalService.from_paths(
        index_path=index_path,
        ids_path=ids_path,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        gallery_root=tmp_path,
    )
    response = service.retrieve_image(
        filename="query.png",
        content=query_path.read_bytes(),
        query_modality="optical_rgb",
        target_modality="sar",
        top_k=1,
    )

    assert response["retrieval_type"] == "cross_modal"
    assert response["results"][0]["gallery_id"] == "S1_A"
    assert response["results"][0]["modality"] == "sar"
    assert service.gallery_path("S1_A") == gallery_path
    assert service.gallery_preview("S1_A").startswith(b"\x89PNG")
