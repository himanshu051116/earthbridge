import torch
from torch import nn

from earthbridge.models import BaselineRetriever, EarthBridgeDualHead


def test_baseline_retriever_outputs_normalized_descriptors():
    model = BaselineRetriever({"optical_rgb": 3, "sar": 2}, embedding_dim=16)
    model.eval()

    with torch.no_grad():
        descriptors = model(torch.randn(2, 3, 32, 32), "optical_rgb")

    assert descriptors.shape == (2, 16)
    assert torch.allclose(descriptors.norm(dim=1), torch.ones(2), atol=1e-5)


def test_baseline_retriever_accepts_modality_specific_channels():
    model = BaselineRetriever({"optical_rgb": 3, "sar": 2}, embedding_dim=16)
    model.eval()

    with torch.no_grad():
        descriptors = model(torch.randn(2, 2, 32, 32), "sar")

    assert descriptors.shape == (2, 16)


def test_baseline_retriever_uses_groupnorm_not_batchnorm():
    model = BaselineRetriever({"optical_rgb": 3, "sar": 2}, embedding_dim=16)

    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())


def test_projection_dropout_is_configurable():
    model = BaselineRetriever(
        {"optical_rgb": 3, "sar": 2},
        embedding_dim=16,
        projection_dropout=0.0,
    )
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]

    assert dropouts
    assert {dropout.p for dropout in dropouts} == {0.0}


def test_dual_head_outputs_cross_and_same_embeddings():
    model = EarthBridgeDualHead({"optical_rgb": 3, "sar": 2}, embedding_dim=16)
    model.eval()

    with torch.no_grad():
        cross = model.encode_cross(torch.randn(2, 3, 32, 32), "optical_rgb")
        same = model.encode_same(torch.randn(2, 3, 32, 32), "optical_rgb")

    assert cross.shape == (2, 16)
    assert same.shape == (2, 16)
    assert torch.allclose(cross.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(same.norm(dim=1), torch.ones(2), atol=1e-5)
