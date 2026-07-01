import torch

from earthbridge.losses import (
    bidirectional_hard_negative_margin_loss,
    bidirectional_pair_loss,
    multilabel_supervised_contrastive_loss,
    supervised_contrastive_loss,
)


def test_bidirectional_pair_loss_is_lower_for_aligned_pairs():
    left = torch.eye(4)
    right_aligned = torch.eye(4)
    right_misaligned = torch.roll(torch.eye(4), shifts=1, dims=0)

    aligned_loss = bidirectional_pair_loss(left, right_aligned, temperature=0.1)
    misaligned_loss = bidirectional_pair_loss(left, right_misaligned, temperature=0.1)

    assert aligned_loss < misaligned_loss


def test_bidirectional_pair_loss_accepts_tensor_temperature():
    temperature = torch.tensor(0.07, requires_grad=True)
    loss = bidirectional_pair_loss(torch.eye(4), torch.eye(4), temperature=temperature)
    loss.backward()

    assert temperature.grad is not None


def test_hard_negative_margin_loss_is_lower_for_separated_pairs():
    left = torch.eye(4)
    right_aligned = torch.eye(4)
    right_misaligned = torch.roll(torch.eye(4), shifts=1, dims=0)

    aligned_loss = bidirectional_hard_negative_margin_loss(right_aligned, left)
    misaligned_loss = bidirectional_hard_negative_margin_loss(right_misaligned, left)

    assert aligned_loss < misaligned_loss


def test_supervised_contrastive_loss_returns_zero_without_positive_pairs():
    embeddings = torch.randn(4, 8)
    labels = torch.arange(4)

    loss = supervised_contrastive_loss(embeddings, labels)

    assert loss.item() == 0.0


def test_supervised_contrastive_loss_is_finite_with_positive_pairs():
    embeddings = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])

    loss = supervised_contrastive_loss(embeddings, labels)

    assert torch.isfinite(loss)


def test_multilabel_supervised_contrastive_loss_uses_shared_labels():
    embeddings = torch.randn(4, 8)
    labels = [
        ("Arable land", "Pastures"),
        ("Pastures",),
        ("Urban fabric",),
        ("Urban fabric", "Industrial"),
    ]

    loss = multilabel_supervised_contrastive_loss(embeddings, labels)

    assert torch.isfinite(loss)


def test_multilabel_supervised_contrastive_loss_returns_zero_without_shared_labels():
    embeddings = torch.randn(3, 8)
    labels = [("A",), ("B",), ()]

    loss = multilabel_supervised_contrastive_loss(embeddings, labels)

    assert loss.item() == 0.0
