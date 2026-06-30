import torch

from earthbridge.losses import bidirectional_pair_loss, supervised_contrastive_loss


def test_bidirectional_pair_loss_is_lower_for_aligned_pairs():
    left = torch.eye(4)
    right_aligned = torch.eye(4)
    right_misaligned = torch.roll(torch.eye(4), shifts=1, dims=0)

    aligned_loss = bidirectional_pair_loss(left, right_aligned, temperature=0.1)
    misaligned_loss = bidirectional_pair_loss(left, right_misaligned, temperature=0.1)

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

