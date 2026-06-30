import numpy as np
import torch
from PIL import Image

from earthbridge.data.image_io import (
    ensure_channels,
    load_image_tensor,
    load_image_tensor_from_bytes,
)


def test_load_image_tensor_returns_expected_channels_and_size(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((12, 10, 3), 128, dtype=np.uint8)).save(image_path)

    tensor = load_image_tensor(image_path, image_size=16, expected_channels=3)

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_ensure_channels_pads_missing_channels():
    tensor = torch.ones(1, 4, 4)

    padded = ensure_channels(tensor, expected_channels=3)

    assert padded.shape == (3, 4, 4)
    assert torch.all(padded[0] == 1)
    assert torch.all(padded[1:] == 0)


def test_load_image_tensor_from_bytes(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((12, 10, 3), 128, dtype=np.uint8)).save(image_path)

    tensor = load_image_tensor_from_bytes(
        filename="rgb.png",
        content=image_path.read_bytes(),
        image_size=16,
        expected_channels=3,
    )

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
