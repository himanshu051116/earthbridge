import numpy as np
import rasterio
import torch
from PIL import Image

from earthbridge.data.image_io import (
    ensure_channels,
    load_image_tensor,
    load_image_tensor_from_bytes,
    load_preview_png,
    normalize_image,
)


def write_tiff(path, bands: int) -> None:
    data = np.arange(bands * 12 * 10, dtype=np.uint16).reshape(bands, 12, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=12,
        width=10,
        count=bands,
        dtype=data.dtype,
    ) as dataset:
        dataset.write(data)


def test_load_image_tensor_returns_expected_channels_and_size(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.fromarray(np.full((12, 10, 3), 128, dtype=np.uint8)).save(image_path)

    tensor = load_image_tensor(image_path, image_size=16, expected_channels=3)

    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_load_image_tensor_reads_two_band_sentinel1_tiff(tmp_path):
    image_path = tmp_path / "s1.tif"
    write_tiff(image_path, bands=2)

    tensor = load_image_tensor(image_path, image_size=16, expected_channels=2)

    assert tensor.shape == (2, 16, 16)
    assert tensor.dtype == torch.float32


def test_load_image_tensor_reads_ten_band_sentinel2_tiff(tmp_path):
    image_path = tmp_path / "s2.tif"
    write_tiff(image_path, bands=10)

    tensor = load_image_tensor(image_path, image_size=16, expected_channels=10)

    assert tensor.shape == (10, 16, 16)
    assert tensor.dtype == torch.float32


def test_normalize_image_uses_band_wise_percentiles():
    low_range = np.array([[0, 1], [2, 3]], dtype=np.float32)
    high_range = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
    image = np.stack([low_range, high_range])

    normalized = normalize_image(image)

    assert normalized.shape == (2, 2, 2)
    assert np.isclose(normalized[0].min(), 0.0)
    assert np.isclose(normalized[0].max(), 1.0)
    assert np.isclose(normalized[1].min(), 0.0)
    assert np.isclose(normalized[1].max(), 1.0)


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


def test_load_preview_png_returns_browser_safe_png(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.fromarray(np.full((12, 10), 128, dtype=np.uint8)).save(image_path)

    preview = load_preview_png(image_path)

    assert preview.startswith(b"\x89PNG")
