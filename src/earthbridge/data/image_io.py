from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_image_chw(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in {".tif", ".tiff", ".jp2"}:
        try:
            import rasterio

            with rasterio.open(image_path) as dataset:
                array = dataset.read()
            return np.asarray(array)
        except Exception:
            pass

    from PIL import Image

    with Image.open(image_path) as image:
        array = np.asarray(image)

    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return np.transpose(array, (2, 0, 1))

    raise ValueError(f"Unsupported image shape for {image_path}: {array.shape}")


def _array_to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return np.transpose(array, (2, 0, 1))
    raise ValueError(f"Unsupported image shape: {array.shape}")


def load_image_chw_from_bytes(filename: str, content: bytes) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff", ".jp2"}:
        try:
            from rasterio.io import MemoryFile

            with MemoryFile(content) as memory_file:
                with memory_file.open() as dataset:
                    return np.asarray(dataset.read())
        except Exception:
            pass

    from PIL import Image

    with Image.open(BytesIO(content)) as image:
        return _array_to_chw(np.asarray(image))


def normalize_image(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)

    if np.issubdtype(array.dtype, np.integer):
        max_value = np.iinfo(array.dtype).max
        if max_value > 0:
            return image / max_value

    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    valid = image[finite]
    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    image[~finite] = 0.0
    return image.astype(np.float32)


def ensure_channels(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError("Expected a CHW tensor")
    if expected_channels <= 0:
        raise ValueError("expected_channels must be positive")

    channels = tensor.shape[0]
    if channels == expected_channels:
        return tensor
    if channels > expected_channels:
        return tensor[:expected_channels]

    padding = torch.zeros(
        expected_channels - channels,
        tensor.shape[1],
        tensor.shape[2],
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, padding], dim=0)


def resize_tensor(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if tensor.ndim != 3:
        raise ValueError("Expected a CHW tensor")

    resized = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def load_image_tensor(
    path: str | Path,
    image_size: int,
    expected_channels: int,
) -> torch.Tensor:
    array = load_image_chw(path)
    normalized = normalize_image(array)
    tensor = torch.from_numpy(normalized.astype(np.float32))
    tensor = ensure_channels(tensor, expected_channels)
    return resize_tensor(tensor, image_size)


def load_image_tensor_from_bytes(
    filename: str,
    content: bytes,
    image_size: int,
    expected_channels: int,
) -> torch.Tensor:
    array = load_image_chw_from_bytes(filename, content)
    normalized = normalize_image(array)
    tensor = torch.from_numpy(normalized.astype(np.float32))
    tensor = ensure_channels(tensor, expected_channels)
    return resize_tensor(tensor, image_size)
