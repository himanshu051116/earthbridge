from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RASTER_EXTENSIONS = {".tif", ".tiff", ".jp2"}
PILLOW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_image_chw(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix in RASTER_EXTENSIONS:
        import rasterio

        with rasterio.open(image_path) as dataset:
            array = dataset.read()
        return np.asarray(array)

    if suffix not in PILLOW_EXTENSIONS:
        raise ValueError(f"Unsupported image extension for {image_path}: {suffix}")

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
    if suffix in RASTER_EXTENSIONS:
        from rasterio.io import MemoryFile

        with MemoryFile(content) as memory_file:
            with memory_file.open() as dataset:
                return np.asarray(dataset.read())

    if suffix not in PILLOW_EXTENSIONS:
        raise ValueError(f"Unsupported image extension for upload {filename}: {suffix}")

    from PIL import Image

    with Image.open(BytesIO(content)) as image:
        return _array_to_chw(np.asarray(image))


def image_chw_to_preview_rgb(array: np.ndarray) -> np.ndarray:
    normalized = normalize_image(array)
    if normalized.ndim != 3:
        raise ValueError(f"Expected CHW image array, got shape {normalized.shape}")

    channels = normalized.shape[0]
    if channels == 1:
        rgb = np.repeat(normalized, repeats=3, axis=0)
    elif channels == 2:
        mean_channel = normalized.mean(axis=0, keepdims=True)
        rgb = np.concatenate([normalized[:2], mean_channel], axis=0)
    else:
        rgb = normalized[:3]

    rgb = np.transpose(rgb, (1, 2, 0))
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def load_preview_png(path: str | Path) -> bytes:
    from PIL import Image

    preview = image_chw_to_preview_rgb(load_image_chw(path))
    output = BytesIO()
    Image.fromarray(preview, mode="RGB").save(output, format="PNG")
    return output.getvalue()


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
