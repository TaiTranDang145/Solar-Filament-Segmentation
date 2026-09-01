from __future__ import annotations

import numpy as np


def native_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    """Return a centered, per-frame normalized grayscale crop."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 2 or crop_size < 1 or crop_size > min(image.shape):
        raise ValueError("native crop must fit inside a grayscale image")
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    radius = min(900, height // 2, width // 2)
    disk = (y - height // 2) ** 2 + (x - width // 2) ** 2 <= radius**2
    histogram = np.bincount(image[disk], minlength=256)
    cumulative = np.cumsum(histogram) / histogram.sum()
    low, center, high = np.searchsorted(cumulative, (0.16, 0.5, 0.84))
    top, left = (height - crop_size) // 2, (width - crop_size) // 2
    crop = image[top : top + crop_size, left : left + crop_size].astype(np.float32)
    return (crop - float(center)) / float(max(high - low, 1))
