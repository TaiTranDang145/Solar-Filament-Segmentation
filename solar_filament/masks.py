from __future__ import annotations

from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def rasterize_instances(
    annotations: Iterable[Mapping[str, Any]], height: int, width: int
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for annotation in annotations:
        polygons = annotation["segmentation"]
        rles = mask_utils.frPyObjects(polygons, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
        masks.append((decoded > 0).astype(np.uint8))
    return masks


def semantic_union(instances: Iterable[np.ndarray], height: int, width: int) -> np.ndarray:
    target = np.zeros((height, width), dtype=np.uint8)
    for instance in instances:
        target |= np.asarray(instance, dtype=np.uint8)
    return target


def connected_components(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 32,
    connectivity: int = 8,
) -> list[np.ndarray]:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (height, width)")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if min_area < 1:
        raise ValueError("min_area must be at least 1")

    binary = (probabilities >= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=connectivity)
    instances: list[np.ndarray] = []
    for label in range(1, count):
        instance = (labels == label).astype(np.uint8)
        if int(instance.sum()) >= min_area:
            instances.append(instance)
    instances.sort(key=lambda mask: int(mask.sum()), reverse=True)
    return instances


def encode_mask(mask: np.ndarray) -> str:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must have shape (height, width)")
    if not binary.any():
        raise ValueError("empty masks must be omitted from submissions")
    encoded = mask_utils.encode(np.asfortranarray(binary))
    counts = encoded["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode_mask(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    if not counts:
        raise ValueError("RLE counts must not be empty")
    decoded = mask_utils.decode(
        {"size": [height, width], "counts": counts.encode("ascii")}
    )
    return (decoded > 0).astype(np.uint8)
