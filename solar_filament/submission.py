from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .masks import decode_mask, encode_mask


FIELDNAMES = ("filament_id", "segmentation_rle")


@dataclass(frozen=True)
class SubmissionReport:
    rows: int
    image_stems: tuple[str, ...]
    errors: tuple[str, ...]


def remove_overlaps(
    masks: Sequence[np.ndarray], min_area: int = 1
) -> list[np.ndarray]:
    if min_area < 1:
        raise ValueError("min_area must be positive")
    occupied: np.ndarray | None = None
    result: list[np.ndarray] = []
    for value in masks:
        mask = np.asarray(value, dtype=bool)
        if occupied is None:
            occupied = np.zeros_like(mask)
        if mask.shape != occupied.shape:
            raise ValueError("inconsistent mask shapes")
        mask = mask & ~occupied
        if int(mask.sum()) < min_area:
            continue
        occupied |= mask
        result.append(mask.astype(np.uint8))
    return result


def build_submission_rows(
    predictions: Mapping[str, Sequence[np.ndarray]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_stem in sorted(predictions):
        try:
            masks = remove_overlaps(predictions[image_stem])
        except ValueError as exc:
            raise ValueError(f"{exc} for {image_stem}") from exc
        for index, mask in enumerate(masks, start=1):
            rows.append(
                {
                    "filament_id": f"{image_stem}_{index}",
                    "segmentation_rle": encode_mask(mask),
                }
            )
    return rows


def write_submission(path: Path | str, rows: Iterable[Mapping[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_submission(
    path: Path | str,
    expected_stems: set[str] | None = None,
    height: int = 2048,
    width: int = 2048,
) -> SubmissionReport:
    errors: list[str] = []
    seen_ids: set[str] = set()
    image_stems: set[str] = set()
    rows = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDNAMES:
            return SubmissionReport(
                rows=0,
                image_stems=(),
                errors=(f"expected columns {FIELDNAMES}, got {tuple(reader.fieldnames or ())}",),
            )
        for line_number, row in enumerate(reader, start=2):
            rows += 1
            filament_id = row["filament_id"]
            if filament_id in seen_ids:
                errors.append(f"line {line_number}: duplicate filament_id {filament_id}")
            seen_ids.add(filament_id)
            if "_" not in filament_id:
                errors.append(f"line {line_number}: invalid filament_id {filament_id}")
                continue
            stem, suffix = filament_id.rsplit("_", 1)
            if not stem or not suffix:
                errors.append(f"line {line_number}: invalid filament_id {filament_id}")
                continue
            image_stems.add(stem)
            if expected_stems is not None and stem not in expected_stems:
                errors.append(f"line {line_number}: unknown image stem {stem}")
            try:
                mask = decode_mask(row["segmentation_rle"], height=height, width=width)
                if not mask.any():
                    errors.append(f"line {line_number}: decoded mask is empty")
            except Exception as exc:
                errors.append(f"line {line_number}: invalid RLE ({exc})")
    return SubmissionReport(
        rows=rows,
        image_stems=tuple(sorted(image_stems)),
        errors=tuple(errors),
    )
