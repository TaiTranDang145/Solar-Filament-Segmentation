from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


@dataclass(frozen=True)
class AnnotationSet:
    image_id: str
    annotations: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    width: int
    height: int
    year: int
    observatory: str
    annotation_sets: tuple[AnnotationSet, ...]

    @property
    def max_instances(self) -> int:
        return max((len(item.annotations) for item in self.annotation_sets), default=0)


@dataclass(frozen=True)
class AuditReport:
    train_files: int
    test_files: int
    image_records: int
    physical_images: int
    annotations: int
    annotation_sets_per_image: Mapping[int, int]
    category_counts: Mapping[int, int]
    observatory_train: Mapping[str, int]
    observatory_test: Mapping[str, int]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_files": self.train_files,
            "test_files": self.test_files,
            "image_records": self.image_records,
            "physical_images": self.physical_images,
            "annotations": self.annotations,
            "annotation_sets_per_image": dict(self.annotation_sets_per_image),
            "category_counts": dict(self.category_counts),
            "observatory_train": dict(self.observatory_train),
            "observatory_test": dict(self.observatory_test),
            "errors": list(self.errors),
        }


def load_coco(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(coco: Mapping[str, Any]) -> list[ImageRecord]:
    annotations_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[str(annotation["image_id"])].append(annotation)

    images_by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for image in coco.get("images", []):
        images_by_file[str(image["file_name"])].append(image)

    manifest: list[ImageRecord] = []
    for file_name in sorted(images_by_file):
        images = sorted(images_by_file[file_name], key=lambda item: str(item["id"]))
        first = images[0]
        stem = Path(file_name).stem
        annotation_sets = tuple(
            AnnotationSet(
                image_id=str(image["id"]),
                annotations=tuple(annotations_by_image.get(str(image["id"]), [])),
            )
            for image in images
        )
        manifest.append(
            ImageRecord(
                file_name=file_name,
                width=int(first["width"]),
                height=int(first["height"]),
                year=int(stem[:4]),
                observatory=stem[-2:],
                annotation_sets=annotation_sets,
            )
        )
    return manifest


def _instance_bucket(count: int) -> str:
    if count <= 4:
        return "few"
    if count <= 10:
        return "medium"
    return "many"


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_folds(
    manifest: Iterable[ImageRecord], n_folds: int = 5, seed: int = 2026
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    strata: dict[tuple[int, str, str], list[ImageRecord]] = defaultdict(list)
    for row in manifest:
        strata[(row.year, row.observatory, _instance_bucket(row.max_instances))].append(row)

    folds: dict[str, int] = {}
    fold_sizes = [0] * n_folds
    for stratum in sorted(strata):
        rows = sorted(strata[stratum], key=lambda row: _stable_order(row.file_name, seed))
        start = min(range(n_folds), key=lambda index: (fold_sizes[index], index))
        for offset, row in enumerate(rows):
            fold = (start + offset) % n_folds
            folds[row.file_name] = fold
            fold_sizes[fold] += 1
    return folds


def save_folds(path: Path | str, folds: Mapping[str, int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(folds.items())), indent=2) + "\n", encoding="utf-8")


def _image_errors(path: Path, expected_size: tuple[int, int]) -> list[str]:
    try:
        with Image.open(path) as image:
            errors = []
            if image.mode != "L":
                errors.append(f"{path}: expected grayscale L, got {image.mode}")
            if image.size != expected_size:
                errors.append(f"{path}: expected {expected_size}, got {image.size}")
            image.verify()
            return errors
    except Exception as exc:  # Pillow exposes several decoder-specific errors.
        return [f"{path}: unreadable image ({exc})"]


def audit_dataset(root: Path | str) -> AuditReport:
    root = Path(root)
    train_dir = root / "train" / "train_images"
    test_dir = root / "test" / "test_images"
    annotation_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = load_coco(annotation_path)
    manifest = build_manifest(coco)
    train_files = sorted(train_dir.glob("*.jpeg"))
    test_files = sorted(test_dir.glob("*.jpeg"))
    errors: list[str] = []

    file_names = {path.name for path in train_files}
    image_ids = {str(image["id"]) for image in coco.get("images", [])}
    for row in manifest:
        if row.file_name not in file_names:
            errors.append(f"missing train image: {row.file_name}")
    for annotation in coco.get("annotations", []):
        if str(annotation.get("image_id")) not in image_ids:
            errors.append(f"orphan annotation: {annotation.get('id')}")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or len(segmentation) != 1:
            errors.append(f"annotation {annotation.get('id')}: expected one polygon")
            continue
        polygon = segmentation[0]
        if len(polygon) < 6 or len(polygon) % 2:
            errors.append(f"annotation {annotation.get('id')}: invalid polygon length")

    for path in train_files + test_files:
        errors.extend(_image_errors(path, (2048, 2048)))

    sets_per_image = Counter(len(row.annotation_sets) for row in manifest)
    categories = Counter(int(annotation["category_id"]) for annotation in coco.get("annotations", []))
    return AuditReport(
        train_files=len(train_files),
        test_files=len(test_files),
        image_records=len(coco.get("images", [])),
        physical_images=len(manifest),
        annotations=len(coco.get("annotations", [])),
        annotation_sets_per_image=dict(sorted(sets_per_image.items())),
        category_counts=dict(sorted(categories.items())),
        observatory_train=dict(sorted(Counter(path.stem[-2:] for path in train_files).items())),
        observatory_test=dict(sorted(Counter(path.stem[-2:] for path in test_files).items())),
        errors=tuple(errors),
    )
