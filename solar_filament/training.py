from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .data import AnnotationSet, ImageRecord, assign_folds, build_manifest, load_coco


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    output_dir: str = "artifacts/fold-0"
    fold: int = 0
    n_folds: int = 5
    seed: int = 2026
    epochs: int = 20
    image_size: int = 768
    batch_size: int = 2
    num_workers: int = 2
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    positive_weight: float = 12.0
    threshold: float = 0.5
    min_area: int = 32
    pretrained_backbone: bool = True
    model_name: str = "deeplabv3_resnet50"
    gradient_accumulation: int = 1
    tta: int = 1
    close_kernel: int = 0

    def __post_init__(self) -> None:
        if self.model_name not in ("deeplabv3_resnet50", "native_unet"):
            raise ValueError(f"unknown model_name: {self.model_name}")
        if self.batch_size < 2 and self.model_name == "deeplabv3_resnet50":
            raise ValueError("batch_size must be at least 2 for DeepLabV3 BatchNorm")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be positive")
        if self.tta not in (1, 4, 8):
            raise ValueError("tta must be 1, 4, or 8")
        if self.close_kernel not in (0, 1) and (
            self.close_kernel < 3 or self.close_kernel % 2 == 0
        ):
            raise ValueError("close_kernel must be zero, one, or an odd integer at least 3")


def _stable_index(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % modulo


def select_annotation_set(record: ImageRecord, epoch: int, seed: int) -> AnnotationSet:
    if not record.annotation_sets:
        raise ValueError(f"{record.file_name} has no annotation sets")
    start = _stable_index(f"{seed}:{record.file_name}", len(record.annotation_sets))
    return record.annotation_sets[(start + epoch) % len(record.annotation_sets)]


class FilamentDataset:
    def __init__(
        self,
        records: Sequence[ImageRecord],
        image_dir: Path,
        image_size: int,
        seed: int,
        augment: bool,
        native: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.seed = seed
        self.augment = augment
        self.native = native
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform

        from .masks import rasterize_instances, semantic_union

        record = self.records[index]
        selected = select_annotation_set(record, self.epoch, self.seed)
        with Image.open(self.image_dir / record.file_name) as source:
            source = source.convert("L")
            if self.native:
                from .preprocessing import native_crop

                image = torch.from_numpy(
                    native_crop(np.asarray(source), self.image_size)
                ).unsqueeze(0)
            else:
                image = transform.pil_to_tensor(source).float() / 255.0
        instances = rasterize_instances(selected.annotations, record.height, record.width)
        target_array = semantic_union(instances, record.height, record.width)
        target = torch.from_numpy(np.asarray(target_array)).unsqueeze(0).float()
        if self.native:
            top = (record.height - self.image_size) // 2
            left = (record.width - self.image_size) // 2
            target = target[:, top : top + self.image_size, left : left + self.image_size]
        else:
            image = image.repeat(3, 1, 1)
            image = transform.resize(image, [self.image_size, self.image_size], antialias=True)
            target = transform.resize(
                target,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )

        if self.augment:
            code = _stable_index(
                f"{self.seed}:{self.epoch}:{record.file_name}:augmentation", 8
            )
            if code & 1:
                image, target = transform.hflip(image), transform.hflip(target)
            if code & 2:
                image, target = transform.vflip(image), transform.vflip(target)
            if code & 4:
                image, target = torch.rot90(image, 1, (-2, -1)), torch.rot90(target, 1, (-2, -1))
        return image, target, record.file_name


def segmentation_loss(logits, target, positive_weight: float):
    import torch
    from torch.nn import functional as functional

    weight = torch.tensor([positive_weight], device=logits.device, dtype=logits.dtype)
    bce = functional.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice_loss


def build_training_loader(dataset, config: TrainConfig):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def optimizer_step_batches(batch_count: int, accumulation: int) -> tuple[int, ...]:
    if batch_count < 0 or accumulation < 1:
        raise ValueError("batch_count must not be negative and accumulation must be positive")
    steps = list(range(accumulation - 1, batch_count, accumulation))
    if batch_count and (not steps or steps[-1] != batch_count - 1):
        steps.append(batch_count - 1)
    return tuple(steps)


def gradient_divisor(batch_index: int, batch_count: int, accumulation: int) -> int:
    if not 0 <= batch_index < batch_count or accumulation < 1:
        raise ValueError("batch_index must identify a batch and accumulation must be positive")
    remainder = batch_count % accumulation
    if remainder and batch_index >= batch_count - remainder:
        return remainder
    return accumulation


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _records(config: TrainConfig) -> tuple[list[ImageRecord], list[ImageRecord], Path]:
    data_root = Path(config.data_root)
    coco = load_coco(
        data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    manifest = build_manifest(coco)
    folds = assign_folds(manifest, n_folds=config.n_folds, seed=config.seed)
    train_records = [row for row in manifest if folds[row.file_name] != config.fold]
    validation_records = [row for row in manifest if folds[row.file_name] == config.fold]
    return train_records, validation_records, data_root / "train" / "train_images"


def prediction_entries(model, records: Sequence[ImageRecord], image_dir: Path, config: TrainConfig, device):
    from .inference import predict_image
    from .masks import rasterize_instances

    for record in records:
        predictions, _ = predict_image(
            model,
            image_dir / record.file_name,
            image_size=config.image_size,
            threshold=config.threshold,
            min_area=config.min_area,
            device=device,
            native=config.model_name == "native_unet",
            tta=config.tta,
            close_kernel=config.close_kernel,
        )
        for annotation_set in record.annotation_sets:
            ground_truth = rasterize_instances(
                annotation_set.annotations, height=record.height, width=record.width
            )
            yield ground_truth, predictions


def evaluate_model(model, records: Sequence[ImageRecord], image_dir: Path, config: TrainConfig, device):
    from .metrics import evaluate_annotation_sets

    return evaluate_annotation_sets(prediction_entries(model, records, image_dir, config, device))


def train(config: TrainConfig) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("training requires torch and torchvision") from exc

    from .model import build_model

    _set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    train_records, validation_records, image_dir = _records(config)
    dataset = FilamentDataset(
        train_records,
        image_dir,
        config.image_size,
        config.seed,
        augment=True,
        native=config.model_name == "native_unet",
    )
    loader = build_training_loader(dataset, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.pretrained_backbone, config.model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_pq = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(config.epochs):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        update_batches = set(
            optimizer_step_batches(len(loader), config.gradient_accumulation)
        )
        for batch_index, (images, targets, _) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)["out"]
                loss = segmentation_loss(logits, targets, config.positive_weight)
            scaler.scale(
                loss
                / gradient_divisor(
                    batch_index, len(loader), config.gradient_accumulation
                )
            ).backward()
            if batch_index in update_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * images.shape[0]

        report = evaluate_model(
            model, validation_records, image_dir, replace(config, tta=1), device
        )
        epoch_report: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(dataset),
            **asdict(report),
        }
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_report) + "\n")
        if report.official_pq > best_pq:
            best_pq = report.official_pq
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch + 1,
                    "official_pq": best_pq,
                },
                best_path,
            )
    return best_path
