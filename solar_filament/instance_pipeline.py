from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data import AnnotationSet, ImageRecord


@dataclass(frozen=True)
class InstanceConfig:
    data_root: str
    output_dir: str = "artifacts/instance-fold-0"
    fold: int = 0
    n_folds: int = 5
    seed: int = 2026
    yolo_model: str = "yolo11s-seg.pt"
    image_size: int = 1024
    batch_size: int = 4
    epochs: int = 30
    patience: int = 5
    num_workers: int = 2
    mask_ratio: int = 2
    nms_iou: float = 0.3
    max_det: int = 100
    crop_size: int = 256
    crop_context: float = 1.8
    min_crop: int = 96
    refiner_epochs: int = 8
    refiner_batch_size: int = 16
    refiner_learning_rate: float = 0.0003

    def __post_init__(self) -> None:
        if not 0 <= self.fold < self.n_folds:
            raise ValueError("fold must be inside n_folds")
        positive = (
            self.image_size,
            self.batch_size,
            self.epochs,
            self.mask_ratio,
            self.crop_size,
            self.crop_context,
            self.min_crop,
            self.refiner_batch_size,
            self.refiner_learning_rate,
            self.max_det,
        )
        if min(positive) <= 0:
            raise ValueError("training sizes must be positive")
        if self.patience < 0 or self.num_workers < 0 or self.refiner_epochs < 0:
            raise ValueError("schedules must not be negative")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou must be between zero and one")


@dataclass(frozen=True)
class InstanceOutcome:
    official_pq: float
    self_evaluation_pq: float
    confidence_threshold: float
    mask_threshold: float
    min_area: int
    fp: int
    fn: int
    yolo_checkpoint: Path
    refiner_checkpoint: Path | None
    submission: Path


def select_complete_annotation_set(record: ImageRecord) -> AnnotationSet:
    if not record.annotation_sets:
        raise ValueError(f"{record.file_name} has no annotation sets")
    return max(
        record.annotation_sets,
        key=lambda item: (
            len(item.annotations),
            sum(float(annotation.get("area", 0)) for annotation in item.annotations),
            item.image_id,
        ),
    )


def yolo_label(annotation: Mapping[str, Any], width: int, height: int) -> str:
    polygons = annotation.get("segmentation", [])
    if len(polygons) != 1 or len(polygons[0]) < 6:
        raise ValueError("each annotation must contain one valid segmentation polygon")
    points = np.asarray(polygons[0], dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] / width, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / height, 0, 1)
    return "0 " + " ".join(f"{value:.6f}" for value in points.ravel())


def square_bounds(
    mask: np.ndarray, context: float = 1.8, minimum: int = 96
) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty mask")
    center_x = (xs.min() + xs.max() + 1) / 2
    center_y = (ys.min() + ys.max() + 1) / 2
    side = int(
        np.ceil(
            max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context
        )
    )
    side = min(max(side, minimum), height, width)
    left = min(max(int(round(center_x - side / 2)), 0), width - side)
    top = min(max(int(round(center_y - side / 2)), 0), height - side)
    return left, top, left + side, top + side


def prepare_yolo_dataset(
    records: Sequence[ImageRecord],
    folds: Mapping[str, int],
    validation_fold: int,
    image_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    image_dir, output_dir = Path(image_dir), Path(output_dir)
    for record in records:
        if record.file_name not in folds:
            raise ValueError(f"missing fold for {record.file_name}")
        split = "val" if folds[record.file_name] == validation_fold else "train"
        images = output_dir / "images" / split
        labels = output_dir / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        source = image_dir / record.file_name
        target = images / record.file_name
        if not target.exists():
            target.symlink_to(source.resolve())
        selected = select_complete_annotation_set(record)
        lines = [
            yolo_label(annotation, record.width, record.height)
            for annotation in selected.annotations
        ]
        (labels / f"{Path(record.file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    config = output_dir / "data.yaml"
    config.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: filament\n",
        encoding="utf-8",
    )
    return config


def threshold_instances(
    confidences: Sequence[float],
    yolo_masks: Sequence[np.ndarray],
    refined_probabilities: Sequence[np.ndarray],
    confidence_threshold: float,
    mask_threshold: float,
    min_area: int,
) -> list[np.ndarray]:
    if not (
        len(confidences) == len(yolo_masks) == len(refined_probabilities)
    ):
        raise ValueError("proposal values must have equal lengths")
    instances = []
    proposals = sorted(
        zip(confidences, yolo_masks, refined_probabilities),
        key=lambda item: item[0],
        reverse=True,
    )
    for confidence, yolo_mask, probability in proposals:
        if confidence < confidence_threshold:
            continue
        refined = (np.asarray(probability) >= mask_threshold).astype(np.uint8)
        if not refined.any():
            refined = np.asarray(yolo_mask, dtype=np.uint8)
        if int(refined.sum()) >= min_area:
            instances.append(refined)
    from .submission import remove_overlaps

    return remove_overlaps(instances, min_area=min_area)


def _records(config: InstanceConfig):
    from .data import assign_folds, build_manifest, load_coco

    root = Path(config.data_root)
    coco = load_coco(
        root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    )
    records = build_manifest(coco)
    folds = assign_folds(records, config.n_folds, config.seed)
    train = [record for record in records if folds[record.file_name] != config.fold]
    validation = [record for record in records if folds[record.file_name] == config.fold]
    return train, validation, folds, root / "train" / "train_images"


def _crop_arrays(
    records: Sequence[ImageRecord], image_dir: Path, config: InstanceConfig
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    from .masks import rasterize_instances

    images, masks = [], []
    for record in records:
        image = cv2.imread(str(image_dir / record.file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / record.file_name)
        selected = select_complete_annotation_set(record)
        for mask in rasterize_instances(
            selected.annotations, record.height, record.width
        ):
            if not mask.any():
                continue
            left, top, right, bottom = square_bounds(
                mask, config.crop_context, config.min_crop
            )
            images.append(
                cv2.resize(
                    image[top:bottom, left:right],
                    (config.crop_size, config.crop_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            masks.append(
                cv2.resize(
                    mask[top:bottom, left:right],
                    (config.crop_size, config.crop_size),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
    return np.stack(images), np.stack(masks)


def _build_refiner(pretrained: bool):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    class ConvBlock(nn.Sequential):
        def __init__(self, in_channels, out_channels):
            super().__init__(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    class CropUNet(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(
                weights=ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1, self.layer2 = encoder.layer1, encoder.layer2
            self.layer3, self.layer4 = encoder.layer3, encoder.layer4
            self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
            self.dec4 = ConvBlock(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.dec1 = ConvBlock(96, 32)
            self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
            self.head = nn.Conv2d(16, 1, 1)
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        def forward(self, image):
            stem = self.stem((image - self.mean) / self.std)
            one = self.layer1(self.pool(stem))
            two = self.layer2(one)
            three = self.layer3(two)
            value = self.layer4(three)
            value = self.dec4(torch.cat([self.up4(value), three], 1))
            value = self.dec3(torch.cat([self.up3(value), two], 1))
            value = self.dec2(torch.cat([self.up2(value), one], 1))
            value = self.dec1(torch.cat([self.up1(value), stem], 1))
            return self.head(self.up0(value))

    return CropUNet()


def _train_refiner(
    train_arrays,
    validation_arrays,
    config: InstanceConfig,
    output_path: Path,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class CropDataset(Dataset):
        def __init__(self, arrays, augment):
            self.images, self.masks = arrays
            self.augment = augment

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            image = torch.from_numpy(self.images[index].copy()).float() / 255
            mask = torch.from_numpy(self.masks[index].copy()).float().unsqueeze(0)
            if self.augment:
                rotations = random.randrange(4)
                image = torch.rot90(image, rotations, (0, 1))
                mask = torch.rot90(mask, rotations, (1, 2))
                if random.random() < 0.5:
                    image, mask = torch.flip(image, (1,)), torch.flip(mask, (2,))
                image = torch.clamp(
                    image ** random.uniform(0.85, 1.2) * random.uniform(0.9, 1.1),
                    0,
                    1,
                )
            return image.unsqueeze(0).repeat(3, 1, 1), mask

    train_loader = DataLoader(
        CropDataset(train_arrays, True),
        batch_size=config.refiner_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        CropDataset(validation_arrays, False),
        batch_size=config.refiner_batch_size * 2,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = _build_refiner(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.refiner_learning_rate, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.refiner_epochs
    )
    scaler = torch.amp.GradScaler("cuda")
    best_dice = -1.0
    for epoch in range(config.refiner_epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(images)
                probabilities = logits.sigmoid()
                intersection = (probabilities * masks).sum((1, 2, 3))
                denominator = probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
                loss = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                    logits, masks
                ) + 0.5 * dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        dice_sum = 0.0
        count = 0
        with torch.inference_mode():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device).bool()
                predictions = model(images).sigmoid() >= 0.65
                intersection = (predictions & masks).sum((1, 2, 3)).float()
                denominator = predictions.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                dice_sum += float(((2 * intersection + 1e-7) / (denominator + 1e-7)).sum())
                count += len(images)
        validation_dice = dice_sum / count
        print(json.dumps({"refiner_epoch": epoch + 1, "validation_dice": validation_dice}))
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save(model.state_dict(), output_path)
        scheduler.step()
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    return model.eval()


def _predict_proposals(yolo, refiner, image_path: Path, config: InstanceConfig, min_confidence: float):
    import cv2
    import torch

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    result = yolo.predict(
        str(image_path),
        imgsz=config.image_size,
        conf=min_confidence,
        iou=config.nms_iou,
        max_det=config.max_det,
        agnostic_nms=True,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.masks is None:
        return [], [], []
    yolo_masks = [mask.cpu().numpy().astype(np.uint8) for mask in result.masks.data]
    confidences = result.boxes.conf.cpu().tolist()
    if refiner is None:
        return confidences, yolo_masks, [mask.astype(np.float32) for mask in yolo_masks]
    probabilities = []
    device = next(refiner.parameters()).device
    with torch.inference_mode():
        for mask in yolo_masks:
            left, top, right, bottom = square_bounds(
                mask, config.crop_context, config.min_crop
            )
            crop = cv2.resize(
                image[top:bottom, left:right],
                (config.crop_size, config.crop_size),
                interpolation=cv2.INTER_LINEAR,
            )
            tensor = (
                torch.from_numpy(crop)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 3, 1, 1)
                .to(device)
                / 255
            )
            probability = refiner(tensor).sigmoid()[0, 0].cpu().numpy()
            probability = cv2.resize(
                probability,
                (right - left, bottom - top),
                interpolation=cv2.INTER_LINEAR,
            )
            canvas = np.zeros_like(mask, dtype=np.float32)
            canvas[top:bottom, left:right] = probability
            probabilities.append(canvas)
    return confidences, yolo_masks, probabilities


def _score_grid(
    yolo,
    refiner,
    records: Sequence[ImageRecord],
    image_dir: Path,
    config: InstanceConfig,
    confidence_thresholds: Sequence[float],
    mask_thresholds: Sequence[float],
    min_areas: Sequence[int],
):
    from .masks import rasterize_instances
    from .metrics import combine_image_scores, score_instances

    candidates = list(product(confidence_thresholds, mask_thresholds, min_areas))
    reports = {candidate: [] for candidate in candidates}
    floor = min(confidence_thresholds)
    for record in records:
        values = _predict_proposals(
            yolo, refiner, image_dir / record.file_name, config, floor
        )
        ground_truth_sets = [
            rasterize_instances(item.annotations, record.height, record.width)
            for item in record.annotation_sets
        ]
        for candidate in candidates:
            confidence, mask_threshold, min_area = candidate
            predictions = threshold_instances(
                *values, confidence, mask_threshold, min_area
            )
            reports[candidate].extend(
                score_instances(ground_truth, predictions)
                for ground_truth in ground_truth_sets
            )
    return {
        candidate: combine_image_scores(values)
        for candidate, values in reports.items()
    }


def _prediction_entries(
    yolo,
    refiner,
    records,
    image_dir,
    config,
    confidence,
    mask_threshold,
    min_area,
):
    from .masks import rasterize_instances

    for record in records:
        values = _predict_proposals(
            yolo, refiner, image_dir / record.file_name, config, confidence
        )
        predictions = threshold_instances(
            *values, confidence, mask_threshold, min_area
        )
        for annotation_set in record.annotation_sets:
            yield (
                rasterize_instances(
                    annotation_set.annotations, record.height, record.width
                ),
                predictions,
            )


def _infer_test(
    yolo,
    refiner,
    config,
    confidence,
    mask_threshold,
    min_area,
    output_path,
):
    from .submission import build_submission_rows, validate_submission, write_submission

    image_dir = Path(config.data_root) / "test" / "test_images"
    rows = []
    paths = sorted(image_dir.glob("*.jpeg"))
    for image_path in paths:
        values = _predict_proposals(
            yolo, refiner, image_path, config, confidence
        )
        instances = threshold_instances(
            *values, confidence, mask_threshold, min_area
        )
        rows.extend(build_submission_rows({image_path.stem: instances}))
    write_submission(output_path, rows)
    report = validate_submission(
        output_path, expected_stems={path.stem for path in paths}
    )
    if report.errors:
        raise ValueError("invalid submission: " + "; ".join(report.errors))


def run_instance_experiment(
    config: InstanceConfig,
    confidence_thresholds: Sequence[float] = (0.1, 0.15, 0.25, 0.35),
    mask_thresholds: Sequence[float] = (0.55, 0.65, 0.75),
    min_areas: Sequence[int] = (20, 50, 100),
) -> InstanceOutcome:
    import torch
    from ultralytics import YOLO

    from .self_evaluation import organizer_self_evaluation

    if not torch.cuda.is_available():
        raise RuntimeError("instance training requires a CUDA GPU")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    train_records, validation_records, folds, image_dir = _records(config)
    data_yaml = prepare_yolo_dataset(
        train_records + validation_records,
        folds,
        config.fold,
        image_dir,
        output_dir / "yolo-data",
    )
    yolo = YOLO(config.yolo_model)
    yolo.train(
        data=str(data_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        workers=config.num_workers,
        device=0,
        patience=config.patience,
        seed=config.seed,
        single_cls=True,
        mask_ratio=config.mask_ratio,
        overlap_mask=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=10,
        scale=0.3,
        mosaic=0.3,
        close_mosaic=5,
        project=str(output_dir / "yolo-runs"),
        name="instance",
    )
    yolo_checkpoint = output_dir / "best-yolo.pt"
    shutil.copy2(yolo.trainer.best, yolo_checkpoint)
    yolo = YOLO(str(yolo_checkpoint))
    refiner_checkpoint = None
    refiner = None
    if config.refiner_epochs:
        refiner_checkpoint = output_dir / "best-refiner.pt"
        refiner = _train_refiner(
            _crop_arrays(train_records, image_dir, config),
            _crop_arrays(validation_records, image_dir, config),
            config,
            refiner_checkpoint,
        )
    scores = _score_grid(
        yolo,
        refiner,
        validation_records,
        image_dir,
        config,
        confidence_thresholds,
        mask_thresholds,
        min_areas,
    )
    best, report = max(
        scores.items(),
        key=lambda item: (item[1].official_pq, -item[1].fp, -item[1].fn),
    )
    confidence, mask_threshold, min_area = best
    self_evaluation_pq = organizer_self_evaluation(
        _prediction_entries(
            yolo,
            refiner,
            validation_records,
            image_dir,
            config,
            confidence,
            mask_threshold,
            min_area,
        )
    )
    submission = output_dir / "submission.csv"
    _infer_test(
        yolo,
        refiner,
        config,
        confidence,
        mask_threshold,
        min_area,
        submission,
    )
    outcome = InstanceOutcome(
        report.official_pq,
        self_evaluation_pq,
        confidence,
        mask_threshold,
        min_area,
        report.fp,
        report.fn,
        yolo_checkpoint,
        refiner_checkpoint,
        submission,
    )
    (output_dir / "outcome.json").write_text(
        json.dumps(
            {
                **asdict(outcome),
                "yolo_checkpoint": str(yolo_checkpoint),
                "refiner_checkpoint": str(refiner_checkpoint) if refiner_checkpoint else None,
                "submission": str(submission),
                "grid": [
                    {
                        "confidence_threshold": candidate[0],
                        "mask_threshold": candidate[1],
                        "min_area": candidate[2],
                        "official_pq": value.official_pq,
                        "fp": value.fp,
                        "fn": value.fn,
                    }
                    for candidate, value in scores.items()
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outcome
