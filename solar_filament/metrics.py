from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ImageScore:
    pq: float
    tp: int
    fp: int
    fn: int
    matched_ious: tuple[float, ...]
    positive_ious: tuple[float, ...]
    positive_dice: tuple[float, ...]
    one_to_many: int
    many_to_one: int


@dataclass(frozen=True)
class EvaluationScore:
    official_pq: float
    macro_pq: float
    tp: int
    fp: int
    fn: int
    mean_positive_iou: float
    mean_positive_dice: float
    one_to_many: int
    many_to_one: int


def _as_pixel_set(mask: Any) -> set[int]:
    if isinstance(mask, (set, frozenset)):
        return set(mask)
    try:
        import numpy as np

        array = np.asarray(mask)
        return set(np.flatnonzero(array).tolist())
    except ImportError:
        if hasattr(mask, "getdata"):
            return {index for index, value in enumerate(mask.getdata()) if value}
        return set(mask)


def score_instances(
    ground_truth: Sequence[Any], predictions: Sequence[Any], iou_threshold: float = 0.5
) -> ImageScore:
    gt_sets = [_as_pixel_set(mask) for mask in ground_truth]
    pred_sets = [_as_pixel_set(mask) for mask in predictions]
    ious: list[list[float]] = []
    dice: list[list[float]] = []
    for gt in gt_sets:
        iou_row: list[float] = []
        dice_row: list[float] = []
        for pred in pred_sets:
            intersection = len(gt & pred)
            union = len(gt | pred)
            iou_row.append(intersection / union if union else 0.0)
            total = len(gt) + len(pred)
            dice_row.append(2 * intersection / total if total else 0.0)
        ious.append(iou_row)
        dice.append(dice_row)

    hits = [[value > iou_threshold for value in row] for row in ious]
    matched = tuple(
        ious[gt_index][pred_index]
        for gt_index in range(len(gt_sets))
        for pred_index in range(len(pred_sets))
        if hits[gt_index][pred_index]
    )
    fp = sum(
        not any(hits[gt_index][pred_index] for gt_index in range(len(gt_sets)))
        for pred_index in range(len(pred_sets))
    )
    fn = sum(not any(row) for row in hits)
    denominator = len(matched) + 0.5 * fp + 0.5 * fn
    positive_ious = tuple(value for row in ious for value in row if value > 0)
    positive_dice = tuple(value for row in dice for value in row if value > 0)
    one_to_many = sum(sum(value > 0 for value in row) > 1 for row in ious)
    many_to_one = sum(
        sum(ious[gt_index][pred_index] > 0 for gt_index in range(len(gt_sets))) > 1
        for pred_index in range(len(pred_sets))
    )
    return ImageScore(
        pq=sum(matched) / denominator if denominator else 0.0,
        tp=len(matched),
        fp=fp,
        fn=fn,
        matched_ious=matched,
        positive_ious=positive_ious,
        positive_dice=positive_dice,
        one_to_many=one_to_many,
        many_to_one=many_to_one,
    )


def evaluate_annotation_sets(
    entries: Iterable[tuple[Sequence[Any], Sequence[Any]]]
) -> EvaluationScore:
    reports: list[ImageScore] = []
    matched: list[float] = []
    positive_ious: list[float] = []
    positive_dice: list[float] = []
    for ground_truth, predictions in entries:
        report = score_instances(ground_truth, predictions)
        reports.append(report)
        matched.extend(report.matched_ious)
        positive_ious.extend(report.positive_ious)
        positive_dice.extend(report.positive_dice)
    tp = sum(report.tp for report in reports)
    fp = sum(report.fp for report in reports)
    fn = sum(report.fn for report in reports)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return EvaluationScore(
        official_pq=sum(matched) / denominator if denominator else 0.0,
        macro_pq=fmean(report.pq for report in reports) if reports else 0.0,
        tp=tp,
        fp=fp,
        fn=fn,
        mean_positive_iou=fmean(positive_ious) if positive_ious else 0.0,
        mean_positive_dice=fmean(positive_dice) if positive_dice else 0.0,
        one_to_many=sum(report.one_to_many for report in reports),
        many_to_one=sum(report.many_to_one for report in reports),
    )
