from __future__ import annotations

from typing import Any, Iterable, Sequence


def organizer_self_evaluation(
    entries: Iterable[tuple[Sequence[Any], Sequence[Any]]],
    iou_threshold: float = 0.5,
) -> float:
    """Run the PQ calculation from the organizer's Self_Evaluation_Notebook v6.

    Source snapshot: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
    """
    import numpy as np
    import torch

    matched_ious: list[float] = []
    fp = 0
    fn = 0
    for ground_truth, predictions in entries:
        if not ground_truth:
            fp += len(predictions)
            continue
        if not predictions:
            fn += len(ground_truth)
            continue
        gt_layers = torch.from_numpy(np.stack(ground_truth).astype(np.float32))
        pred_layers = torch.from_numpy(np.stack(predictions).astype(np.float32))
        gt_flat = gt_layers.reshape(len(ground_truth), -1)
        pred_flat = pred_layers.reshape(len(predictions), -1)
        intersection = torch.matmul(gt_flat, pred_flat.t())
        gt_areas = gt_flat.sum(dim=1).view(-1, 1)
        pred_areas = pred_flat.sum(dim=1).view(1, -1)
        union = gt_areas + pred_areas - intersection
        iou = torch.where(union == 0, torch.tensor(0.0), intersection / union)
        hits = iou > iou_threshold
        matched_ious.extend(iou[hits].tolist())
        fp += int((hits.sum(dim=0) == 0).sum().item())
        fn += int((hits.sum(dim=1) == 0).sum().item())
    denominator = len(matched_ious) + 0.5 * fp + 0.5 * fn
    return sum(matched_ious) / denominator if denominator else 0.0
