from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PostprocessingScore:
    threshold: float
    min_area: int
    official_pq: float
    fp: int
    fn: int
    close_kernel: int = 0


def choose_best_postprocessing(candidates: Sequence[PostprocessingScore]) -> PostprocessingScore:
    if not candidates:
        raise ValueError("at least one post-processing candidate is required")
    return max(candidates, key=lambda item: (item.official_pq, -item.fp, -item.fn))


def score_postprocessing_grid(
    probability_entries,
    thresholds: Sequence[float],
    min_areas: Sequence[int],
    close_kernels: Sequence[int] = (0,),
) -> list[PostprocessingScore]:
    from .masks import connected_components
    from .metrics import combine_image_scores, score_instances

    candidates = list(product(thresholds, min_areas, close_kernels))
    if not candidates:
        raise ValueError("at least one post-processing candidate is required")
    reports = {candidate: [] for candidate in candidates}
    for ground_truth_sets, probabilities in probability_entries:
        for candidate in candidates:
            threshold, min_area, close_kernel = candidate
            predictions = connected_components(
                probabilities,
                threshold=float(threshold),
                min_area=int(min_area),
                close_kernel=int(close_kernel),
            )
            reports[candidate].extend(
                score_instances(ground_truth, predictions)
                for ground_truth in ground_truth_sets
            )
    scores = []
    for threshold, min_area, close_kernel in candidates:
        report = combine_image_scores(reports[(threshold, min_area, close_kernel)])
        scores.append(
            PostprocessingScore(
                threshold=float(threshold),
                min_area=int(min_area),
                official_pq=report.official_pq,
                fp=report.fp,
                fn=report.fn,
                close_kernel=int(close_kernel),
            )
        )
    return scores


@dataclass(frozen=True)
class TuningOutcome:
    best: PostprocessingScore
    self_evaluation_pq: float
    tuned_checkpoint: Path
    scores: tuple[PostprocessingScore, ...]

    def to_dict(self):
        return {
            "best": asdict(self.best),
            "self_evaluation_pq": self.self_evaluation_pq,
            "tuned_checkpoint": str(self.tuned_checkpoint),
            "scores": [asdict(score) for score in self.scores],
        }


def tune_checkpoint(
    checkpoint_path: Path | str,
    thresholds: Sequence[float],
    min_areas: Sequence[int],
    output_path: Path | str,
    close_kernels: Sequence[int] = (0,),
) -> TuningOutcome:
    import torch

    from .inference import load_model, predict_probability
    from .masks import rasterize_instances
    from .self_evaluation import organizer_self_evaluation
    from .training import TrainConfig, _records, prediction_entries

    candidates = list(product(thresholds, min_areas, close_kernels))
    if not candidates:
        raise ValueError("at least one post-processing candidate is required")
    if any(
        not 0 <= threshold <= 1 or min_area < 1
        for threshold, min_area, _ in candidates
    ):
        raise ValueError("thresholds must be in [0, 1] and min_areas must be positive")

    model, saved_config, device = load_model(checkpoint_path)
    base_config = TrainConfig(**saved_config)
    _, validation_records, image_dir = _records(base_config)
    def probability_entries():
        for record in validation_records:
            probability = predict_probability(
                model,
                image_dir / record.file_name,
                base_config.image_size,
                device,
                native=base_config.model_name == "native_unet",
                tta=base_config.tta,
            )
            ground_truth_sets = [
                rasterize_instances(
                    item.annotations, height=record.height, width=record.width
                )
                for item in record.annotation_sets
            ]
            yield ground_truth_sets, probability

    scores = score_postprocessing_grid(
        probability_entries(), thresholds, min_areas, close_kernels
    )

    best = choose_best_postprocessing(scores)
    tuned_config = replace(
        base_config,
        threshold=best.threshold,
        min_area=best.min_area,
        close_kernel=best.close_kernel,
    )
    self_evaluation_pq = organizer_self_evaluation(
        prediction_entries(model, validation_records, image_dir, tuned_config, device)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["config"] = asdict(tuned_config)
    checkpoint["tuning"] = {
        "internal_pq": best.official_pq,
        "self_evaluation_pq": self_evaluation_pq,
        "scores": [asdict(score) for score in scores],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return TuningOutcome(best, self_evaluation_pq, output_path, tuple(scores))
