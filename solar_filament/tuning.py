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


def choose_best_postprocessing(candidates: Sequence[PostprocessingScore]) -> PostprocessingScore:
    if not candidates:
        raise ValueError("at least one post-processing candidate is required")
    return max(candidates, key=lambda item: (item.official_pq, -item.fp, -item.fn))


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
) -> TuningOutcome:
    import torch

    from .inference import load_model
    from .self_evaluation import organizer_self_evaluation
    from .training import TrainConfig, _records, evaluate_model, prediction_entries

    pairs = list(product(thresholds, min_areas))
    if not pairs:
        raise ValueError("at least one threshold and min_area are required")
    if any(not 0 <= threshold <= 1 or min_area < 1 for threshold, min_area in pairs):
        raise ValueError("thresholds must be in [0, 1] and min_areas must be positive")

    model, saved_config, device = load_model(checkpoint_path)
    base_config = TrainConfig(**saved_config)
    _, validation_records, image_dir = _records(base_config)
    scores = []
    for threshold, min_area in pairs:
        candidate_config = replace(base_config, threshold=float(threshold), min_area=int(min_area))
        report = evaluate_model(model, validation_records, image_dir, candidate_config, device)
        scores.append(
            PostprocessingScore(
                threshold=float(threshold),
                min_area=int(min_area),
                official_pq=report.official_pq,
                fp=report.fp,
                fn=report.fn,
            )
        )

    best = choose_best_postprocessing(scores)
    tuned_config = replace(base_config, threshold=best.threshold, min_area=best.min_area)
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
