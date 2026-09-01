from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .data import assign_folds, audit_dataset, build_manifest, load_coco, save_folds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solar-filament")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="validate competition files and COCO schema")
    audit.add_argument("data_root")
    audit.add_argument("--output", default="artifacts/data-audit.json")

    folds = commands.add_parser("folds", help="create leakage-safe grouped folds")
    folds.add_argument("data_root")
    folds.add_argument("--output", default="artifacts/folds.json")
    folds.add_argument("--n-folds", type=int, default=5)
    folds.add_argument("--seed", type=int, default=2026)

    train_parser = commands.add_parser("train", help="train one semantic segmentation fold")
    train_parser.add_argument("data_root")
    train_parser.add_argument("--output-dir", default="artifacts/fold-0")
    train_parser.add_argument("--fold", type=int, default=0)
    train_parser.add_argument("--n-folds", type=int, default=5)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--image-size", type=int, default=768)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=0.0002)
    train_parser.add_argument("--positive-weight", type=float, default=12.0)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--min-area", type=int, default=32)
    train_parser.add_argument(
        "--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True
    )

    infer = commands.add_parser("infer", help="create and validate submission CSV")
    infer.add_argument("checkpoint")
    infer.add_argument("image_dir")
    infer.add_argument("output_csv")
    infer.add_argument("--run-manifest")

    validate = commands.add_parser("validate-submission", help="decode-audit a CSV")
    validate.add_argument("submission")
    validate.add_argument("--image-dir")

    automate = commands.add_parser("automate", help="run a bounded GitHub/Kaggle experiment loop")
    automate.add_argument("--kernel", required=True)
    automate.add_argument("--competition", default="filament-segmentation-2026")
    automate.add_argument("--kernel-dir", default="automation/kaggle")
    automate.add_argument("--experiments", default="automation/experiments.json")
    automate.add_argument("--work-dir", default="artifacts/automation")
    automate.add_argument("--best-pq", type=float, default=0.1484172851438538)
    automate.add_argument("--min-delta", type=float, default=0.001)
    automate.add_argument("--parity-tolerance", type=float, default=1e-6)
    automate.add_argument("--max-runs", type=int, default=1)
    automate.add_argument("--max-submissions", type=int, default=1)
    automate.add_argument("--poll-seconds", type=float, default=30)
    automate.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        report = audit_dataset(args.data_root)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report.as_dict(), indent=2))
        return 1 if report.errors else 0

    if args.command == "folds":
        root = Path(args.data_root)
        coco = load_coco(
            root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        )
        folds = assign_folds(build_manifest(coco), args.n_folds, args.seed)
        save_folds(args.output, folds)
        print(f"wrote {len(folds)} grouped assignments to {args.output}")
        return 0

    if args.command == "train":
        from .training import TrainConfig, train

        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            fold=args.fold,
            n_folds=args.n_folds,
            seed=args.seed,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            positive_weight=args.positive_weight,
            threshold=args.threshold,
            min_area=args.min_area,
            pretrained_backbone=args.pretrained_backbone,
        )
        print(train(config))
        return 0

    if args.command == "infer":
        from .inference import infer_directory

        report = infer_directory(
            args.checkpoint, args.image_dir, args.output_csv, args.run_manifest
        )
        print(json.dumps(asdict(report), indent=2))
        return 0

    if args.command == "validate-submission":
        from .submission import validate_submission

        expected = None
        if args.image_dir:
            expected = {path.stem for path in Path(args.image_dir).glob("*.jpeg")}
        report = validate_submission(args.submission, expected_stems=expected)
        print(json.dumps(asdict(report), indent=2))
        return 1 if report.errors else 0
    if args.command == "automate":
        from .automation import AutomationConfig, load_experiments, run_automation

        result = run_automation(
            AutomationConfig(
                kernel=args.kernel,
                competition=args.competition,
                kernel_dir=Path(args.kernel_dir),
                work_dir=Path(args.work_dir),
                execute=args.execute,
                max_runs=args.max_runs,
                max_submissions=args.max_submissions,
                best_pq=args.best_pq,
                min_delta=args.min_delta,
                parity_tolerance=args.parity_tolerance,
                poll_seconds=args.poll_seconds,
            ),
            load_experiments(args.experiments),
        )
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
