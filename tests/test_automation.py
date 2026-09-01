import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from solar_filament.automation import (
    AutomationConfig,
    CandidateManifest,
    Experiment,
    decide_candidate,
    run_automation,
)
from solar_filament.cli import main
from solar_filament.metrics import evaluate_annotation_sets
from solar_filament.self_evaluation import organizer_self_evaluation
from solar_filament.tuning import PostprocessingScore, choose_best_postprocessing


def candidate(**overrides):
    values = {
        "revision": "abc123",
        "experiment": "threshold-055-area-64",
        "training_config": {"fold": 0},
        "threshold": 0.55,
        "min_area": 64,
        "internal_pq": 0.16,
        "self_evaluation_pq": 0.16,
        "checkpoint": "artifacts/best.pt",
        "submission": "artifacts/submission.csv",
    }
    values.update(overrides)
    return CandidateManifest.from_dict(values)


class CandidateGateTests(unittest.TestCase):
    def test_accepts_exact_minimum_improvement_when_evaluators_agree(self):
        decision = decide_candidate(candidate(), best_pq=0.15, min_delta=0.01)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "validation_improved")

    def test_rejects_candidate_when_organizer_evaluator_disagrees(self):
        decision = decide_candidate(
            candidate(self_evaluation_pq=0.159),
            best_pq=0.15,
            min_delta=0.001,
            parity_tolerance=1e-6,
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "self_evaluation_mismatch")

    def test_rejects_candidate_below_minimum_improvement(self):
        decision = decide_candidate(candidate(internal_pq=0.159, self_evaluation_pq=0.159), 0.15, 0.01)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "validation_not_improved")

    def test_rejects_invalid_manifest_at_boundary(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            candidate(threshold=1.01)
        with self.assertRaisesRegex(ValueError, "min_area"):
            candidate(min_area=0)
        with self.assertRaisesRegex(ValueError, "internal_pq"):
            candidate(internal_pq=-0.01)

    def test_internal_metric_matches_organizer_self_evaluation(self):
        import numpy as np

        def mask(*pixels):
            value = np.zeros((2, 4), dtype=np.uint8)
            value.flat[list(pixels)] = 1
            return value

        entries = [
            ([mask(0, 1)], [mask(0, 1, 2, 3)]),  # exact IoU 0.5: not a TP
            ([mask(0, 1, 2, 3)], [mask(0, 1, 2), mask(3)]),
            ([mask(4, 5)], []),
        ]

        internal = evaluate_annotation_sets(entries).official_pq
        organizer = organizer_self_evaluation(entries)

        self.assertAlmostEqual(internal, organizer, places=12)

    def test_tuning_prefers_higher_pq_then_fewer_false_positives(self):
        best = choose_best_postprocessing(
            [
                PostprocessingScore(0.50, 32, 0.16, fp=900, fn=500),
                PostprocessingScore(0.55, 64, 0.17, fp=800, fn=510),
                PostprocessingScore(0.60, 64, 0.17, fp=700, fn=520),
            ]
        )

        self.assertEqual((best.threshold, best.min_area), (0.60, 64))

    def test_tuning_rejects_empty_candidate_grid(self):
        with self.assertRaisesRegex(ValueError, "candidate"):
            choose_best_postprocessing([])


class RecordingRunner:
    def __init__(self, manifest):
        self.calls = []
        self.manifest = manifest

    def __call__(self, command, cwd=None):
        self.calls.append(command)
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "abc123\n"
        if command[:3] == ["git", "rev-parse", "origin/main"]:
            return "abc123\n"
        if command[:3] == ["kaggle", "kernels", "status"]:
            return "complete\n"
        if command[:3] == ["kaggle", "kernels", "output"]:
            output = Path(command[command.index("-p") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "candidate.json").write_text(json.dumps(self.manifest))
            (output / "submission.csv").write_text("filament_id,segmentation_rle\n")
            (output / "best.pt").write_bytes(b"checkpoint")
            return "downloaded\n"
        if command[:3] == ["kaggle", "competitions", "submissions"]:
            return json.dumps([{"description": "auto:threshold-055-area-64:abc123", "status": "complete", "publicScore": "0.17"}])
        if command[:3] == ["kaggle", "competitions", "leaderboard"]:
            return json.dumps([{"teamName": "TAI TRAN DANG", "score": "0.17", "rank": 300}])
        return "ok\n"


class AutomationLoopTests(unittest.TestCase):
    def make_config(self, root, execute=True):
        kernel_dir = root / "kernel"
        kernel_dir.mkdir()
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({"code_file": "experiment.ipynb"}))
        (kernel_dir / "experiment.ipynb").write_text(
            json.dumps({"cells": [{"source": ["__EXPERIMENT_CONFIG__"]}]})
        )
        return AutomationConfig(
            kernel="taitrandang/solar-filament-automation",
            competition="filament-segmentation-2026",
            kernel_dir=kernel_dir,
            work_dir=root / "runs",
            execute=execute,
            max_runs=1,
            max_submissions=1,
            best_pq=0.15,
            min_delta=0.01,
            poll_seconds=0,
        )

    def test_dry_run_plans_without_calling_external_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = RecordingRunner({})
            result = run_automation(
                self.make_config(Path(directory), execute=False),
                [Experiment("threshold-055-area-64", {"threshold": 0.55, "min_area": 64})],
                runner=runner,
            )

        self.assertEqual(runner.calls, [])
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["planned_runs"], 1)

    def test_submits_improved_parity_checked_candidate_and_records_leaderboard(self):
        manifest = candidate().to_dict()
        with tempfile.TemporaryDirectory() as directory:
            runner = RecordingRunner(manifest)
            result = run_automation(
                self.make_config(Path(directory)),
                [Experiment("threshold-055-area-64", {"threshold": 0.55, "min_area": 64})],
                runner=runner,
                sleep=lambda _: None,
            )

        commands = [" ".join(call) for call in runner.calls]
        self.assertLess(commands.index("git push origin main"), next(i for i, value in enumerate(commands) if "kernels push" in value))
        output_command = next(call for call in runner.calls if call[:3] == ["kaggle", "kernels", "output"])
        self.assertIn("--page-size", output_command)
        self.assertEqual(output_command[output_command.index("--page-size") + 1], "200")
        self.assertEqual(sum("competitions submit" in value for value in commands), 1)
        self.assertEqual(result["best_validation_pq"], 0.16)
        self.assertEqual(result["latest_public_score"], 0.17)

    def test_never_dispatches_beyond_run_or_submission_limits(self):
        manifest = candidate().to_dict()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config = AutomationConfig(**{**config.__dict__, "max_runs": 2})
            runner = RecordingRunner(manifest)
            result = run_automation(
                config,
                [Experiment("threshold-055-area-64", {}), Experiment("second", {})],
                runner=runner,
                sleep=lambda _: None,
            )

        self.assertEqual(sum(call[:3] == ["kaggle", "kernels", "push"] for call in runner.calls), 1)
        self.assertEqual(result["runs"], 1)
        self.assertEqual(result["submissions"], 1)

    def test_does_not_submit_candidate_without_required_improvement(self):
        manifest = candidate(internal_pq=0.159, self_evaluation_pq=0.159).to_dict()
        with tempfile.TemporaryDirectory() as directory:
            runner = RecordingRunner(manifest)
            result = run_automation(
                self.make_config(Path(directory)),
                [Experiment("threshold-055-area-64", {})],
                runner=runner,
                sleep=lambda _: None,
            )

        self.assertFalse(any(call[:3] == ["kaggle", "competitions", "submit"] for call in runner.calls))
        self.assertEqual(result["history"][0]["decision"], "validation_not_improved")

    def test_refuses_to_overwrite_an_unfinished_remote_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config.work_dir.mkdir(parents=True)
            (config.work_dir / "state.json").write_text(json.dumps({"status": "kernel_dispatched"}))
            runner = RecordingRunner(candidate().to_dict())

            with self.assertRaisesRegex(RuntimeError, "reconcile"):
                run_automation(
                    config,
                    [Experiment("threshold-055-area-64", {})],
                    runner=runner,
                )

        self.assertEqual(runner.calls, [])

    def test_cli_is_dry_run_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = root / "experiments.json"
            experiments.write_text(json.dumps([{"name": "tune", "thresholds": [0.5]}]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "automate",
                        "--kernel",
                        "taitrandang/solar-filament-automation",
                        "--kernel-dir",
                        str(root),
                        "--experiments",
                        str(experiments),
                        "--work-dir",
                        str(root / "runs"),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "dry_run")

    def test_kaggle_notebook_emits_the_controller_candidate_contract(self):
        root = Path(__file__).parents[1]
        notebook = json.loads(
            (root / "automation/kaggle/automated-experiment.ipynb").read_text()
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        metadata = json.loads((root / "automation/kaggle/kernel-metadata.json").read_text())

        self.assertIn("tune_checkpoint", source)
        self.assertIn("organizer", source.lower())
        self.assertIn("candidate.json", source)
        self.assertIn("infer_directory", source)
        self.assertTrue(metadata["enable_gpu"])
        self.assertIn("filament-segmentation-2026", metadata["competition_sources"])


if __name__ == "__main__":
    unittest.main()
