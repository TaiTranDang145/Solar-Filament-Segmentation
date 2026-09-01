from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CandidateManifest:
    revision: str
    experiment: str
    training_config: Mapping[str, Any]
    threshold: float
    min_area: int
    internal_pq: float
    self_evaluation_pq: float
    checkpoint: str
    submission: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "experiment": self.experiment,
            "training_config": dict(self.training_config),
            "threshold": self.threshold,
            "min_area": self.min_area,
            "internal_pq": self.internal_pq,
            "self_evaluation_pq": self.self_evaluation_pq,
            "checkpoint": self.checkpoint,
            "submission": self.submission,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CandidateManifest":
        try:
            candidate = cls(
                revision=str(values["revision"]),
                experiment=str(values["experiment"]),
                training_config=dict(values["training_config"]),
                threshold=float(values["threshold"]),
                min_area=int(values["min_area"]),
                internal_pq=float(values["internal_pq"]),
                self_evaluation_pq=float(values["self_evaluation_pq"]),
                checkpoint=str(values["checkpoint"]),
                submission=str(values["submission"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate manifest: {exc}") from exc
        if not 0 <= candidate.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if candidate.min_area < 1:
            raise ValueError("min_area must be positive")
        for name in ("internal_pq", "self_evaluation_pq"):
            if not 0 <= getattr(candidate, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not all((candidate.revision, candidate.experiment, candidate.checkpoint, candidate.submission)):
            raise ValueError("candidate manifest strings must not be empty")
        for name in ("checkpoint", "submission"):
            path = Path(getattr(candidate, name))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a safe relative path")
        return candidate


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reason: str


def decide_candidate(
    candidate: CandidateManifest,
    best_pq: float,
    min_delta: float,
    parity_tolerance: float = 1e-9,
) -> CandidateDecision:
    if min_delta < 0 or parity_tolerance < 0:
        raise ValueError("min_delta and parity_tolerance must not be negative")
    if abs(candidate.internal_pq - candidate.self_evaluation_pq) > parity_tolerance:
        return CandidateDecision(False, "self_evaluation_mismatch")
    if candidate.internal_pq < best_pq + min_delta:
        return CandidateDecision(False, "validation_not_improved")
    return CandidateDecision(True, "validation_improved")


@dataclass(frozen=True)
class Experiment:
    name: str
    config: Mapping[str, Any]


def load_experiments(path: Path | str) -> list[Experiment]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("experiments file must contain a non-empty JSON list")
    experiments = []
    for value in values:
        if not isinstance(value, dict) or not str(value.get("name", "")).strip():
            raise ValueError("every experiment must have a non-empty name")
        experiments.append(
            Experiment(str(value["name"]), {key: item for key, item in value.items() if key != "name"})
        )
    if len({item.name for item in experiments}) != len(experiments):
        raise ValueError("experiment names must be unique")
    return experiments


@dataclass(frozen=True)
class AutomationConfig:
    kernel: str
    competition: str
    kernel_dir: Path
    work_dir: Path
    execute: bool = False
    max_runs: int = 1
    max_submissions: int = 1
    best_pq: float = 0.0
    min_delta: float = 0.001
    parity_tolerance: float = 1e-6
    poll_seconds: float = 30
    poll_attempts: int = 240
    repo_root: Path = Path.cwd()

    def __post_init__(self) -> None:
        if self.max_runs < 1 or self.max_submissions < 0:
            raise ValueError("max_runs must be positive and max_submissions must not be negative")
        if not 0 <= self.best_pq <= 1 or self.min_delta <= 0:
            raise ValueError("best_pq must be in [0, 1] and min_delta must be positive")
        if self.poll_seconds < 0 or self.poll_attempts < 1:
            raise ValueError("poll settings must not be negative")


def _default_runner(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _wait_for_kernel(kernel: str, config: AutomationConfig, runner, sleep) -> None:
    for _ in range(config.poll_attempts):
        status = runner(["kaggle", "kernels", "status", kernel]).strip().lower()
        if "complete" in status:
            return
        if any(value in status for value in ("error", "failed", "cancel")):
            raise RuntimeError(f"Kaggle kernel failed: {status}")
        sleep(config.poll_seconds)
    raise TimeoutError(f"Kaggle kernel did not finish after {config.poll_attempts} polls")


def _artifact(run_dir: Path, relative_name: str) -> Path:
    direct = run_dir / relative_name
    if direct.is_file():
        return direct
    matches = list(run_dir.rglob(Path(relative_name).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {relative_name} in {run_dir}, found {len(matches)}")
    return matches[0]


def _wait_for_submission(message: str, config: AutomationConfig, runner, sleep):
    command = [
        "kaggle",
        "competitions",
        "submissions",
        config.competition,
        "--page-size",
        "100",
        "--format",
        "json",
    ]
    for _ in range(config.poll_attempts):
        rows = json.loads(runner(command))
        matched = next((row for row in rows if row.get("description") == message), None)
        if matched:
            status = str(matched.get("status", "")).lower()
            if any(value in status for value in ("error", "failed", "invalid")):
                raise RuntimeError(f"Kaggle submission failed: {status}")
            if matched.get("publicScore") not in (None, ""):
                return matched
        sleep(config.poll_seconds)
    raise TimeoutError(f"Kaggle submission was not scored after {config.poll_attempts} polls")


def _stage_kernel(source: Path, destination: Path, experiment: Experiment) -> None:
    shutil.copytree(source, destination)
    metadata = json.loads((destination / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook_path = destination / metadata["code_file"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    payload = json.dumps({"name": experiment.name, **dict(experiment.config)})
    replaced = False
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if "__EXPERIMENT_CONFIG__" in line:
                source[index] = line.replace("__EXPERIMENT_CONFIG__", payload)
                replaced = True
    if not replaced:
        raise ValueError("Kaggle notebook is missing __EXPERIMENT_CONFIG__")
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def run_automation(
    config: AutomationConfig,
    experiments: list[Experiment],
    runner=_default_runner,
    sleep=time.sleep,
) -> dict[str, Any]:
    planned = min(config.max_runs, len(experiments))
    if not config.execute:
        return {
            "status": "dry_run",
            "planned_runs": planned,
            "planned_experiments": [item.name for item in experiments[:planned]],
        }

    state_path = config.work_dir / "state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("status") not in ("complete", "finished"):
            raise RuntimeError(
                f"unfinished automation state ({previous.get('status')}); reconcile Kaggle before retrying"
            )
    if runner(["git", "status", "--porcelain"], cwd=config.repo_root).strip():
        raise RuntimeError("Git worktree must be clean before automation")
    revision = runner(["git", "rev-parse", "HEAD"], cwd=config.repo_root).strip()
    remote_revision = runner(["git", "rev-parse", "origin/main"], cwd=config.repo_root).strip()
    if revision != remote_revision:
        raise RuntimeError("HEAD must match origin/main before automation")
    runner(["kaggle", "kernels", "list", "--mine", "--page-size", "1", "--format", "json"])
    runner(["git", "push", "origin", "main"], cwd=config.repo_root)

    state: dict[str, Any] = {
        "status": "running",
        "revision": revision,
        "runs": 0,
        "submissions": 0,
        "best_validation_pq": config.best_pq,
        "latest_public_score": None,
        "history": [],
    }
    _write_state(state_path, state)

    for experiment in experiments[:planned]:
        if state["submissions"] >= config.max_submissions:
            break
        with tempfile.TemporaryDirectory(prefix="solar-filament-kernel-") as directory:
            staged = Path(directory) / "kernel"
            _stage_kernel(config.kernel_dir, staged, experiment)
            state["runs"] += 1
            state["current_experiment"] = experiment.name
            state["status"] = "dispatching_kernel"
            _write_state(state_path, state)
            runner(["kaggle", "kernels", "push", "-p", str(staged)])

        state["status"] = "kernel_dispatched"
        _write_state(state_path, state)
        _wait_for_kernel(config.kernel, config, runner, sleep)

        run_dir = config.work_dir / f"run-{state['runs']:03d}-{experiment.name}"
        runner(
            [
                "kaggle",
                "kernels",
                "output",
                config.kernel,
                "-p",
                str(run_dir),
                "--force",
                "--page-size",
                "200",
            ]
        )
        manifest_path = _artifact(run_dir, "candidate.json")
        manifest = CandidateManifest.from_dict(json.loads(manifest_path.read_text()))
        if manifest.revision != revision or manifest.experiment != experiment.name:
            raise RuntimeError("candidate manifest does not match dispatched revision/experiment")
        decision = decide_candidate(
            manifest,
            best_pq=float(state["best_validation_pq"]),
            min_delta=config.min_delta,
            parity_tolerance=config.parity_tolerance,
        )
        history = {
            "experiment": experiment.name,
            "internal_pq": manifest.internal_pq,
            "self_evaluation_pq": manifest.self_evaluation_pq,
            "decision": decision.reason,
        }
        state["history"].append(history)
        state["status"] = "evaluated"
        _write_state(state_path, state)
        if not decision.accepted:
            continue

        submission = _artifact(run_dir, manifest.submission)
        message = f"auto:{experiment.name}:{revision[:12]}"
        state["submission_message"] = message
        state["status"] = "submitting"
        _write_state(state_path, state)
        runner(
            [
                "kaggle",
                "competitions",
                "submit",
                config.competition,
                "-f",
                str(submission),
                "-m",
                message,
            ]
        )
        state["submissions"] += 1
        state["best_validation_pq"] = manifest.internal_pq
        state["status"] = "submitted"
        _write_state(state_path, state)

        matched = _wait_for_submission(message, config, runner, sleep)
        state["latest_public_score"] = float(matched["publicScore"])
        leaderboard = json.loads(
            runner(
                ["kaggle", "competitions", "leaderboard", config.competition, "--format", "json"]
            )
        )
        state["leaderboard"] = leaderboard
        state["status"] = "complete"
        _write_state(state_path, state)

    if state["status"] != "complete":
        state["status"] = "finished"
        _write_state(state_path, state)
    return state
