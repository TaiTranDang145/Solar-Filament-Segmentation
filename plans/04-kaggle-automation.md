# Plan: Bounded Kaggle experiment loop

**Branch**: `main`
**Status**: Active

## Goal

One command safely pushes the current GitHub revision, runs a Kaggle experiment,
downloads its evidence, submits only a validation improvement that passes the
organizer self-evaluation parity check, records the leaderboard result, and
stops after a configured number of runs.

## Acceptance Criteria

- [ ] Preflight rejects missing Kaggle auth, a dirty/unpushed Git tree, invalid
      limits, or a missing competition attachment before dispatching work.
- [ ] Every Kaggle run emits a machine-readable candidate manifest containing
      the Git SHA, training config, tuned threshold/min-area, internal PQ,
      organizer self-evaluation PQ, checkpoint, and submission paths.
- [ ] A candidate is eligible only when both evaluators agree within tolerance
      and validation PQ exceeds the saved best by the configured minimum delta.
- [ ] Submission and GPU runs never exceed explicit per-invocation limits;
      failures stop the loop and resumable state records the last known action.
- [ ] After an eligible submission, the controller polls Kaggle, records the
      public score and rank when available, and continues only while candidates
      and run budget remain.
- [ ] Dry-run is the default and performs no GitHub, kernel, or competition
      mutation; `--execute` is required for external state changes.

## Slices

### 1. Candidate quality gate — behavior change

**Actor**: experiment controller.  
**Trigger**: a Kaggle output manifest is downloaded.  
**Outcome**: deterministic accept/reject reason without external mutation.  
**Production path**: manifest parser → parity gate → improvement gate → state.  
**Evidence**: focused unit tests for equality tolerance, exact improvement
boundary, malformed manifests, and exhausted budgets.

### 2. One resumable Kaggle round — behavior change

**Actor**: repository owner.  
**Trigger**: `solar-filament automate --execute`.  
**Outcome**: pushed kernel output is downloaded and either rejected or submitted
once, with state persisted after each remote action.  
**Production path**: CLI → Git/Kaggle subprocesses → manifest → gate → submit →
submission polling → JSON state.  
**Evidence**: controller test with a recording command runner plus CLI dry-run.

### 3. Bounded candidate sequence — behavior change

**Actor**: repository owner.  
**Trigger**: controller receives a candidate file and `--max-runs`.  
**Outcome**: threshold/min-area candidates run in order and stop on budget,
failure, or exhaustion.  
**Production path**: candidate config → GitHub update → Kaggle round → state.  
**Evidence**: test proves no `max-runs + 1` dispatch and no submission without
an improvement.

### 4. Kaggle experiment notebook — behavior change

**Actor**: controller.  
**Trigger**: Kaggle executes the committed notebook.  
**Outcome**: train/tune/self-evaluate/infer produces the agreed manifest and
artifacts.  
**Production path**: competition data → model checkpoint → validation
post-processing sweep → organizer evaluator parity → test inference → outputs.  
**Evidence**: notebook JSON validation, contract test for output fields, existing
21 pipeline tests, and Kaggle smoke execution once authentication is available.

## Safety Defaults

- `max_runs=1`, `max_submissions=1`, positive `min_delta`.
- No automatic source-code invention; later training configurations come only
  from the reviewed candidate file.
- No force-push, no automatic merge, no secret in flags/config/logs.
- Ambiguous remote mutations stop and require status reconciliation rather than
  blind retry.

