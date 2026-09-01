# Solar filament segmentation 2026

This repository provides a reproducible instance-segmentation baseline for the
MAGFiLO Kaggle challenge. It prevents annotation leakage, reproduces the
organizer's PQ implementation, trains a grayscale filament model, and writes a
decode-validated COCO RLE submission.

## Quick start

The commands below assume the downloaded competition directory is at
`filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m solar_filament audit \
  filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026
```

The audit exits with code `0` only when all images and annotations satisfy the
data contract. Its machine-readable output is written to
`artifacts/data-audit.json`.

## Train one leakage-safe fold

```bash
.venv/bin/python -m solar_filament folds \
  filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --output artifacts/folds.json

.venv/bin/python -m solar_filament train \
  filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026 \
  --fold 0 \
  --epochs 20 \
  --output-dir artifacts/fold-0
```

Training uses DeepLabV3-ResNet50 from `torchvision`, BCE + Dice loss, and one
annotator's target per physical image per epoch. Annotation selection cycles
deterministically; targets from different annotators are never merged into
duplicate instances. Batch size must be at least 2; incomplete one-item tail
batches are dropped because DeepLabV3's ASPP BatchNorm cannot train on them.
The validation prediction is compared with every
annotator set using the official `IoU > 0.5` rule.

The default ImageNet backbone downloads weights on first use. Pass
`--no-pretrained-backbone` in an offline Kaggle session.

## Create a submission

```bash
.venv/bin/python -m solar_filament infer \
  artifacts/fold-0/best.pt \
  filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  artifacts/submission.csv

.venv/bin/python -m solar_filament validate-submission \
  artifacts/submission.csv \
  --image-dir filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

Inference restores masks to `2048 x 2048`, extracts connected components,
encodes each non-empty instance with `pycocotools`, and decodes every row before
returning success. `artifacts/submission.run.json` records per-image instance
counts, areas, and runtime.

## Run on Kaggle

Open [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb), attach
the `filament-segmentation-2026` competition data, enable a GPU and Internet,
then run all cells. The notebook clones or pulls this repository, discovers the
competition directory, audits it, trains fold 0, creates `submission.csv`, and
validates the result.

## Automated Kaggle loop

Authenticate the Kaggle CLI once without putting a token in this repository:

```bash
kaggle auth login
```

Preview the bounded run (the default is dry-run and changes nothing):

```bash
solar-filament automate \
  --kernel taitrandang/solar-filament-automation
```

Dispatch one GPU run and allow at most one improved submission:

```bash
solar-filament automate \
  --kernel taitrandang/solar-filament-automation \
  --max-runs 1 \
  --max-submissions 1 \
  --execute
```

Candidate training configurations and post-processing grids live in
`automation/experiments.json`. Each Kaggle run trains one reviewed config,
tunes `threshold`/`min_area`, checks its PQ against the organizer's
Self_Evaluation_Notebook logic, and emits `candidate.json`. The controller only
submits when both evaluators agree within `1e-6` and validation PQ improves by
at least `0.001` over the configured baseline (currently `0.1484172851`). It
then records the scored submission and leaderboard response in
`artifacts/automation/state.json`.

The controller requires a clean `main` equal to `origin/main`, never commits or
force-pushes source, defaults to one run and one submission, and refuses to
overwrite an unfinished remote state after an ambiguous failure.

## What is measured

The organizer's self-evaluation notebook v6, retrieved on 31 August 2026,
defines a true-positive pair as `IoU > 0.5`. It sums matched IoU values and uses
the denominator `TP + 0.5*FP + 0.5*FN`. Its aggregation unit is an
annotator-image record, so an image annotated three times contributes three
comparisons. This project reports that value as `official_pq` and also emits
`macro_pq` as a diagnostic mean across annotation sets.

The local fixtures cover perfect matching, the strict `0.5` boundary,
fragmentation, false positives, false negatives, and multi-annotator
aggregation.

## Data contract

The checked dataset snapshot contains 707 train images, 180 test images, 1,154
annotator-image records, and 8,199 filament instances. All image files are
grayscale JPEGs at `2048 x 2048`. Grouped folds use the physical `file_name`,
which keeps all annotation sets for the same pixels on one side of a split.

See [artifacts/data-audit.json](artifacts/data-audit.json) for the generated
snapshot and [plans/00-solution-roadmap.md](plans/00-solution-roadmap.md) for
active experiment work.

## Project map

| Path | Purpose |
| --- | --- |
| `solar_filament/data.py` | COCO parsing, audit, and grouped folds |
| `solar_filament/metrics.py` | Official PQ-compatible overlap accounting |
| `solar_filament/masks.py` | Polygon rasterization, components, and COCO RLE |
| `solar_filament/training.py` | Multi-annotator dataset and fold training |
| `solar_filament/inference.py` | Full-resolution inference and run manifest |
| `solar_filament/submission.py` | CSV generation and decode audit |
| `tests/` | Behavior fixtures and real-dataset contract checks |
| `report/technical-report.md` | Four-page report source outline |

## Limits

The repository contains a runnable baseline, not a trained checkpoint. Full
training and five-fold ablations require a Kaggle GPU. Connected components are
deliberately the first instance extractor; add watershed, a boundary head,
tiling, TTA, or an ensemble only when out-of-fold split/merge evidence justifies
the added mechanism.
