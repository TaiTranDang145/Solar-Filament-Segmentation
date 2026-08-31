# Automatic segmentation of solar filaments in GONG H-alpha observations

> Transfer this content into the organizer's four-page Overleaf template.
> Replace every `RESULT REQUIRED` field with evidence from the frozen Kaggle
> run before submission.

## Abstract

We present a reproducible baseline for instance segmentation of solar
filaments in 2048 × 2048 grayscale GONG H-alpha observations. The method trains
a binary DeepLabV3-ResNet50 semantic segmenter and converts the full-resolution
probability map into individual filament masks with connected components. Data
splits group all annotation sets belonging to the same physical observation,
preventing pixel-identical train-validation leakage. The implementation
matches the competition's strict IoU-above-0.5 Panoptic Quality protocol and
exports decode-validated COCO run-length encodings. On five-fold validation,
the frozen configuration achieved **RESULT REQUIRED: mean PQ ± standard
deviation**, with **RESULT REQUIRED: inference time** per image.

## 1. Problem and data

The MAGFiLO competition data contain 707 training and 180 test observations.
Every image is an 8-bit grayscale JPEG of 2048 × 2048 pixels. The training JSON
contains 1,154 annotator-image records and 8,199 filament instances because a
physical image may have one to three independent annotation sets. Filaments
are thin and highly imbalanced: the median annotated area in the supplied
snapshot is 1,228 pixels, while the full image contains 4,194,304 pixels.

We use only the provided H-alpha images at inference. Category labels and
spines are excluded because the target is a single filament mask class. Folds
are grouped by physical filename and approximately balanced by year,
observatory, and instance-count bucket.

## 2. Method

Each image is normalized to `[0, 1]`, repeated across three channels for the
ImageNet backbone, and resized to 768 × 768. Horizontal flips, vertical flips,
and 90-degree rotations provide deterministic epoch-varying augmentation. For
images with multiple annotation sets, training cycles through exactly one set
per epoch rather than merging inconsistent instances.

DeepLabV3 with a ResNet50 encoder predicts one foreground logit map. Training
minimizes the sum of positive-weighted binary cross-entropy and soft Dice loss.
At inference, logits are bilinearly restored to 2048 × 2048, thresholded at
0.5, split into 8-connected components, and filtered below 32 pixels. Each
remaining component is encoded independently with COCO compressed RLE.

## 3. Evaluation and experiments

The official scorer declares a ground-truth/prediction pair true positive only
when IoU is strictly greater than 0.5. It computes

`PQ = sum(matched IoU) / (TP + 0.5 FP + 0.5 FN)`.

We aggregate over every annotator-image record, matching the organizer's
self-evaluation notebook v6. We additionally inspect Dice, positive-pair IoU,
one-to-many fragmentation, many-to-one merging, false positives, false
negatives, and per-image latency.

| Configuration | PQ | Dice | One-to-many | Many-to-one | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Semantic baseline, 768 px | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED |
| Resolution ablation | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED |
| Post-processing ablation | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED | RESULT REQUIRED |

Include one morphology figure with raw image, ground truth, prediction, TP/FP/FN
overlay, and one failure figure showing fragmentation or over-merging.

## 4. Results, limitations, and reproducibility

**RESULT REQUIRED:** Report mean and per-fold PQ, variance, small-instance
recall, and the observed difference between public leaderboard and out-of-fold
performance. Discuss whether the dominant error is missed low-contrast
filaments, fragmentation, over-merging, or false detections on active regions.

The baseline loses fine structure when resizing a full solar disk to 768 × 768
and cannot separate touching filament instances without a boundary cue.
Connected-component parameters may also trade fragmentation for merging.
These limitations motivate overlapping tiles, boundary-aware targets, or
watershed only when controlled ablations improve PQ.

The public repository includes pinned dependencies, grouped fold assignments,
the official-metric fixtures, training and inference code, a Kaggle notebook,
and submission decode checks. Record the final repository URL, commit hash,
checkpoint checksum, random seed, Kaggle accelerator, package versions, and
wall-clock runtime here before exporting the PDF.
