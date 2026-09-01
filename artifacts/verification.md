# Verification record — 1 September 2026

## Final-tree checks

- `python -m unittest discover -s tests -v`: 35 tests passed.
- `python -m compileall -q solar_filament tests`: passed.
- Package editable build and `solar-filament --help`: passed.
- Both Kaggle notebook files are valid nbformat 4 JSON (11 pipeline cells and
  8 automated-experiment cells).
- Local Markdown file-link check: passed.
- Full dataset audit: 707 train files, 180 test files, 1,154 image records,
  8,199 annotations, and zero contract errors.
- CPU model smoke: DeepLabV3 forward/backward produced `(2, 1, 64, 64)` logits.
- Real-image dataset smoke: produced `(3, 64, 64)` input and binary
  `(1, 64, 64)` target.
- One real test image completed model inference, full-resolution connected
  components, COCO RLE encode, CSV write, and exact decode validation.
- Kaggle T4 baseline completed 20 epochs: validation `official_pq=0.1484172851`;
  its 1,712-instance submission scored `0.12` on the public leaderboard.
- Automation dry-run planned one bounded experiment without making an external
  call; fake-adapter integration tests cover kernel dispatch/output, gating,
  submission, leaderboard capture, and unfinished-state protection.
- The first automated T4 run selected `threshold=0.6`, `min_area=128`, reached
  validation PQ `0.1744241482`, matched organizer PQ within `1.91e-11`, and
  produced a decode-valid 1,580-instance submission. Public score improved from
  `0.12` to `0.16` (rank 407 at 07:58 UTC).

## Manual mutation gate

An automated Python mutation harness is not part of this repository. Three
high-value manual mutants were applied one at a time and reverted:

| Mutant | Expected protecting test | Result |
| --- | --- | --- |
| Official match `IoU > 0.5` changed to `>=` | strict threshold fixture | Killed |
| Component area `>= min_area` changed to `>` | exact-boundary component fixture | Killed |
| PQ false-positive weight `0.5` changed to `1.0` | fragmentation fixture | Killed |
| Minimum improvement `<` changed to `<=` | exact `min_delta` boundary | Killed |
| Evaluator mismatch `>` changed to `<` | disagreement gate | Killed |
| Submission budget `>=` changed to `>` | bounded dispatch test | Killed |
| Organizer IoU `>` changed to `>=` | cross-evaluator 0.5 boundary | Killed |

## Evidence boundary

The automated kernel has completed one authenticated end-to-end run. Five-fold
training remains outside the current bounded experiment scope.
