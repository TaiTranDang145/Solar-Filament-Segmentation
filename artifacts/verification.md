# Verification record — 31 August 2026

## Final-tree checks

- `python -m unittest discover -s tests -v`: 21 tests passed.
- `python -m compileall -q solar_filament tests`: passed.
- Package editable build and `solar-filament --help`: passed.
- Notebook schema validation: 9 cells passed `nbformat.validate`.
- Local Markdown file-link check: passed.
- Full dataset audit: 707 train files, 180 test files, 1,154 image records,
  8,199 annotations, and zero contract errors.
- CPU model smoke: DeepLabV3 forward/backward produced `(2, 1, 64, 64)` logits.
- Real-image dataset smoke: produced `(3, 64, 64)` input and binary
  `(1, 64, 64)` target.
- One real test image completed model inference, full-resolution connected
  components, COCO RLE encode, CSV write, and exact decode validation.

## Manual mutation gate

An automated Python mutation harness is not part of this repository. Three
high-value manual mutants were applied one at a time and reverted:

| Mutant | Expected protecting test | Result |
| --- | --- | --- |
| Official match `IoU > 0.5` changed to `>=` | strict threshold fixture | Killed |
| Component area `>= min_area` changed to `>` | exact-boundary component fixture | Killed |
| PQ false-positive weight `0.5` changed to `1.0` | fragmentation fixture | Killed |

## Evidence boundary

No full 20-epoch or five-fold GPU training was run in this local CPU
environment. The Kaggle notebook owns that operational gate; its results must
replace the `RESULT REQUIRED` fields in the report before final submission.
