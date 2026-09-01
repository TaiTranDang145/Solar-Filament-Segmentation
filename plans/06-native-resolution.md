# Plan: Native-resolution segmentation

**Status**: Active

## Goal

Raise the trustworthy grouped-validation PQ beyond `0.25` by preserving native filament detail, then submit at most one parity-checked candidate.

## Acceptance Criteria

- [x] Native mode trains and predicts from a centered `1888×1888` crop without resizing it to `768`.
- [x] Each grayscale frame is normalized from its own on-disk intensity distribution.
- [x] Native inference reconstructs `2048×2048` masks and supports dihedral TTA.
- [x] Post-processing searches threshold, minimum area, and morphological closing from one inference pass per image.
- [ ] The organizer self-evaluation PQ matches internal PQ within `1e-6`.
- [ ] Submit at most once, only when grouped validation PQ is at least `0.25`.

## Slices

### Slice 1: Preserve native image geometry

The Kaggle training path accepts batch-size-one native crops and reconstructs predictions at full image size. Verify with focused dataset/config behavior tests.

### Slice 2: Train the native U-Net

The same `train` entry point builds the EfficientViT U-Net, uses gradient accumulation, and saves a loadable checkpoint. Verify the default DeepLab contract remains compatible and run a native forward smoke test.

### Slice 3: Tune once, score many post-processors

The notebook evaluates threshold/min-area/closing candidates without repeating neural inference for every grid cell, runs organizer parity, and emits the existing candidate contract.

### Slice 4: Kaggle promotion gate

Run one GPU experiment and submit only if PQ reaches `0.25`; otherwise stop and promote the detector→crop-refiner strategy.
