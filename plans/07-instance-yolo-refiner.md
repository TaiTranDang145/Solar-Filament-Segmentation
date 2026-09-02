# Plan: Class-agnostic instance pipeline

**Branch**: main
**Status**: Active

## Goal

Produce a grouped-fold, official-PQ-scored YOLO11s-seg plus crop-refiner candidate using segmentation ground truth only.

## Acceptance Criteria

- [ ] Every YOLO label is class `0`; chirality, supplied boxes, and spines are never consumed.
- [ ] Physical-image folds remain grouped and one segmentation-derived annotation set is selected deterministically.
- [ ] One Kaggle run trains YOLO and the crop refiner, tunes instance confidence/mask threshold with official PQ, and emits a valid candidate/submission.
- [ ] Internal and organizer self-evaluation PQ agree within `1e-6`.

## Slices

### Slice 1: A Kaggle operator runs one legal instance experiment and receives a gated candidate

**Value**: The operator can test a model that preserves filament identity instead of recovering it from a semantic union.
**Path**: experiment config -> grouped COCO conversion -> YOLO/refiner training -> official PQ tuning -> test inference -> candidate and submission artifacts.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: planning, TDD, testing; refactoring N/A unless GREEN exposes duplication; mutation-testing at PR readiness.
**Reduction program**: N/A.
**Acceptance criteria**: The four plan-level criteria above; accepted by the owner on 2026-09-02.
**RED**: Converter behavior tests reject category leakage and verify deterministic segmentation-only annotation selection; notebook/config contract test requires the new route.
**GREEN**: Minimum instance pipeline and Kaggle notebook branch.
**REFACTOR**: N/A unless it reduces code after GREEN.
**PRE-PR MUTATION or alternate evidence**: Manual Python mutations if no Python mutation harness is present, plus full non-watch unittest discovery and notebook JSON validation.
**PR-ready when**: Tests and static checks pass, the notebook dry-run selects only this experiment, and the owner approves the commit.
**Slice complete when**: The approved commit is pushed and the one-fold Kaggle proof produces official PQ.

## Pre-PR Quality Gate

1. Focused RED/GREEN evidence.
2. Manual mutation evidence for conversion boundaries and threshold filtering.
3. Full unittest discovery, compileall, JSON validation, and automation dry-run.
4. Owner approval before commit.
