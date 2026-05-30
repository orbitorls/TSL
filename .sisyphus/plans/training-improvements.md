# Thai Sign Language Training Improvements

## TL;DR
> **Summary**: Make isolated-sign training trustworthy before tuning: fix grouped evaluation, train-only augmentation, feature/preprocessing parity, and migrate the active trainer from legacy into `src/train/`. Optimize for real-world Macro F1 with minimal dependency additions.
> **Deliverables**:
> - Grouped split + augmentation-safety implementation and tests
> - 162-dim feature schema/preprocessing manifests shared by training and inference
> - Canonical `src/train/` training pipeline used by CLI and Colab
> - Reproducible baseline and controlled GRU/MLP tuning reports
> - Fast pytest/CI validation plus minimal notebook syntax validation
> **Effort**: Large
> **Parallel**: YES - 3 implementation waves + final verification
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 5 → Task 8 → Task 9 → Task 10

## Context
### Original Request
User asked in Thai to find all improvements so training becomes as good, usable, and accurate as possible.

### Interview Summary
- Primary goal: real-world accuracy.
- Primary metric: Macro F1.
- Dependency policy: minimal additions only.
- Scope: isolated TSL-51 signs only; sentence-level/CTC implementation is out of scope.
- CLI strategy: migrate canonical active training into `src/train/`.
- Test strategy: TDD for risky refactors; tests-after acceptable for low-risk notebook/docs wiring.

### Research Summary
- `src/cli/train.py` delegates to `legacy/root_scripts/train_tsl51_v3.py`; active CLI training is still legacy.
- Legacy/Colab path already has StratifiedKFold, per-fold normalization, class weights, augmentation, early stopping, metrics, JSON/PNG outputs, and checkpointing.
- `src/train/trainer.py` has modular optimization logic but does not own disk checkpointing and is not the authoritative end-to-end path.
- Highest-risk accuracy issue: augmented variants can leak across validation/test unless grouped by original/base sample; augmentation must happen after splitting and only for training.
- Feature/preprocessing logic is fragmented across `src/core/features.py`, `src/data/feature_extraction.py`, `src/data/extractor.py`, `src/train/augment.py`, notebooks, and web inference.
- Test infra exists via `pyproject.toml`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `tests/conftest.py`, and representative tests, but lacks grouped split tests, notebook validation, stable tiny fixture-data, and end-to-end train/save/load tests.

### Metis Review (gaps addressed)
- Defaulted primary real-world validation protocol to **video-family-held-out grouped split** because signer metadata availability is unknown. If signer/source metadata exists, report signer/source grouped metrics as secondary; do not make them the primary gate unless metadata is reliably present.
- Defaulted `tsl51_expert_full` augmented samples to **train-only**; validation/test must use originals/non-augmented samples.
- Defaulted feature scope to **existing 162-dim basic schema only**. Do not implement enhanced features in this plan; reject or mark `enhanced` unsupported until separately planned.
- Defaulted compatibility to preserve the `tsl-train` console script, common legacy flags, checkpoint loadability, and result JSON fields while moving active implementation to `src/train/`.
- Defaulted success threshold: after trustworthiness migration, the tuned model must improve grouped-split Macro F1 by at least **+2.0 percentage points** over the trusted baseline or explicitly keep the simpler baseline if tuning fails to improve.

## Work Objectives
### Core Objective
Create a trustworthy, reproducible, isolated-sign training workflow where Macro F1 on a grouped real-world split is the primary gate, and where training, evaluation, CLI, Colab, and inference share one feature/preprocessing contract.

### Deliverables
- Grouped split module and split manifests.
- Train-only augmentation flow and tests.
- 162-dim feature schema manifest and parity tests.
- Preprocessing/normalization manifest saved with every checkpoint.
- Canonical `src/train/` pipeline and migrated CLI entrypoint.
- Tiny fixture training smoke suite.
- Baseline + tuning reports using Macro F1 as primary metric.
- Minimal notebook validation command.

### Definition of Done (verifiable conditions with commands)
- `python -m pytest tests/train/test_grouped_splits.py tests/train/test_augmentation_split_safety.py -q` passes.
- `python -m pytest tests/core/test_feature_schema.py tests/inference/test_preprocessing_manifest.py -q` passes.
- `python -m pytest tests/train/test_tiny_training_smoke.py -q` passes and writes checkpoint, metrics JSON, split manifest, and preprocessing manifest under a temp directory.
- `python -m src.cli.train --help` and `tsl-train --help` both resolve to the canonical `src/train/` implementation and expose preserved common flags.
- A trusted baseline run and tuned run both write metrics containing `macro_f1`, `weighted_f1`, `accuracy`, top-k metrics, per-class metrics, split strategy, seed, dataset, feature schema version, preprocessing manifest path, and checkpoint path.
- Tuned run improves grouped-split Macro F1 by ≥2.0 percentage points over the trusted baseline, or the report explicitly selects the simpler baseline with evidence.

### Must Have
- Primary gate: Macro F1 on video-family-held-out grouped split.
- Validation/test contain original/non-augmented samples only.
- Augmentation generated/used only after split and only in training.
- Feature scope locked to 162-dim basic schema for this plan.
- Minimal dependencies: prefer stdlib, existing PyTorch/sklearn/pytest; notebook validation must use pytest JSON inspection and add no new dependency.
- Active implementation in `src/train/`, `src/core/`, `src/data/`, and relevant `src/inference/` code; no new active implementation in `legacy/`, top-level `scripts/`, or top-level `utils/`.

### Must NOT Have
- No sentence recognition, CTC implementation, language modeling, or translation pipeline work.
- No new web UI work.
- No new dataset collection.
- No heavy experiment tracking frameworks such as MLflow unless separately approved.
- No architecture tuning before grouped split, augmentation safety, feature schema, and preprocessing parity tests pass.
- No validation/test metrics from leaked augmented variants as primary evidence.
- No manual notebook execution as an acceptance gate.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: TDD for risky refactors; tests-after for low-risk notebook/docs wiring. Framework: pytest from `pyproject.toml`.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Tasks 1-4 — split trustworthiness, augmentation safety, feature schema, preprocessing manifest.
Wave 2: Tasks 5-8 — canonical pipeline migration, CLI/Colab wiring, fixture smoke tests, metrics/result contract.
Wave 3: Tasks 9-12 — trusted baseline, controlled tuning, notebook validation, brittle-test cleanup.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 8, 9, 10.
- Task 2 blocks Tasks 8, 9, 10.
- Task 3 blocks Tasks 4, 5, 7, 9, 10.
- Task 4 blocks Tasks 5, 7, 9, 10.
- Task 5 blocks Tasks 6, 7, 8, 9, 10.
- Task 6 blocks Tasks 9, 11.
- Task 7 blocks Tasks 9, 12.
- Task 8 blocks Tasks 9, 10.
- Task 9 blocks Task 10.
- Task 10 blocks final verification.
- Tasks 11-12 can run after Task 6/7 respectively and before final verification.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → `deep`, `unspecified-high`
- Wave 2 → 4 tasks → `deep`, `quick`, `unspecified-high`
- Wave 3 → 4 tasks → `unspecified-high`, `quick`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Implement grouped split metadata and leakage tests

  **What to do**: Create grouped split utilities under `src/train/` that produce deterministic train/val/test split manifests for isolated signs. Primary group key is `video_family_id`, derived from available metadata when present or from filename/base sample ID by stripping augmentation suffixes and fold-specific artifacts. Validation/test must exclude augmented samples. Add TDD tests first for no group overlap and deterministic seeds.
  **Must NOT do**: Do not use random `StratifiedKFold` as the primary real-world gate. Do not require signer metadata to exist. Do not include augmented variants in validation/test.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: high-risk data leakage logic controls all reported accuracy.
  - Skills: `data-pipeline-safety`, `tdd` - Need leakage-safe data pipeline and failing tests first.
  - Omitted: `hpo-search-stability` - No hyperparameter study state in this task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2, 8, 9, 10 | Blocked By: none

  **References**:
  - Pattern: `src/data/loader.py:110-817` - dataset loading/caching/validation entrypoints to integrate split metadata.
  - Pattern: `legacy/root_scripts/train_tsl51_v3.py:1320-1576` - current train/CV behavior to replace for real-world gate.
  - Test: `tests/test_loader.py:10-64` - existing validation test style.
  - New: `src/train/splits.py`, `tests/train/test_grouped_splits.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/train/test_grouped_splits.py -q` passes.
  - [ ] Test asserts no `video_family_id` appears in more than one split.
  - [ ] Test asserts split manifest is deterministic with the same seed.
  - [ ] Test asserts augmented rows are absent from validation/test split manifests.
  - [ ] Split manifest includes dataset name, seed, split strategy, group key, class counts, and sample counts.

  **QA Scenarios**:
  ```
  Scenario: Grouped split prevents leakage
    Tool: Bash
    Steps: python -m pytest tests/train/test_grouped_splits.py::test_video_family_groups_do_not_cross_splits -q
    Expected: Exit code 0; assertion proves train/val/test group sets are pairwise disjoint.
    Evidence: .sisyphus/evidence/task-1-grouped-split.txt

  Scenario: Augmented variant leakage is rejected
    Tool: Bash
    Steps: python -m pytest tests/train/test_grouped_splits.py::test_augmented_variant_in_validation_is_rejected -q
    Expected: Exit code 0; invalid split raises/asserts a clear leakage error.
    Evidence: .sisyphus/evidence/task-1-grouped-split-error.txt
  ```

  **Commit**: YES | Message: `feat(train): add leakage-safe grouped splits` | Files: `src/train/splits.py`, `tests/train/test_grouped_splits.py`

- [x] 2. Move augmentation after splitting and enforce train-only augmentation

  **What to do**: Refactor training data assembly so augmentation is applied only to the training split after Task 1 split manifests exist. Add a guard that rejects augmentation on validation/test. Keep current noise/scale behavior, but gate flip augmentation to the 162-dim basic schema only.
  **Must NOT do**: Do not augment before split. Do not silently apply flip to non-162 feature vectors. Do not change augmentation semantics for validation/test.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: contained but safety-critical data-flow change.
  - Skills: `data-pipeline-safety`, `tdd` - Prevent leakage and write failure tests first.
  - Omitted: `hpo-search-stability` - No optimization state involved.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 8, 9, 10 | Blocked By: 1

  **References**:
  - Pattern: `src/train/augment.py:6-70` - existing augmentation functions and 162-dim flip assumption.
  - Pattern: `legacy/root_scripts/train_tsl51_v3.py:1320-1576` - current legacy augmentation/split flow.
  - Pattern: `colab/02_train.ipynb` grep lines 165-226, 336-419 - notebook augmentation/training flow to align later.
  - New: `tests/train/test_augmentation_split_safety.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/train/test_augmentation_split_safety.py -q` passes.
  - [ ] Tests prove augmented sample count increases only in train split.
  - [ ] Tests prove validation/test sample IDs remain original-only.
  - [ ] Non-162 flip request raises a clear `ValueError` or skips with explicit reason per project convention.

  **QA Scenarios**:
  ```
  Scenario: Train-only augmentation
    Tool: Bash
    Steps: python -m pytest tests/train/test_augmentation_split_safety.py::test_augmentation_applies_only_to_train_split -q
    Expected: Exit code 0; train count increases while val/test counts and IDs stay unchanged.
    Evidence: .sisyphus/evidence/task-2-augmentation.txt

  Scenario: Invalid flip schema fails safely
    Tool: Bash
    Steps: python -m pytest tests/train/test_augmentation_split_safety.py::test_flip_rejects_non_basic_schema -q
    Expected: Exit code 0; non-162 vector cannot be flipped silently.
    Evidence: .sisyphus/evidence/task-2-augmentation-error.txt
  ```

  **Commit**: YES | Message: `fix(train): isolate augmentation to training split` | Files: `src/train/augment.py`, `src/train/splits.py`, `tests/train/test_augmentation_split_safety.py`

- [x] 3. Lock and centralize the 162-dim basic feature schema

  **What to do**: Make `src/core/features.py` the authoritative source for the 162-dim basic schema and expose a schema manifest/version used by data extraction, augmentation, training, and inference. For this plan, reject `feature_level="enhanced"` with a clear unsupported message instead of partially supporting it.
  **Must NOT do**: Do not implement enhanced features. Do not maintain duplicate feature-order definitions. Do not alter the 162 feature order without migration tests.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: schema changes affect training, data, inference, and web parity.
  - Skills: `ml-backend-integration`, `tdd` - Must audit all dispatch/consumer paths and test interface completeness.
  - Omitted: `hpo-search-stability` - No HPO state involved.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 7, 9, 10 | Blocked By: none

  **References**:
  - API/Type: `src/core/features.py:10-74` - core feature constants/extractor.
  - Pattern: `src/data/feature_extraction.py:13-179` - data/inference feature extraction API and enhanced inconsistency.
  - Pattern: `src/train/augment.py:6-70` - augmentation assumes basic 162-dim layout.
  - Test: `tests/test_feature_extraction.py:11-30` - existing feature shape tests.
  - Test: `tests/test_integration.py:60-112` - registry/dim consistency tests.
  - New: `tests/core/test_feature_schema.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/core/test_feature_schema.py tests/test_feature_extraction.py tests/test_integration.py -q` passes.
  - [ ] One exported schema object/version defines feature order, dim=162, and level=`basic`.
  - [ ] Data extraction, augmentation, trainer config, and inference import or validate against that schema.
  - [ ] `feature_level="enhanced"` has one explicit behavior: unsupported error with actionable message.

  **QA Scenarios**:
  ```
  Scenario: Basic schema parity
    Tool: Bash
    Steps: python -m pytest tests/core/test_feature_schema.py::test_basic_schema_is_162_dim_and_shared -q
    Expected: Exit code 0; all consumers report schema level basic, dim 162, same order/version.
    Evidence: .sisyphus/evidence/task-3-feature-schema.txt

  Scenario: Enhanced schema is not silently accepted
    Tool: Bash
    Steps: python -m pytest tests/core/test_feature_schema.py::test_enhanced_feature_level_is_explicitly_unsupported -q
    Expected: Exit code 0; enhanced request raises/returns a clear unsupported error.
    Evidence: .sisyphus/evidence/task-3-feature-schema-error.txt
  ```

  **Commit**: YES | Message: `refactor(core): centralize basic feature schema` | Files: `src/core/features.py`, `src/data/feature_extraction.py`, `src/train/augment.py`, `tests/core/test_feature_schema.py`

- [x] 4. Save and load preprocessing manifests with normalization parity tests

  **What to do**: Add a preprocessing manifest saved alongside checkpoints containing feature schema version, normalization type, mean/std stats, target frames, sequence mode, dataset, split strategy, MediaPipe/source metadata when available, and label map path/hash. Ensure inference loads this manifest and uses the same normalization semantics.
  **Must NOT do**: Do not let inference guess feature dimensions or normalization behavior from checkpoint tensors. Do not create incompatible checkpoint formats without migration/loading tests.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: cross-module train/inference contract and checkpoint compatibility.
  - Skills: `ml-backend-integration`, `tdd` - New interface must be complete across training/inference loaders.
  - Omitted: `data-pipeline-safety` - Less about transformations, more about interface parity.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 7, 9, 10 | Blocked By: 3

  **References**:
  - Pattern: `src/core/normalizer.py:6-53` - current normalizer serialization.
  - Pattern: `src/data/extractor.py:27-50,107-185` - inference-time normalization helper.
  - Pattern: `src/inference/predict_video.py` - video inference path to validate manifest loading.
  - Pattern: `tsl_web/app.py` - web model/inference endpoint uses packaged runtime assets.
  - New: `tests/inference/test_preprocessing_manifest.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/inference/test_preprocessing_manifest.py -q` passes.
  - [ ] Training save path writes `preprocessing_manifest.json` beside checkpoint.
  - [ ] Inference refuses to run when manifest schema/dim mismatches model expectation.
  - [ ] Manifest includes normalization stats and label map info sufficient for reproducible inference.

  **QA Scenarios**:
  ```
  Scenario: Inference loads training preprocessing manifest
    Tool: Bash
    Steps: python -m pytest tests/inference/test_preprocessing_manifest.py::test_inference_uses_saved_normalization_stats -q
    Expected: Exit code 0; loaded stats match saved stats exactly for deterministic fixture.
    Evidence: .sisyphus/evidence/task-4-preprocessing-manifest.txt

  Scenario: Schema mismatch fails closed
    Tool: Bash
    Steps: python -m pytest tests/inference/test_preprocessing_manifest.py::test_schema_mismatch_refuses_prediction -q
    Expected: Exit code 0; mismatch raises clear error and no prediction is returned.
    Evidence: .sisyphus/evidence/task-4-preprocessing-manifest-error.txt
  ```

  **Commit**: YES | Message: `feat(train): persist preprocessing manifests` | Files: `src/core/normalizer.py`, `src/train/*`, `src/inference/*`, `tests/inference/test_preprocessing_manifest.py`

- [x] 5. Build canonical `src/train/` training pipeline and retire legacy authority

  **What to do**: Move end-to-end orchestration into `src/train/` using existing modular pieces: config, loader, Trainer, evaluator, grouped splits, augmentation, manifests, checkpoint saving, and result writing. Legacy script may remain for compatibility, but it must no longer be the authoritative path for `tsl-train`.
  **Must NOT do**: Do not add active implementation to `legacy/`. Do not change model math and pipeline migration in the same step unless covered by tests. Do not tune hyperparameters in this task.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: central migration across CLI, config, training, evaluation, checkpoint outputs.
  - Skills: `ml-backend-integration`, `tdd` - New pipeline must implement all expected branches/outputs.
  - Omitted: `hpo-search-stability` - No HPO yet.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 6, 7, 8, 9, 10 | Blocked By: 3, 4

  **References**:
  - Pattern: `src/train/trainer.py:109-462` - modular optimization/training logic.
  - Pattern: `src/train/config.py:12-182` - training config and presets.
  - Pattern: `src/train/evaluator.py:37-250` - metrics/result helpers.
  - Pattern: `legacy/root_scripts/train_tsl51_v3.py:1063-1619` - legacy CLI/pipeline behavior to preserve where required.
  - Pattern: `src/cli/train.py:4-7` - current shim to replace.
  - New: `src/train/pipeline.py`, `tests/train/test_pipeline_contract.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/train/test_pipeline_contract.py -q` passes.
  - [ ] Pipeline accepts config object and writes checkpoint, metrics JSON, split manifest, preprocessing manifest, and label map.
  - [ ] Pipeline uses grouped split and train-only augmentation from Tasks 1-2.
  - [ ] Pipeline reports Macro F1 as primary selection metric.
  - [ ] Legacy script is not imported by `src.cli.train` for normal execution.

  **QA Scenarios**:
  ```
  Scenario: Canonical pipeline writes all required artifacts
    Tool: Bash
    Steps: python -m pytest tests/train/test_pipeline_contract.py::test_pipeline_writes_required_artifacts -q
    Expected: Exit code 0; temp run directory contains checkpoint, metrics, split manifest, preprocessing manifest, label map.
    Evidence: .sisyphus/evidence/task-5-pipeline.txt

  Scenario: Pipeline rejects missing grouped split config
    Tool: Bash
    Steps: python -m pytest tests/train/test_pipeline_contract.py::test_pipeline_requires_grouped_split_for_real_world_mode -q
    Expected: Exit code 0; invalid config fails before training with clear message.
    Evidence: .sisyphus/evidence/task-5-pipeline-error.txt
  ```

  **Commit**: YES | Message: `refactor(train): make src training pipeline canonical` | Files: `src/train/pipeline.py`, `src/train/trainer.py`, `src/train/config.py`, `src/cli/train.py`, `tests/train/test_pipeline_contract.py`

- [x] 6. Migrate CLI and Colab wiring to the canonical pipeline

  **What to do**: Update `src/cli/train.py` so `python -m src.cli.train --help` and `tsl-train --help` use the canonical `src/train/` pipeline. Preserve common legacy flags and output locations where practical, and add explicit `--real-world-mode`, `--split-strategy`, and `--primary-metric` flags. Update `colab/02_train.ipynb` to call/import the canonical path rather than duplicating training logic.
  **Must NOT do**: Do not require manual Colab execution as verification. Do not break console script declarations in `pyproject.toml`. Do not remove legacy files unless separately approved.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: wiring across CLI/notebook/package metadata.
  - Skills: `tdd` - CLI compatibility tests before edits.
  - Omitted: `data-pipeline-safety` - No new data transform logic.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 11 | Blocked By: 5

  **References**:
  - Pattern: `src/cli/train.py:4-7` - current legacy delegation.
  - API/Type: `pyproject.toml:52-57,125-152` - console script and pytest config.
  - Pattern: `colab/02_train.ipynb` grep lines 55-76, 165-226, 336-419 - current notebook config/training flow.
  - New: `tests/cli/test_train_cli.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/cli/test_train_cli.py -q` passes.
  - [ ] `python -m src.cli.train --help` exits 0 and includes dataset, model type, epochs, seed, split strategy, primary metric, real-world mode, and output dir flags.
  - [ ] `tsl-train --help` exits 0 and maps to same parser.
  - [ ] `colab/02_train.ipynb` contains canonical import/call and no duplicated split/augmentation implementation cells.

  **QA Scenarios**:
  ```
  Scenario: CLI help uses canonical parser
    Tool: Bash
    Steps: python -m src.cli.train --help
    Expected: Exit code 0; stdout includes --split-strategy, --primary-metric, --real-world-mode, --output-dir, and no legacy import traceback.
    Evidence: .sisyphus/evidence/task-6-cli-help.txt

  Scenario: Console script remains available
    Tool: Bash
    Steps: tsl-train --help
    Expected: Exit code 0; stdout matches canonical help flags.
    Evidence: .sisyphus/evidence/task-6-console-help.txt
  ```

  **Commit**: YES | Message: `feat(cli): route training through canonical pipeline` | Files: `src/cli/train.py`, `colab/02_train.ipynb`, `tests/cli/test_train_cli.py`

- [ ] 7. Add deterministic tiny fixture training smoke test and replace brittle paths

  **What to do**: Add tiny generated or checked-in fixture data sufficient for a fast train/save/load smoke. Replace tests that hardcode local checkpoints/data paths with temp fixtures or skips tied to explicit markers. Make smoke test run on CPU under CI time constraints.
  **Must NOT do**: Do not commit large model/data artifacts. Do not rely on `D:\TSL\TSL\models\...` or local full dataset paths. Do not require GPU.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: test reliability and artifact contract coverage.
  - Skills: `tdd` - Add failing smoke/brittle path tests before implementation.
  - Omitted: `hpo-search-stability` - No HPO.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 12 | Blocked By: 3, 4, 5

  **References**:
  - Fixture: `tests/conftest.py` - existing synthetic fixtures.
  - Brittle: `tests/test_evaluation.py` - hardcoded checkpoint path noted by test-infra research.
  - Brittle: `tests/test_model_with_videos.py` - local full-data/model dependencies noted by test-infra research.
  - Manual: `tests/smoke_inference.py` - not pytest-discovered.
  - New: `tests/train/test_tiny_training_smoke.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/train/test_tiny_training_smoke.py tests/test_evaluation.py -q` passes without local model/data paths.
  - [ ] Tiny smoke produces checkpoint, metrics JSON, split manifest, preprocessing manifest in `tmp_path`.
  - [ ] CPU runtime stays under 60 seconds on representative local/CI environment.
  - [ ] Data-dependent/manual tests are either converted to pytest with fixtures or marked/skipped with explicit reason.

  **QA Scenarios**:
  ```
  Scenario: Tiny CPU training smoke writes artifacts
    Tool: Bash
    Steps: python -m pytest tests/train/test_tiny_training_smoke.py::test_tiny_training_run_writes_artifacts -q
    Expected: Exit code 0; all required artifacts exist under pytest tmp_path.
    Evidence: .sisyphus/evidence/task-7-tiny-smoke.txt

  Scenario: Hardcoded local paths are gone
    Tool: Bash
    Steps: python -m pytest tests/test_evaluation.py -q
    Expected: Exit code 0; test uses generated temp checkpoint/metrics instead of D:\TSL\TSL\models path.
    Evidence: .sisyphus/evidence/task-7-brittle-paths.txt
  ```

  **Commit**: YES | Message: `test(train): add tiny training smoke coverage` | Files: `tests/conftest.py`, `tests/train/test_tiny_training_smoke.py`, `tests/test_evaluation.py`, `tests/test_model_with_videos.py`, `tests/smoke_inference.py`

- [ ] 8. Make metrics/result contract Macro-F1-first and grouped-split-aware

  **What to do**: Update evaluator/result serialization so `macro_f1` is the primary selection/reporting metric for real-world mode. Result JSON must include grouped split metadata, secondary metrics, per-class metrics, confusion matrix data/path, seed, feature schema version, preprocessing manifest path, and checkpoint path.
  **Must NOT do**: Do not remove accuracy/top-k metrics. Do not select best checkpoint by validation accuracy in real-world mode. Do not rely on visual plot inspection.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: metric contract affects model selection and reporting.
  - Skills: `tdd` - Metrics contract tests should fail before result schema changes.
  - Omitted: `data-pipeline-safety` - Split safety handled in earlier tasks.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 9, 10 | Blocked By: 1, 2, 5

  **References**:
  - Pattern: `src/train/evaluator.py:37-250` - current metric computation/save helpers.
  - Pattern: `src/train/trainer.py:109-462` - current early stopping/metric aggregation.
  - Test: `tests/test_evaluator.py:48-130` - existing metric correctness coverage.
  - Pattern: `colab/03_evaluate.ipynb` grep lines 84-96, 248, 280, 305-317 - evaluation output expectations.
  - New: `tests/train/test_metrics_contract.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/test_evaluator.py tests/train/test_metrics_contract.py -q` passes.
  - [ ] Best checkpoint selection in real-world mode uses validation `macro_f1`.
  - [ ] Result JSON includes all required contract fields.
  - [ ] Missing `macro_f1` in metrics raises a contract error before checkpoint selection.

  **QA Scenarios**:
  ```
  Scenario: Macro F1 controls best checkpoint
    Tool: Bash
    Steps: python -m pytest tests/train/test_metrics_contract.py::test_real_world_mode_selects_by_macro_f1 -q
    Expected: Exit code 0; fixture with lower accuracy but higher macro_f1 is selected.
    Evidence: .sisyphus/evidence/task-8-metrics.txt

  Scenario: Incomplete metrics fail contract
    Tool: Bash
    Steps: python -m pytest tests/train/test_metrics_contract.py::test_missing_macro_f1_fails_result_contract -q
    Expected: Exit code 0; missing macro_f1 raises clear contract error.
    Evidence: .sisyphus/evidence/task-8-metrics-error.txt
  ```

  **Commit**: YES | Message: `feat(eval): make macro f1 the primary training metric` | Files: `src/train/evaluator.py`, `src/train/trainer.py`, `tests/train/test_metrics_contract.py`, `tests/test_evaluator.py`

- [ ] 9. Reproduce a trusted baseline run on grouped split

  **What to do**: Run the canonical pipeline with locked 162-dim basic schema, grouped split, train-only augmentation, and Macro F1 primary metric. Save baseline artifacts under ignored output locations and commit only a small baseline summary/report if appropriate. The baseline is the comparison point for Task 10.
  **Must NOT do**: Do not tune hyperparameters in this task. Do not use leaked/random split as primary baseline. Do not commit large checkpoints/results.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: execution/verification of baseline training workflow.
  - Skills: `data-pipeline-safety` - Validate split/augmentation invariants before trusting metrics.
  - Omitted: `hpo-search-stability` - No search; fixed baseline.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 10 | Blocked By: 1, 2, 3, 4, 5, 6, 7, 8

  **References**:
  - Pattern: `src/train/config.py:12-182` - config presets/default assumptions.
  - Pattern: `src/core/models.py:22-289` - GRU/MLP model registry.
  - Pattern: `requirements.txt` - environment parity baseline.
  - Output: `models/`, `results/`, `artifacts/runs/` are ignored per AGENTS.md.

  **Acceptance Criteria**:
  - [ ] `python -m src.cli.train --dataset tsl51_user_sign --model-type gru --split-strategy video_family_grouped --primary-metric macro_f1 --feature-level basic --output-dir artifacts/runs/baseline-smoke --epochs 1` completes for smoke mode or documented small config.
  - [ ] Run writes metrics JSON, split manifest, preprocessing manifest, checkpoint, and label map.
  - [ ] Metrics JSON reports grouped split strategy and Macro F1.
  - [ ] Baseline summary records command, seed, dataset, feature schema, commit hash if available, and artifact paths.

  **QA Scenarios**:
  ```
  Scenario: Baseline smoke run completes with trusted artifacts
    Tool: Bash
    Steps: python -m src.cli.train --dataset tsl51_user_sign --model-type gru --split-strategy video_family_grouped --primary-metric macro_f1 --feature-level basic --output-dir artifacts/runs/baseline-smoke --epochs 1
    Expected: Exit code 0; required artifacts exist and metrics JSON includes macro_f1 and split_strategy=video_family_grouped.
    Evidence: .sisyphus/evidence/task-9-baseline-run.txt

  Scenario: Baseline rejects random primary split in real-world mode
    Tool: Bash
    Steps: python -m src.cli.train --dataset tsl51_user_sign --split-strategy random --primary-metric macro_f1 --real-world-mode --epochs 1
    Expected: Non-zero exit or clear error explaining grouped split is required for real-world mode.
    Evidence: .sisyphus/evidence/task-9-baseline-error.txt
  ```

  **Commit**: NO | Message: `n/a` | Files: ignored run artifacts only; commit only small summary if project convention accepts it

- [ ] 10. Run controlled GRU/MLP tuning under trusted Macro F1 gate

  **What to do**: Compare a small, fixed candidate set: MLP basic baseline, GRU sequence model, and one GRU regularization variant. Use grouped split Macro F1 as the gate. Keep search manual/fixed, not heavy HPO. If Optuna or persisted studies are introduced later, isolate study names by device/search-space fingerprint and clamp runtime constraints after suggestions.
  **Must NOT do**: Do not tune before Task 9 trusted baseline exists. Do not use random split as the primary metric. Do not add heavy HPO dependencies now. Do not expand to Hybrid/Transformer/CTC in this plan.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: model comparison must be statistically honest and reproducible.
  - Skills: `hpo-search-stability`, `data-pipeline-safety` - Guard against unstable search state and leaked evaluation.
  - Omitted: `ml-backend-integration` - Existing model registry is sufficient for fixed candidates.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: final verification | Blocked By: 9

  **References**:
  - API/Type: `src/core/models.py:22-289` - GRU/MLP model architectures and registry.
  - Pattern: `src/train/config.py:12-182` - config presets to extend for fixed candidates.
  - Pattern: `src/train/trainer.py:109-462` - optimization loop and early stopping.
  - Pattern: `src/train/evaluator.py:37-250` - Macro F1/top-k/per-class metrics.

  **Acceptance Criteria**:
  - [ ] Candidate report includes command/config for each run, seed, grouped Macro F1, secondary metrics, and artifact paths.
  - [ ] Winning tuned model improves grouped Macro F1 by ≥2.0 percentage points over Task 9 baseline, OR report selects the simpler baseline with rationale and no regression.
  - [ ] All runs use the same split manifest and feature/preprocessing schema.
  - [ ] No new dependency is added for HPO.

  **QA Scenarios**:
  ```
  Scenario: Fixed candidate comparison uses same trusted split
    Tool: Bash
    Steps: python -m pytest tests/train/test_tuning_contract.py::test_candidate_runs_share_split_manifest -q
    Expected: Exit code 0; all candidate result fixtures point to identical split manifest hash.
    Evidence: .sisyphus/evidence/task-10-tuning-contract.txt

  Scenario: Tuning report rejects non-improving complex model
    Tool: Bash
    Steps: python -m pytest tests/train/test_tuning_contract.py::test_non_improving_complex_model_is_not_selected -q
    Expected: Exit code 0; complex model below +2pp threshold is not marked winner.
    Evidence: .sisyphus/evidence/task-10-tuning-contract-error.txt
  ```

  **Commit**: YES | Message: `feat(train): add trusted model comparison workflow` | Files: `src/train/config.py`, `tests/train/test_tuning_contract.py`

- [ ] 11. Add minimal notebook validation for Colab workflow

  **What to do**: Add lightweight validation that notebook JSON parses and required canonical imports/calls exist in `colab/02_train.ipynb` and `colab/03_evaluate.ipynb`. Use pytest-based JSON inspection only; add no notebook execution dependency.
  **Must NOT do**: Do not execute notebooks in CI. Do not require Google Drive/GPU. Do not add papermill or heavy notebook infrastructure.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: bounded validation around notebooks after CLI migration.
  - Skills: `tdd` - Add tests for required notebook cells/imports.
  - Omitted: `data-pipeline-safety` - No data processing changes.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: final verification | Blocked By: 6

  **References**:
  - Pattern: `colab/README.md` - notebook workflow docs, no validation runner found.
  - Pattern: `colab/02_train.ipynb` - training notebook to validate.
  - Pattern: `colab/03_evaluate.ipynb` - evaluation notebook to validate.
  - New: `tests/colab/test_notebook_contract.py`

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests/colab/test_notebook_contract.py -q` passes.
  - [ ] Test asserts training notebook imports/calls canonical `src.train` pipeline.
  - [ ] Test asserts evaluation notebook expects result JSON contract from Task 8.
  - [ ] No notebook execution or Google Colab runtime is required.

  **QA Scenarios**:
  ```
  Scenario: Training notebook references canonical pipeline
    Tool: Bash
    Steps: python -m pytest tests/colab/test_notebook_contract.py::test_train_notebook_uses_canonical_pipeline -q
    Expected: Exit code 0; notebook JSON contains canonical import/call and no duplicated legacy train loop marker.
    Evidence: .sisyphus/evidence/task-11-notebook.txt

  Scenario: Broken notebook JSON fails validation
    Tool: Bash
    Steps: python -m pytest tests/colab/test_notebook_contract.py::test_invalid_notebook_contract_fails_with_clear_message -q
    Expected: Exit code 0; fixture invalid notebook produces clear assertion failure message in test.
    Evidence: .sisyphus/evidence/task-11-notebook-error.txt
  ```

  **Commit**: YES | Message: `test(colab): validate notebook training contract` | Files: `colab/02_train.ipynb`, `colab/03_evaluate.ipynb`, `tests/colab/test_notebook_contract.py`

- [ ] 12. Tighten CI markers and fast/slow QA boundaries

  **What to do**: Make pytest markers meaningful. Fast unit/contract tests run in normal CI. Tiny training smoke can be marked `integration` if runtime requires, but must remain CPU-only and runnable by agents. Manual/video-data tests must be marked/skipped clearly or converted to fixture-based tests.
  **Must NOT do**: Do not hide critical split/schema/manifest tests behind slow markers. Do not make CI depend on full HuggingFace downloads or GPU.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: test organization and CI command adjustments.
  - Skills: `tdd` - Verify marker behavior with pytest commands.
  - Omitted: `hpo-search-stability` - Not relevant.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: final verification | Blocked By: 7

  **References**:
  - Config: `pyproject.toml:125-152` - pytest markers/testpaths.
  - CI: `.github/workflows/ci.yml` - current ruff/mypy/pytest workflow.
  - Hooks: `.pre-commit-config.yaml` - local pytest hook.
  - Tests: `tests/test_model_with_videos.py`, `tests/smoke_inference.py` - manual/data-dependent areas.

  **Acceptance Criteria**:
  - [ ] `python -m pytest tests -q` passes without full dataset, GPU, or local checkpoints.
  - [ ] `python -m pytest -m integration tests -q` runs CPU-only integration smoke tests or skips with clear reasons.
  - [ ] CI command remains compatible with Windows Python 3.10-3.12.
  - [ ] Critical split/schema/manifest tests are unmarked fast tests.

  **QA Scenarios**:
  ```
  Scenario: Fast CI suite has no external data dependency
    Tool: Bash
    Steps: python -m pytest tests -q
    Expected: Exit code 0; no failures due to missing D:\TSL\TSL\models, full dataset, GPU, or Colab runtime.
    Evidence: .sisyphus/evidence/task-12-fast-ci.txt

  Scenario: Integration marker is meaningful
    Tool: Bash
    Steps: python -m pytest -m integration tests -q
    Expected: Exit code 0 or explicit skips only; no unmarked long/full-data tests run accidentally.
    Evidence: .sisyphus/evidence/task-12-integration-ci.txt
  ```

  **Commit**: YES | Message: `test(ci): define reliable training qa boundaries` | Files: `pyproject.toml`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `tests/*`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright only if web inference touched; otherwise CLI/pytest QA)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit per task when acceptance criteria pass.
- Do not commit large `models/`, `results/`, or `artifacts/` outputs.
- Suggested branch commits:
  1. `feat(train): add leakage-safe grouped splits`
  2. `fix(train): isolate augmentation to training split`
  3. `refactor(core): centralize basic feature schema`
  4. `feat(train): persist preprocessing manifests`
  5. `refactor(train): make src training pipeline canonical`
  6. `feat(cli): route training through canonical pipeline`
  7. `test(train): add tiny training smoke coverage`
  8. `feat(eval): make macro f1 the primary training metric`
  9. `feat(train): add trusted model comparison workflow`
  10. `test(colab): validate notebook training contract`
  11. `test(ci): define reliable training qa boundaries`

## Success Criteria
- Training accuracy claims are based on video-family-held-out grouped split Macro F1, not leaked/random split metrics.
- Augmented expert/full samples are train-only.
- `src/train/` is the canonical active training implementation for CLI and Colab.
- Feature/preprocessing manifest makes training and inference use the same 162-dim schema and normalization stats.
- Fast pytest suite runs without full dataset, GPU, local checkpoints, or manual Colab steps.
- Tuned model improves grouped Macro F1 by ≥2.0 percentage points over trusted baseline, or simpler baseline is selected with evidence.
