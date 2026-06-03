# TSL-51 Inference Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify why the current TSL-51 model predicts the wrong signs, apply the smallest effective fix using existing artifacts first, and only retrain if all valid artifacts still perform badly.

**Architecture:** Start from the existing runtime selection path (`tsl_translate.registry` + `webcam_runtime.load_demo_artifacts`) and gather evidence about which artifact is chosen and whether it satisfies the TSL-51 contract. Then compare artifact metadata and external-eval evidence to decide between a runtime-selection fix, an artifact switch, or a retraining step.

**Tech Stack:** Python, pytest, TensorFlow/Keras, joblib, MediaPipe, JSON manifests, existing repo scripts under `scripts/`

---

## File structure

- `python-legacy/src/tsl_translate/registry.py` — artifact discovery and ranking for TSL-51 runtime selection.
- `python-legacy/src/webcam_runtime.py` — artifact contract validation for labels, scaler, input shape, output shape, and manifest track.
- `python-legacy/tests/test_tsl_translate_runtime.py` — existing runtime and registry regression tests; best place to add focused TSL-51 selection/eligibility tests.
- `scripts/report_tsl51_goal_readiness.py` — summarizes candidate artifacts and current external-eval evidence.
- `scripts/evaluate_tsl51_video.py` — runs offline evaluation on labeled TSL-51 clips.
- `reports/tsl51_goal_readiness.json` — existing evidence that current internal candidate passes internal accuracy but external eval is 0.0 and ready=false.
- `docs/superpowers/specs/2026-06-03-tsl51-inference-recovery-design.md` — approved design source for this plan.

### Task 1: Capture current evidence and chosen artifact

**Files:**
- Modify: none
- Test: none
- Output: `reports/tsl51_goal_readiness.json` (existing report, regenerated if needed)

- [ ] **Step 1: Confirm the current readiness report and candidate list**

Run:
```bash
python scripts/report_tsl51_goal_readiness.py \
  --artifact-dir ".tools/tsl51_experiments/full51_v2/artifacts/tsl51" \
  --artifact-dir ".tools/tsl51_experiments/full51_v3_external_weighted/artifacts/tsl51" \
  --external-report "reports/current_reviewed_external_eval/full51_v2/summary.json" \
  --reviewed-split-summary "reports/reviewed_external_split_summary.json" \
  --review-queue-summary "reports/tsl51_review_queue_summary.json" \
  --review-apply-summary "reports/tsl51_review_apply_summary.json" \
  --review-validate-summary "reports/tsl51_review_validate_summary.json" \
  --reviewed-assets-summary "reports/reviewed_external_assets_summary.json" \
  --strategy-summary "reports/current_reviewed_external_eval/strategy_summary.json" \
  --out "reports/tsl51_goal_readiness.json"
```
Expected: output prints `ready=False` and shows a valid internal candidate plus missing trusted external-holdout pass.

- [ ] **Step 2: Record the exact current problem from the report**

Check these fields in `reports/tsl51_goal_readiness.json`:
```json
{
  "internal_pass": true,
  "external_holdout_pass": false,
  "ready": false,
  "best_internal_candidate": {
    "artifact_dir": ".tools\\tsl51_experiments\\full51_v2\\artifacts\\tsl51"
  },
  "strategy_summary": {
    "all_strategies_failed": true
  }
}
```
Expected: internal accuracy looks good, but every current external strategy still fails.

- [ ] **Step 3: Commit the evidence refresh only if the report changed**

```bash
git add reports/tsl51_goal_readiness.json
git commit -m "chore: refresh tsl51 readiness evidence"
```
Expected: either a small report-only commit or no commit if nothing changed.

### Task 2: Add a failing regression test for artifact ranking preference

**Files:**
- Modify: `python-legacy/tests/test_tsl_translate_runtime.py`
- Test: `python-legacy/tests/test_tsl_translate_runtime.py`

- [ ] **Step 1: Write the failing test**

Add this test near the registry tests in `python-legacy/tests/test_tsl_translate_runtime.py`:
```python
def test_tsl51_registry_prefers_candidate_with_external_holdout_evidence(tmp_path: Path) -> None:
    baseline = tmp_path / ".tools" / "tsl51_experiments" / "baseline" / "artifacts" / "tsl51"
    external = tmp_path / ".tools" / "tsl51_experiments" / "external" / "artifacts" / "tsl51"
    write_artifact_stub(baseline, 51)
    write_artifact_stub(external, 51)
    write_manifest(baseline, test_accuracy=0.92, external_augmented=False)
    write_manifest(external, test_accuracy=0.91, external_augmented=True, external_val_samples=8)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert [candidate.name for candidate in candidates] == [
        ".tools\\tsl51_experiments\\external\\artifacts\\tsl51",
        ".tools\\tsl51_experiments\\baseline\\artifacts\\tsl51",
    ]
```

- [ ] **Step 2: Run the single test to confirm the current behavior fails**

Run:
```bash
cd python-legacy && python -m pytest tests/test_tsl_translate_runtime.py::test_tsl51_registry_prefers_candidate_with_external_holdout_evidence -q
```
Expected: FAIL because `_candidate_rank()` currently prefers clean baseline artifacts over externally validated artifacts.

- [ ] **Step 3: Commit only after the implementation in Task 3 passes**

```bash
git add python-legacy/tests/test_tsl_translate_runtime.py python-legacy/src/tsl_translate/registry.py
git commit -m "fix: prefer validated tsl51 runtime artifacts"
```
Expected: do not run this yet; hold until Task 3 is green.

### Task 3: Implement the minimal registry ranking fix

**Files:**
- Modify: `python-legacy/src/tsl_translate/registry.py`
- Test: `python-legacy/tests/test_tsl_translate_runtime.py`

- [ ] **Step 1: Change ranking to reward external holdout evidence when eligible**

Update `_candidate_rank()` in `python-legacy/src/tsl_translate/registry.py` so it ranks candidates in this order:
```python
def _candidate_rank(candidate: ArtifactCandidate, track: TrackSpec) -> tuple[int, int, float, float]:
    manifest = _manifest_json(candidate.manifest) if candidate.manifest else {}
    external_validated = 0
    clean_rank = 1
    if track.key == "tsl51":
        if manifest.get("external_augmented") is True:
            clean_rank = 0
            if int(manifest.get("external_val_samples") or 0) > 0:
                external_validated = 1
    try:
        accuracy = float(manifest.get("test_accuracy") or 0.0)
    except (TypeError, ValueError):
        accuracy = 0.0
    try:
        modified = candidate.model.stat().st_mtime
    except OSError:
        modified = 0.0
    return external_validated, clean_rank, accuracy, modified
```
This keeps the existing eligibility filter but stops punishing a TSL-51 artifact that actually has external validation evidence.

- [ ] **Step 2: Run the targeted runtime tests**

Run:
```bash
cd python-legacy && python -m pytest tests/test_tsl_translate_runtime.py -q
```
Expected: PASS, including the new preference test and the existing class-count / eligibility tests.

- [ ] **Step 3: Create the code commit**

```bash
git add python-legacy/src/tsl_translate/registry.py python-legacy/tests/test_tsl_translate_runtime.py
git commit -m "fix: prefer validated tsl51 runtime artifacts"
```
Expected: commit created with only the registry/test change.

### Task 4: Re-run offline evaluation against the currently selected artifact and the best alternate artifact

**Files:**
- Modify: none unless later tasks require code changes
- Test: script output under `reports/`

- [ ] **Step 1: Evaluate the baseline artifact on the reviewed external samples**

Run:
```bash
python scripts/evaluate_tsl51_video.py \
  --samples "work/reviewed_external_assets/external_test_samples.csv" \
  --artifact-dir ".tools/tsl51_experiments/full51_v2/artifacts/tsl51" \
  --out-dir "reports/current_reviewed_external_eval/full51_v2_recheck"
```
Expected: summary JSON is produced; top-1 accuracy confirms whether the failure is reproducible.

- [ ] **Step 2: Evaluate the alternate artifact if it exists and is contract-valid**

Run:
```bash
python scripts/evaluate_tsl51_video.py \
  --samples "work/reviewed_external_assets/external_test_samples.csv" \
  --artifact-dir ".tools/tsl51_experiments/full51_v3_external_weighted/artifacts/tsl51" \
  --out-dir "reports/current_reviewed_external_eval/full51_v3_external_weighted_recheck"
```
Expected: summary JSON is produced for the alternate artifact so the choice is evidence-based.

- [ ] **Step 3: Compare the two summaries directly**

Check these files:
```text
reports/current_reviewed_external_eval/full51_v2_recheck/summary.json
reports/current_reviewed_external_eval/full51_v3_external_weighted_recheck/summary.json
```
Expected: either one artifact clearly wins, or both remain poor and point toward retraining.

### Task 5: Encode the winning artifact choice in runtime behavior

**Files:**
- Modify: `python-legacy/src/tsl_translate/registry.py` only if Task 4 proves the alternate artifact should be preferred
- Test: `python-legacy/tests/test_tsl_translate_runtime.py`

- [ ] **Step 1: If Task 4 showed the alternate artifact wins, add/adjust one test that matches the real manifest values**

Use a test shape like this in `python-legacy/tests/test_tsl_translate_runtime.py`:
```python
def test_tsl51_registry_prefers_externally_validated_candidate_over_cleaner_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / ".tools" / "tsl51_experiments" / "full51_v2" / "artifacts" / "tsl51"
    validated = tmp_path / ".tools" / "tsl51_experiments" / "full51_v3_external_weighted" / "artifacts" / "tsl51"
    write_artifact_stub(baseline, 51)
    write_artifact_stub(validated, 51)
    write_manifest(baseline, test_accuracy=0.9245, external_augmented=False)
    write_manifest(validated, test_accuracy=0.9746, external_augmented=True, external_val_samples=8)

    candidates = ModelRegistry(tmp_path).discover(TRACKS["tsl51"])

    assert candidates[0].name.endswith("full51_v3_external_weighted\\artifacts\\tsl51")
```
If Task 4 shows the baseline still wins, skip this step and keep the Task 3 test as the only code regression.

- [ ] **Step 2: Run the exact targeted test set again**

Run:
```bash
cd python-legacy && python -m pytest tests/test_tsl_translate_runtime.py -q
```
Expected: PASS with the final ranking behavior.

- [ ] **Step 3: Commit only if code changed after Task 4**

```bash
git add python-legacy/src/tsl_translate/registry.py python-legacy/tests/test_tsl_translate_runtime.py
git commit -m "fix: lock tsl51 runtime to the best validated artifact"
```
Expected: skip if no additional code change was needed.

### Task 6: Decide whether retraining is required

**Files:**
- Modify: none for the decision itself
- Test: summary outputs

- [ ] **Step 1: Stop and inspect the evaluation outputs before training**

Review these fields in the generated summary JSON:
```json
{
  "total_samples": 5,
  "top1_accuracy": 0.0,
  "top3_accuracy": 0.0,
  "artifact_dir": "..."
}
```
Expected: if both artifacts remain near 0.0, the issue is model quality, not selection.

- [ ] **Step 2: If at least one artifact is clearly better, do not retrain**

Decision rule:
```text
If one contract-valid artifact materially improves top1/top3 accuracy over the current choice,
stop here and keep the runtime-selection fix only.
```
Expected: this is the preferred outcome.

- [ ] **Step 3: If all valid artifacts are still poor, prepare retraining inputs**

Run:
```bash
python scripts/report_tsl51_inventory.py \
  --metadata "data/tsl51/metadata/train.csv" \
  --labels ".tools/tsl51_experiments/full51_v2/artifacts/tsl51/tsl51_labels.json" \
  --out "reports/tsl51_inventory_recheck.json"
```
Expected: a fresh inventory report confirms class coverage before retraining.

### Task 7: Retrain only if Task 6 proves existing artifacts are not usable

**Files:**
- Modify: none unless a script bug is found first
- Test: new artifact output and follow-up evaluation

- [ ] **Step 1: Run a dry-run of the external-augmentation training path first**

Run:
```bash
python scripts/train_external_augmented.py \
  --track tsl51 \
  --base-cache "reports/external_benchmark/train_smoke_tsl51/train_tsl51.npz" \
  --external-cache "reports/external_benchmark/train_smoke_tsl51/tsl51_combined_external_augmented.npz" \
  --external-cache-split train \
  --out-dir ".tools/tsl51_experiments/retrain_candidate/artifacts/tsl51" \
  --base-artifact-dir ".tools/tsl51_experiments/full51_v2/artifacts/tsl51" \
  --dry-run
```
Expected: dry-run passes argument validation and exposes missing data/setup issues before expensive training.

- [ ] **Step 2: Run the real retraining command only after the dry-run is clean**

Run:
```bash
python scripts/train_external_augmented.py \
  --track tsl51 \
  --base-cache "reports/external_benchmark/train_smoke_tsl51/train_tsl51.npz" \
  --external-cache "reports/external_benchmark/train_smoke_tsl51/tsl51_combined_external_augmented.npz" \
  --external-cache-split train \
  --out-dir ".tools/tsl51_experiments/retrain_candidate/artifacts/tsl51" \
  --base-artifact-dir ".tools/tsl51_experiments/full51_v2/artifacts/tsl51"
```
Expected: new `tsl51_model.keras`, `tsl51_labels.json`, `tsl51_scaler.pkl`, and manifest are written.

- [ ] **Step 3: Evaluate the retrained artifact immediately**

Run:
```bash
python scripts/evaluate_tsl51_video.py \
  --samples "work/reviewed_external_assets/external_test_samples.csv" \
  --artifact-dir ".tools/tsl51_experiments/retrain_candidate/artifacts/tsl51" \
  --out-dir "reports/current_reviewed_external_eval/retrain_candidate"
```
Expected: retraining is only acceptable if this beats the prior artifact evidence.

- [ ] **Step 4: Commit only the code changes, not generated model binaries**

```bash
git add python-legacy/src/tsl_translate/registry.py python-legacy/tests/test_tsl_translate_runtime.py reports/tsl51_goal_readiness.json docs/superpowers/specs/2026-06-03-tsl51-inference-recovery-design.md docs/superpowers/plans/2026-06-03-tsl51-inference-recovery-plan.md
git commit -m "fix: improve tsl51 inference artifact selection"
```
Expected: do not add `.keras`, `.pkl`, `.npz`, or other generated artifacts.

## Self-review

- Spec coverage: tasks cover runtime selection, artifact integrity evidence, offline comparison, minimal fix, and retraining escalation only if needed.
- Placeholder scan: all command steps, tests, and code snippets are concrete; conditional steps explicitly state when to skip.
- Type consistency: all referenced functions and files exist in the inspected codebase (`ModelRegistry`, `write_artifact_stub`, `write_manifest`, `report_tsl51_goal_readiness.py`, `evaluate_tsl51_video.py`, `train_external_augmented.py`).
