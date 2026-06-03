# TSL-51 inference recovery design

Date: 2026-06-03
Status: approved for investigation

## Goal
Improve the current TSL-51 runtime so predictions stop being consistently wrong, starting with the existing artifacts and runtime path before retraining.

## Scope
In scope:
- Identify which TSL-51 artifact the app/runtime selects today.
- Verify artifact contract correctness: model file, labels, scaler, manifest, class count, input feature dimension, sequence length.
- Compare available TSL-51 artifacts and determine whether a better existing candidate should be preferred.
- Run focused evaluation/inspection to distinguish runtime-selection problems from true model-quality problems.
- Apply the smallest effective fix if the issue is caused by selection, metadata, or runtime wiring.
- Escalate to retraining only if current artifacts are structurally valid but still perform poorly.

Out of scope:
- Fingerspelling changes.
- Broad UI refactors.
- Unrelated training-pipeline cleanup.

## Current observations
- The web runtime reports "ยังไม่โหลดโมเดล" until a model is loaded, but the user-reported issue is wrong predictions rather than load failure.
- TSL-51 expects a strict contract: 51 labels, sequence input, expected feature dimension, and matching scaler/manifest.
- Available candidates currently exist under `artifacts/tsl51/`, `artifacts/tsl51-yt-augmented/`, and `artifacts/tsl51-yt-augmented-v2/`.
- Artifact discovery ranks candidates partly by cleanliness and manifest accuracy, so wrong behavior may come from candidate selection as well as model quality.

## Approach options considered
1. Inspect current artifacts and runtime first. Recommended because it can quickly reveal wrong candidate selection, incomplete class coverage, or mismatched metadata.
2. Swap to a different existing artifact immediately. Lower effort, but only valid after verifying why the current one is bad.
3. Retrain immediately. Highest cost; only justified if existing artifacts are valid yet underperform.

## Chosen approach
Start with evidence gathering on the current TSL-51 runtime and artifacts, then branch:
- If the runtime is choosing the wrong artifact or validating against the wrong metadata, fix that directly.
- If another existing artifact is demonstrably better and contract-valid, switch selection to that artifact.
- If all valid artifacts perform poorly, move to retraining with the existing training scripts.

## Investigation design
### 1. Reproduce and inspect runtime selection
- Identify which TSL-51 artifact candidate is selected by the registry.
- Inspect manifest values, label count, and ranking behavior.
- Confirm the runtime track, expected class count, input feature dimension, and sequence length.

### 2. Verify artifact integrity
For each TSL-51 candidate:
- Check presence of required files.
- Check label cardinality equals 51.
- Check scaler feature count matches runtime expectations.
- Check model input/output shapes align with the TSL-51 contract.
- Check manifest values that affect eligibility and ranking.

### 3. Evaluate quality versus configuration
- Run the existing evaluation/inspection path against the chosen artifact and, if needed, the other available artifacts.
- Compare whether the current production choice is actually the best available one.
- Determine whether wrong translations are caused by systematic class confusion, incomplete coverage, or a bad runtime selection.

### 4. Apply the smallest fix
Possible fixes, in order of preference:
1. Fix candidate selection or manifest-based eligibility/ranking.
2. Point the runtime to a better existing artifact.
3. Repair metadata only if the underlying artifact is otherwise valid.
4. Retrain TSL-51 only if evaluation shows current artifacts are insufficient.

## Testing strategy
- Use existing tests where possible for artifact loading and runtime validation.
- Add a focused failing test only if the identified root cause is not already covered.
- Re-run the relevant TSL-51 checks after any fix.
- If retraining becomes necessary, verify the newly produced artifact with the same contract checks before considering it usable.

## Success criteria
- We can explain why the current TSL-51 predictions are wrong.
- The selected runtime artifact is contract-valid and intentionally chosen.
- Either the current runtime is corrected with a minimal fix, or we have evidence that retraining is necessary.
- After the chosen fix, targeted evaluation shows better TSL-51 behavior than the current state.

## Risks and decisions
- The runtime may be loading a technically valid artifact that is simply low quality; this would require retraining rather than runtime changes.
- Manifest ranking may favor a suboptimal artifact if metadata is stale or misleading.
- Real-world sign performance may still differ from offline evaluation, so artifact contract validation alone is not enough.
