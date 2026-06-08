## 2024-05-18 - FakeTrainer Missing Validation Metrics in Tests
**Learning:** In `tests/train/test_pipeline_contract.py`, the `FakeTrainer` mock fails to return all metrics required by `evaluator.py`, such as `val_precision`, `val_recall`, `val_top3_acc`, `val_top5_acc`, `per_class_metrics`, and `confusion_matrix`. This causes a `PipelineConfigError: training result contract requires 'val_precision' before checkpoint selection`.
**Action:** Update the `FakeTrainer` mock in the tests to return dummy values for all these missing metrics to ensure it correctly fulfills the contract.
