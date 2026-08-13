## 2023-10-18 - [Missing Metrics in FakeTrainer]
**Learning:** When mocking training components in tests like `tests/train/test_pipeline_contract.py`, `FakeTrainer` mock objects must return a complete set of validation metrics (including `val_precision`, `val_recall`, `val_top3_acc`, `val_top5_acc`, `per_class_metrics`, and `confusion_matrix`) to comply with the `evaluator.py` result contract and prevent pipeline test failures.
**Action:** When updating mocked testing configurations or returning results from training mocks, ensure all required metrics are present in the returned dictionary.
