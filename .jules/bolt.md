## 2024-05-24 - Efficient DataFrame iteration in sequence extraction
**Learning:** `extract_sequence_from_landmark_df` constructs a numpy array iteratively by looping over columns of a Pandas DataFrame. Since Pandas column indexing inside a loop is slow, this creates a major bottleneck during sequence feature extraction.
**Action:** Identify available columns using `lm_df.columns.intersection(col_list)`, map them to their target indices efficiently using `pd.Index(col_list).get_indexer(available_cols)`, and use bulk assignment via advanced indexing.

## 2024-05-24 - Missing metrics in FakeTrainer mock
**Learning:** `test_pipeline_contract.py`'s `FakeTrainer` was missing essential evaluation metrics (`val_precision`, `val_recall`, `val_top3_acc`, `val_top5_acc`, `per_class_metrics`, `confusion_matrix`) required by `_build_result_metrics_contract`.
**Action:** When mocking training components in tests, mock objects must return a complete set of validation metrics to comply with the `evaluator.py` result contract and prevent pipeline test failures.
