
## 2024-05-18 - Pandas Iteration Anti-Pattern in Data Extraction
**Learning:** Iterating through expected columns using `safe_mean(lm_df[col])` via a Python loop in feature extraction is a significant performance bottleneck (taking ~9.8s for 100 extractions of face features). Pandas Series operations incur massive overhead when invoked iteratively inside loops.
**Action:** When extracting data from Pandas DataFrames, replace Python iteration with vectorized Pandas bulk operations, such as identifying available columns via `lm_df.columns.intersection(expected_cols)` and computing metrics in bulk using `.mean(numeric_only=True).fillna(0.0).to_dict()`. This provided a ~28x speedup. Cache expected column lists via dictionaries to avoid repeated string concatenation and lookup overhead.
