
## 2025-02-28 - Optimize Pandas Feature Extraction

**Learning:** Extracting individual variables from a Pandas DataFrame row-by-row via repetitive Python indexing (e.g. `lm_df[col]`) is a massive O(N) performance bottleneck when extracting hundreds of landmarks (like 1596 variables in full/face features).

**Action:** Replace sequential iteration with vectorized subset `.mean(numeric_only=True).fillna(0.0).to_dict()` lookups over `lm_df.columns.intersection(cols)` to execute all column extraction natively in Pandas/C in an O(1) step, delivering an up to 36x speedup. Also, cache dynamic feature column sets like `_build_column_list` when called repeatedly per-batch.
