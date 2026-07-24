## 2025-02-28 - Pandas Dataframe extraction bottlenecks
**Learning:** Extracting features column-by-column from Pandas DataFrames (especially using iterative `.mean()` or `.to_numpy()` for 162+ dimensions) introduces significant Python iteration and Pandas series-creation overhead.
**Action:** When extracting sequence features or aggregated features from DataFrames, always prefer bulk assignment operations. Identify available columns using list comprehensions, and use advanced indexing for means and sequence arrays. This drops extraction time by >90%.
