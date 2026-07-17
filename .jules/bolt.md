## 2024-05-24 - Pandas Series Iteration Bottleneck
**Learning:** Extracting features using a Python `for` loop that iterates over hundreds of column names and individually computes `safe_mean(lm_df[col])` creates massive Python-level overhead and nullifies Pandas' C-level performance advantages.
**Action:** When extracting multiple features from a DataFrame, always build an intersection of required columns and use a single vectorized operation: `lm_df[available_cols].mean(numeric_only=True).fillna(0.0).to_dict()`, then fetch results using standard dict lookups.
