## 2024-05-18 - [Pandas Extraction Optimization]
**Learning:** Using `safe_mean` iteratively over columns for feature extraction from a pandas DataFrame introduces significant overhead (approx 9x slower). Vectorized operations `df[available_cols].mean(numeric_only=True)` dramatically speed up feature extraction, which is especially important given this can be a bottleneck in data processing/training.
**Action:** Replace iterative feature extraction over DataFrames with Pandas vectorized means, converting the result to a dict for quick ordered feature lookup.
