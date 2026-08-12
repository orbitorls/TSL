## 2024-05-15 - Optimize Pandas extraction via vectorization
**Learning:** Feature extraction from Pandas DataFrames (such as `extract_features` in `src/core/features.py` and `extract_features_from_landmark_df` in `src/data/feature_extraction.py`) is very slow due to repeated calls to `safe_mean(lm_df[col])`. Using Pandas vectorized operations like `lm_df[cols].mean(numeric_only=True).fillna(0.0).to_dict()` provides a 10x-30x speedup.
**Action:** Optimize feature extraction functions by replacing loop-based `safe_mean` calls with vectorized mean computation across valid columns.
