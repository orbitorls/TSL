## 2025-03-01 - [Optimized extract_features_from_landmark_df]
**Learning:** Feature extraction from DataFrame was extremely slow because of repeated pandas series accessing via `safe_mean(lm_df[col])` in python loops. This was generating a lot of overhead.
**Action:** Bulk array assignments and pandas vectorized operations such as `lm_df[available_cols].mean(numeric_only=True).fillna(0.0).to_dict()` are significantly faster.
