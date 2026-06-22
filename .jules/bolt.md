## 2025-06-22 - [Optimized DataFrame extraction and sequence extraction]
**Learning:** `extract_features_from_landmark_df` and `extract_sequence_from_landmark_df` had major bottlenecks due to repeated column fetching logic (`safe_mean(lm_df[col])`) and iterating column-by-column for DataFrame mapping.
**Action:** Use vectorized DataFrame properties (`lm_df[cols].mean()`) and cache `_build_column_list`. Reduced time for `extract_features_from_landmark_df` on full sequence from ~5s to ~0.15s (30x speedup), and `extract_sequence_from_landmark_df` from ~6s to ~1.3s.
