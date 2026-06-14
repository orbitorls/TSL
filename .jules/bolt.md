## 2024-06-14 - Feature Extraction Pandas Loop Bottleneck
**Learning:** `safe_mean(lm_df[col])` inside a Python `for` loop over 1500+ DataFrame columns creates massive overhead due to repeated Pandas series access and unvectorized scalar aggregations. This made `extract_features_from_landmark_df` painfully slow (~1 second per sequence).
**Action:** Always compute cross-column aggregations on Pandas DataFrames using natively vectorized methods (e.g. `lm_df[cols].mean().fillna(0.0).to_dict()`) rather than looping column-by-column in Python.

## 2024-06-14 - Real-Time Landmark Dict Lookup Overhead
**Learning:** Converting per-frame dictionaries to feature arrays in `_frame_dict_to_vector` using on-the-fly string formatting (e.g., `f"lh_{c}{i}"`) inside nested loops takes substantial time and creates a real-time CPU bottleneck for video processing.
**Action:** Use pre-computed, static lists of keys (like `_BASIC_KEYS`) and simple list comprehensions (`[frame.get(k, 0.0) for k in _BASIC_KEYS]`) to bypass dynamic string construction in performance-critical inner loops.
