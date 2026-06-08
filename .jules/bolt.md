## 2024-05-18 - Pandas Data Extraction Bottleneck
**Learning:** Iterating over columns and individually computing `.mean()` using `safe_mean` on a Pandas DataFrame in a hot path causes significant overhead (~1.04s per 100 calls for 162 columns) because of Python iteration and repeatedly entering/exiting C-level pandas code.
**Action:** Replace iterative column access with vectorized pandas operations like `lm_df[cols].mean().fillna(0.0).to_dict()` whenever computing aggregate statistics across many columns. This yields a ~10x speedup while preserving missing-value fallback semantics.
