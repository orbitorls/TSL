## 2024-05-18 - Vectorized Pandas operations vs. Iteration
**Learning:** In `src/data/feature_extraction.py`, extracting mean features iteratively from a Pandas DataFrame (e.g., using `safe_mean(lm_df[col])` in a for loop across thousands of columns) was a significant performance bottleneck.
**Action:** When computing aggregations over many columns in Pandas, use vectorized operations (e.g., `lm_df[cols].mean().fillna(0.0).to_dict()`) and avoid iterative scalar extraction or slow fallback function calls.
