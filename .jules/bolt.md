## 2024-05-24 - Pandas Vectorization
**Learning:** Iterating over columns in pandas DataFrames and using Series methods per column is a significant bottleneck, particularly with high-dimensional feature spaces (e.g., 1434 features).
**Action:** Use vectorized DataFrame operations like `df[cols].mean(numeric_only=True)` and bulk assignment (`seq[:, col_indices] = df[cols].to_numpy()`) instead of loops to gain order-of-magnitude speedups.
