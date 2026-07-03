## 2024-07-03 - [Pandas Vectorization over iterative loops]
**Learning:** Extracting data column by column using `safe_mean(df[col])` in a loop across 162+ features is extremely slow in pandas. We measured a >10x difference.
**Action:** Always use vectorized operations like `df[cols].mean(numeric_only=True)` when processing multiple columns from a DataFrame simultaneously instead of a Python `for` loop.
