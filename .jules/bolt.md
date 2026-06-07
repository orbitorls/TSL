## 2025-02-28 - Pandas Iterative Series Mean Bottleneck
**Learning:** In Pandas, iterating over DataFrame columns using `series.mean()` element-by-element (like `[df[col].mean() for col in cols]`) causes massive performance bottlenecks due to Python-level loops and Series object creation overhead.
**Action:** Use vectorized operations like `df[cols].mean().to_dict()` when calculating aggregate means across many columns simultaneously to enable underlying C/NumPy speedups.
