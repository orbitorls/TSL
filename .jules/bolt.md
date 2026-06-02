## 2024-05-18 - Vectorizing Pandas Operations for 10x Speedup
**Learning:** In data extraction loops (`extract_features` in `src/core/features.py`), iteratively calling `.mean()` or iterating column-by-column on Pandas DataFrames is a major performance bottleneck due to Pandas' per-call scalar overhead.
**Action:** When aggregating multiple columns, use vectorized operations like `df[cols].mean().fillna(0.0).to_dict()` combined with caching the feature keys. This simple change provided a nearly ~10x performance improvement (~10.8s down to ~0.9s for 1000 iterations).
