## 2024-05-24 - Pandas vectorization in feature extraction
**Learning:** Using `pandas.DataFrame.mean(numeric_only=True).to_dict()` is extremely fast compared to repeatedly calling `.mean()` on individual `pd.Series` columns during sequence and frame feature extraction. Replacing iterative column extraction drops latency per feature call drastically (e.g. from ~100ms for full features to ~3.5ms).
**Action:** Always prefer `df[cols].mean()` over `[df[c].mean() for c in cols]` inside hot data extraction paths when handling pandas DataFrames.
