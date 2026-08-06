## 2024-05-19 - Pandas Vectorization in Feature Extraction
**Learning:** Iterating over columns and using `safe_mean` repeatedly in Python is a massive performance bottleneck. Vectorized execution pushes these operations down to C level.
**Action:** Always prefer `df[cols].mean()` and advanced NumPy indexing instead of python iteration when processing DataFrame rows and sequences.
