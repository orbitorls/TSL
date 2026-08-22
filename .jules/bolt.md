## 2024-05-24 - Pandas Vectorization
**Learning:** Extracting features using row/column iteration with safe_mean is slow and unoptimized.
**Action:** Use Pandas vectorized operations like .mean(numeric_only=True).fillna(0.0).to_dict() or just direct .to_numpy() instead of slow manual string formatting and iterative calls.
