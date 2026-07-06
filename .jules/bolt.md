
## 2024-02-15 - [Pandas Vectorized Extract Features]
**Learning:** Extracting features column-by-column iteratively in pandas is highly inefficient for large schemas (e.g., 162-1596 dims). Caching column lists globally and performing vectorized calculations (`lm_df[cols].mean()`) offers significant performance improvements.
**Action:** When working with structured DataFrames, identify and eliminate slow iterative loops (`for col in cols: val = df[col].mean()`). Favor vectorized array methods (like `df.mean().to_dict()`) and module-level constants to prevent redundant allocations during hot-paths like inference loops. Avoid polluting `.lock` and `pyproject.toml` dependencies to satisfy downstream mocks for unrelated module tests.
