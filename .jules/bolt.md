## 2024-07-11 - DataFrame Feature Extraction Bottleneck
**Learning:** In pandas DataFrames containing 1000+ landmark features, performing `.mean()` iteratively on individual columns is a major bottleneck (O(N) operations with heavy pandas dispatch overhead per column).
**Action:** Replace iterative series computations with batched vectorized pandas operations like `df[cols].mean().to_dict()` and cache constructed feature column lists.
