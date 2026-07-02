## 2025-03-03 - [Vectorized mean in Pandas data loading routines]
**Learning:** In python scripts processing pandas DataFrames repeatedly, iteratively slicing columns and invoking custom math functions (like `safe_mean(lm_df[col])`) creates significant hidden overhead.
**Action:** When extracting mathematical aggregations over dataframes where columns match a schema, pre-filter for available columns (`avail_cols = [c for c in SCHEMA if c in df.columns]`) and run native vectorized methods like `df[avail_cols].mean(numeric_only=True).fillna(0.0).to_dict()` instead of writing loop-level logic.
