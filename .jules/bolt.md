## 2024-05-24 - Pandas Series Lookups within Loops Cause Extreme Bottlenecks

**Learning:** Iterating over columns to calculate statistics using Python loops over Pandas series (e.g., calling `safe_mean(lm_df[col])` in a loop) incurs severe overhead due to repeated method dispatching and internal Pandas checks. For features with many columns (like the 'full' extraction using 1596 columns), this translates to an incredible penalty (e.g., ~10 seconds for 100 loops of a small dataframe).

**Action:** Whenever applying statistics (like `mean`, `sum`, `std`) across numerous columns in Pandas, always leverage vectorized operations (e.g. `lm_df[cols].mean()`). This approach avoids iterating in Python and delegates the loop to optimized C backend structures, yielding massive speedups (observed a ~20x performance improvement).
