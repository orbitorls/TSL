## 2024-05-18 - Pandas iteration bottleneck in DataFrame parsing
**Learning:** Iterating through columns manually and performing `safe_mean()` over `lm_df[col]` triggers huge pandas Series extraction overhead, increasing execution time geometrically based on column counts (e.g. going up to ~11s for a synthetic loop).
**Action:** Always compute column means over a subset by using vectorized built-in methods like `lm_df[cols_to_use].mean().fillna(0.0).to_dict()`, avoiding thousands of internal method calls.

## 2024-05-18 - Redundant string creation inside loops
**Learning:** Generating the same complex column names continuously using f-strings inside a loop (like `lh_x0`, `lh_y0`) during live feature extraction is a significant hotspot.
**Action:** Pre-generate these fixed keys and cache them locally or at module-level (e.g., `_FEATURE_COLUMN_CACHE`), then reuse them via list slicing `keys[:feature_dim]` whenever possible.
