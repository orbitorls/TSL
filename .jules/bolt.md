## 2024-05-24 - Bulk Vector Assignment for Pandas Features
**Learning:** Extracting sequence features from a Pandas DataFrame column-by-column in a loop is extremely slow due to Pandas overhead.
**Action:** When extracting multiple columns into a NumPy array, identify available columns via `lm_df.columns.intersection(cols)` and use bulk assignment (`lm_df[available_cols].values`) for significant performance gains.
