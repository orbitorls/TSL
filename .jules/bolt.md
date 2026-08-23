## 2024-08-23 - Avoid String Formatting in Hot Paths
**Learning:** String interpolation (e.g. `f"lh_x{i}"`) inside tight nested loops for real-time feature extraction (like `_frame_dict_to_vector`) causes significant overhead. Additionally, dynamically initializing constants using helper functions can trigger automated static analysis tools to flag NameErrors.
**Action:** Use list comprehensions over statically initialized constants (e.g., `_BASIC_KEYS = [...]`) to eliminate redundant formatting and bypass linter false-positives for performance-critical dictionary lookups.
