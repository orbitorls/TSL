## 2024-05-18 - Avoid dynamic string formatting in hot loops
**Learning:** In python, doing dynamic string formatting inside nested loops (`f"lh_{c}{i}"`) inside a per-frame hot path (`_frame_dict_to_vector`) takes up unnecessary time and acts as a hidden bottleneck, because pre-computing the exact list of string keys (`_BASIC_KEYS`) and iterating over it is ~3x faster.
**Action:** When extracting data from dictionaries based on a known schema, pre-compute the list of keys instead of reconstructing them dynamically on every call.
