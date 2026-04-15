## 2024-05-18 - Avoid redundant file I/O operations in real-time loops
**Learning:** Redundant file I/O operations like loading PIL fonts (`ImageFont.truetype` and `ImageFont.load_default`) from disk inside real-time rendering loops are severe performance bottlenecks that drastically drop framerates.
**Action:** Always cache these assets at an instance level (e.g., using an initialized dictionary in `__init__`) and fetch from memory in hot loops instead of loading them repeatedly every frame.
