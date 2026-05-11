## 2024-05-11 - Font Caching in Real-Time Camera Translation
**Learning:** Redundant file I/O operations in real-time rendering loops, like loading PIL fonts from disk repeatedly using `ImageFont.truetype`, cause significant performance bottlenecks.
**Action:** Always implement a dictionary-based caching mechanism for assets like fonts, initialized in `__init__`, and use immutable/hashable keys like tuples `(font_size, tuple(font_paths))` for lookups to prevent disk reads in the hot loop.
