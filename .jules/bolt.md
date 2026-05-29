## 2025-02-23 - Caching TrueType Fonts to Prevent Slow Disk I/O
**Learning:** Parsing and loading PIL TrueType fonts from disk (e.g., `ImageFont.truetype`) inside a hot frame loop drastically degrades real-time rendering performance.
**Action:** Always implement a dictionary-based instance cache initialized in `__init__` for assets like fonts or images that are requested repeatedly, using immutable keys (e.g., `(size, tuple(paths))`).
