## 2026-04-24 - PIL ImageFont in Hot Loop
**Learning:** Calling `ImageFont.truetype()` or `ImageFont.load_default()` inside a real-time rendering loop (per-frame) introduces significant performance bottlenecks due to disk I/O and font parsing.
**Action:** Always cache these assets using an instance-level dictionary initialized in `__init__` to avoid redundant file I/O operations.
