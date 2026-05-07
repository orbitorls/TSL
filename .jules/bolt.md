## 2024-05-07 - Disk I/O in Render Loop
**Learning:** In `camera_translate.py`, the real-time translation rendering loop performs redundant disk I/O by loading fonts (`ImageFont.truetype`) every frame, which severely degrades rendering performance and frame rate.
**Action:** Implement dictionary-based caching for assets like fonts in `__init__` and reuse them to eliminate repeated disk reads during hot loops.
