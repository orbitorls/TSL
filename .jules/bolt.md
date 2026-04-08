## 2025-04-08 - Font loading in real-time loop bottleneck
**Learning:** In the `camera_translate.py` real-time camera loop, loading a TrueType font (`ImageFont.truetype()`) directly on every frame causes significant disk I/O overhead and reduces FPS.
**Action:** Always implement a caching mechanism (like a `_font_cache` dict based on font size and paths) to load fonts exactly once in high-frequency rendering loops.
