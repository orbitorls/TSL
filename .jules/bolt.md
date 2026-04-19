## 2026-04-19 - Font Caching in Real-Time Loop
**Learning:** Loading PIL fonts via `ImageFont.truetype()` or `ImageFont.load_default()` involves significant disk I/O. Doing this redundantly inside a real-time frame processing loop (`run` and `_draw_thai_text` methods in `camera_translate.py`) causes a severe performance bottleneck.
**Action:** Always cache these assets using an instance-level dictionary (e.g., `self._font_cache = {}` in `__init__`) to retrieve the loaded font from memory instead of reading it from disk on every frame.
