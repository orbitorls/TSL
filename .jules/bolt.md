## $(date +%Y-%m-%d) - Cached PIL ImageFont loading in rendering loops
**Learning:** In real-time computer vision scripts like `camera_translate.py`, redundantly calling `ImageFont.truetype()` inside the hot frame-processing loop causes massive file I/O overhead, severely degrading framerate.
**Action:** Always extract static asset loading (fonts, base images) from rendering loops. If dynamic sizing or multiple paths are needed, implement a dictionary-based cache using `(font_size, tuple(font_paths))` as a hashable key.
