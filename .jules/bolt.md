## 2024-05-18 - Caching Pillow ImageFont

**Learning:** Initializing `ImageFont.truetype` and fallback fonts inside tight loops like per-frame drawing functions (e.g., OpenCV/Pillow rendering) introduces severe disk I/O bottlenecks. Loading the same `.ttf` file repeatedly on every frame causes significant lag in real-time video processing.

**Action:** Always implement a simple dictionary cache (e.g., `self._font_cache`) that stores loaded `ImageFont` instances keyed by font size and type, and fetch from it instead of re-instantiating the font class on every frame.
