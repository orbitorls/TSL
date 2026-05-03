## 2024-05-18 - Caching Fonts in the Hot Loop
**Learning:** The real-time camera translation loop (camera_translate.py) was performing disk I/O on every frame by calling ImageFont.truetype and ImageFont.load_default sequentially for multiple paths. This caused unnecessary latency.
**Action:** Use an instance-level dictionary cache with a hashable key like (font_size, tuple(font_paths)) to store and re-use fonts loaded via PIL, moving disk I/O out of the hot loop.
