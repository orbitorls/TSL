## 2025-04-27 - Cache PIL Fonts for Real-Time Rendering
**Learning:** In a real-time computer vision hot loop (like standard webcam inference loops reading ~30fps), repeated file I/O such as PIL `ImageFont.truetype()` will severely bottleneck processing and frame rates.
**Action:** Always pre-load fonts or cache them using immutable keys (e.g. `(font_size, tuple(font_paths))`) upon class initialization to entirely avoid reading from disk on every frame render.
