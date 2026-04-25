## 2025-02-20 - Redundant disk I/O in Real-time Render Loop
**Learning:** Loading TrueType fonts from disk via `ImageFont.truetype` inside the main rendering loop `process_frame` or `_draw_thai_text` can act as a severe performance bottleneck because file disk reading logic is executed on every frame resulting in extremely sluggish video and reduced FPS.
**Action:** When implementing or optimizing a real-time rendering loop involving `PIL.ImageFont`, cache font objects using a local dictionary map mapping file path and size instead of directly repeatedly loading it.
