## 2024-05-06 - Prevent Disk I/O in Render Loops
**Learning:** Instantiating `ImageFont.truetype` objects on every frame within a hot loop causes severe disk I/O bottlenecks and drops framerate significantly in real-time camera translation.
**Action:** Always load font assets (and similar static resources like images) once during initialization and cache them in memory using a dictionary (e.g., `self._font_cache`) mapped by attributes like font size and path.
