## 2024-04-22 - Redundant I/O in Real-Time Hot Loops
**Learning:** Initializing PIL fonts using `ImageFont.truetype(font_path, size)` repeatedly within real-time rendering loops (like a `cv2` video stream) creates massive file I/O bottlenecks. In python, the default behavior of `truetype` reads the font from disk on every call, killing FPS in real-time camera translation.
**Action:** When drawing text iteratively in applications, initialize and cache the fonts using an instance-level dictionary based on sizes during `__init__`, and reuse the cached objects to minimize disk access.
