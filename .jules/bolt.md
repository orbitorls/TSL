## 2024-05-24 - File I/O in render loop
**Learning:** Calling `ImageFont.truetype` inside a real-time camera loop (e.g. OpenCV frame processing) causes massive disk I/O bottlenecks.
**Action:** Always cache font loading using a dictionary indexed by font size and paths.
