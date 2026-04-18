## 2024-05-19 - PIL Font Loading in Real-Time Loops
**Learning:** Performing disk I/O operations like `ImageFont.truetype` inside a hot real-time rendering loop (e.g., inside cv2 video capture loop) creates a significant performance bottleneck, leading to frame drops.
**Action:** Cache PIL fonts in memory using an instance-level dictionary based on font sizes during class initialization or the first render pass, and reuse them to eliminate redundant disk accesses.
