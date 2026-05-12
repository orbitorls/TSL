## 2024-05-18 - Caching ImageFont in Video Processing Loops
**Learning:** In real-time video processing using OpenCV and PIL, `ImageFont.truetype` and `ImageFont.load_default` can introduce significant disk I/O overhead if called inside the per-frame hot loop.
**Action:** Always instantiate immutable assets (like fonts, loaded images) once, or cache them using an instance-level dictionary based on parameters (like font path and size) to ensure smooth frame rendering.
