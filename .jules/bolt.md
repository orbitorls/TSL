
## 2024-05-30 - PIL ImageFont Initialization Overhead in Video Processing Loops
**Learning:** Initializing PIL `ImageFont.truetype` inside a video processing loop (like real-time webcam inference) introduces significant redundant file I/O overhead on every frame, which severely degrades FPS and increases latency.
**Action:** Always cache loaded fonts or other static assets initialized from disk using instance-level dictionaries or variables (e.g. `self._font_cache = {}`) when drawing on frames inside a high-frequency loop.
