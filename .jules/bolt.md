## 2024-05-24 - Prevent Redundant Font Loading in Translation Loops
**Learning:** In real-time computer vision applications, performing disk I/O operations inside high-frequency per-frame loops (such as loading Truetype fonts with `ImageFont.truetype` for overlay text) creates a significant performance bottleneck that reduces maximum achievable FPS and increases CPU usage.
**Action:** When rendering text on continuous video frames, always cache the loaded font instances by size to ensure fonts are loaded from disk only once instead of recreating the font object on every single frame.
