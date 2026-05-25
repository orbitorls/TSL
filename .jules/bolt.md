## 2024-05-24 - File I/O in Hot Rendering Loops
**Learning:** Found synchronous disk I/O (`ImageFont.truetype()`) directly inside the webcam frame processing loop (`while True:`), leading to significant framerate drops since the font was re-read from disk every single frame.
**Action:** When working on real-time computer vision or rendering scripts, always cache assets (fonts, images) in an instance-level dictionary (e.g., `self._font_cache`) to avoid redundant file operations per frame.
