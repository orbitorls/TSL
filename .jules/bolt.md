## 2026-05-22 - Cached PIL fonts in real-time inference loop
**Learning:** Instantiating `ImageFont.truetype` directly from disk inside real-time rendering loops (like `cv2.VideoCapture` processing) causes severe I/O bottlenecks and drops FPS significantly.
**Action:** Always refactor redundant file I/O operations into an instance-level dictionary cache (e.g., `self._font_cache`) initialized in `__init__` and provide a safe getter method to fetch from memory instead of disk.
