## 2024-04-23 - Prevent synchronous disk I/O in OpenCV real-time rendering loops

**Learning:** Loading PIL fonts via `ImageFont.truetype` directly from disk inside the real-time processing loop (`while True:` with `cv2.VideoCapture`) is a massive performance bottleneck. The constant file I/O for the `.ttf` files drastically degrades the frames per second (FPS), making the camera translation stream laggy.

**Action:** Always cache assets like PIL fonts (or images) into an instance-level dictionary inside the `__init__` method, using size and paths as cache keys. Retrieve the loaded objects from memory during the hot loop to maintain real-time performance.
