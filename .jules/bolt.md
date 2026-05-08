## 2024-05-24 - Avoid Disk I/O in Real-Time Render Loops
**Learning:** Loading fonts (or images) from disk inside a real-time rendering loop like `cv2.imshow` or frame processing functions (e.g., `ImageFont.truetype()`) causes significant, unnecessary I/O overhead and frame rate drops.
**Action:** Implement a caching mechanism (e.g., using a dictionary) initialized in `__init__` to load and store reusable assets like fonts once, and retrieve them via a helper method in the rendering hot loop. Use immutable types (like tuples of paths and font sizes) for cache keys.
