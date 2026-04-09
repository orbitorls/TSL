## 2024-05-24 - File I/O inside tight rendering loops
**Learning:** Initializing fonts (`ImageFont.truetype()`) directly inside a real-time `cv2` and `PIL` video rendering loop triggers expensive redundant disk I/O for every frame. This creates a severe performance bottleneck.
**Action:** Always pre-load or memoize/cache heavy assets (like fonts, images, and static resources) outside of high-frequency loops (e.g. `while True:` rendering loops).
