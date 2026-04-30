## 2024-05-18 - Caching Fonts for Real-time Rendering

**Learning:** Repeatedly loading `PIL.ImageFont.truetype` and `PIL.ImageFont.load_default` inside a real-time rendering loop across multiple font paths is extremely slow and blocks the main thread.
**Action:** Always pre-load and cache fonts (e.g., using an instance-level dictionary based on font sizes and paths) during initialization or on the first request to eliminate redundant disk I/O in hot loops.
