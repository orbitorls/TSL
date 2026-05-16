## 2026-05-16 - Font caching in inference scripts
**Learning:** Continuous redundant disk I/O for loading fonts (via `ImageFont.truetype`) inside a real-time computer vision frame processing loop causes a massive performance bottleneck, reducing FPS significantly.
**Action:** Always cache file-based resources (like fonts or images) at the class or module level using a dictionary, ensuring keys are immutable (like tuples), before entering the hot execution path.
