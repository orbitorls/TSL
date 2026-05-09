## 2024-05-09 - Remove redundant file I/O in real-time rendering loops
**Learning:** Instantiating objects that read files from disk (like PIL's `ImageFont.truetype`) within a real-time frame rendering loop (like OpenCV video processing) causes severe performance bottlenecks due to constant, redundant disk I/O.
**Action:** When working with assets like fonts or images that are rendered frequently, always implement a dictionary-based cache at the instance level (initialized in `__init__`) and use immutable/hashable types like tuples for cache keys to avoid reading from disk on every frame.
