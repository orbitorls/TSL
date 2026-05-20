## 2024-05-20 - Font Loading Bottleneck in Real-Time Inference Loop
**Learning:** Calling `ImageFont.truetype` and `ImageFont.load_default` inside a high-frequency real-time inference loop (like processing 30+ frames per second) creates a severe performance bottleneck due to continuous and redundant disk I/O operations and font parsing.
**Action:** When working on real-time rendering logic, always initialize and cache external assets (fonts, images, complex object instances) in the `__init__` method or via a caching dictionary to ensure they are loaded only once and reused across frames.
