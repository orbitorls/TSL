## 2024-05-24 - PIL Font Loading in Real-Time Loop
**Learning:** Loading TrueType fonts via `ImageFont.truetype` from disk inside a hot video processing loop (`while True: cap.read()`) causes significant, redundant disk I/O and CPU overhead, acting as a major performance bottleneck for real-time translation pipelines.
**Action:** When drawing text with PIL in video processing loops, always cache the instantiated `ImageFont` objects in an instance dictionary (`self._font_cache = {}`) keyed by font size and properties to ensure they are loaded from disk exactly once.
