## 2024-05-02 - Eliminate Font File I/O in Hot Loop
**Learning:** Loading PIL ImageFonts via `ImageFont.truetype` directly within the per-frame render loop (e.g., in `ThaiSignTranslator.run()`) causes a significant and unnecessary CPU/Disk I/O bottleneck, as the same font files are loaded repeatedly.
**Action:** Always employ a dictionary-based caching mechanism (`self._font_cache = {}`) using an immutable cache key (like `(font_size, tuple(font_paths))`) for assets initialized in loops to ensure fast lookups and eliminate repeated disk access.
