## 2024-05-21 - Caching PIL Images/Fonts in Hot Loops
**Learning:** In real-time computer vision or text rendering applications, repeatedly parsing and loading binary assets like `.ttf` fonts from disk during every frame's draw cycle creates a massive bottleneck that decimates FPS.
**Action:** Always initialize an asset cache (e.g. `self._font_cache = {}`) during object construction and wrap file I/O calls in a helper method that keys by path and arguments to reuse instances.
