## 2024-05-24 - File I/O inside tight inference loops degrades performance
**Learning:** Instantiating `ImageFont.truetype` within real-time visual overlays causes significant frame drops due to repeated disk access for the same font files.
**Action:** Use a dictionary cache indexed by `(font_size, tuple(font_paths))` to memoize the loaded fonts within the class lifecycle instead.
