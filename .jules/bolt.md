## 2024-05-14 - Pillow Sub-Module Mocking in Python
**Learning:** When mocking complex nested packages like `PIL` that are imported via explicit sub-module imports (e.g., `from PIL import Image, ImageDraw, ImageFont`), assigning basic `MagicMock` instances to `sys.modules['PIL']` may cause ModuleNotFoundErrors.
**Action:** Use `types.ModuleType` to create dummy modules and explicitly attach them into `sys.modules` for both the parent package and each necessary sub-module.
