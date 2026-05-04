## 2025-05-04 - Cache UI Assets in Inference Loops
**Learning:** Calling OS-level font loading functions like `ImageFont.truetype()` inside the per-frame prediction loop creates a major performance bottleneck due to redundant disk I/O, significantly dragging down FPS during real-time camera inference.
**Action:** When working on real-time application code, extract I/O operations (like font loading, image decoding) from the hot loop. Instead, lazily load and cache these assets into a dictionary during initialization or on their first request.
