## 2026-04-11 - [Font Loading Bottleneck]
**Learning:** Loading TTF fonts from disk via ImageFont.truetype in the main OpenCV frame loop causes significant frame rate drops due to redundant disk I/O.
**Action:** Cache loaded fonts in memory using a dictionary keyed by font size to reuse them across frames.
