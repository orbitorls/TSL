import re

with open("src/inference/video.py", "r") as f:
    content = f.read()

# Fix E701
content = content.replace("if not preds: return None, 0", "if not preds:\n            return None, 0")

with open("src/inference/video.py", "w") as f:
    f.write(content)
