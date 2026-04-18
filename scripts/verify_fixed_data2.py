# The data shows garbled in terminal but may be correct internally
# Let's verify by checking actual string content

import numpy as np

data = np.load('.cache/tsl51/expert_full_data_fixed.npz', allow_pickle=True)
classes = data['classes']

# Check first few classes - compare to known Thai words
print("Class 1:", repr(classes[1]))  # Should be กรุงเทพ
print("Class 2:", repr(classes[2]))  # Should be กลัว

# Try to match known Thai words
thai_words = ['กรุงเทพ', 'กลัว', 'กิน', 'ขนมปัง', 'ขอบคุณ']
for i, cls in enumerate(classes):
    for word in thai_words:
        if word in cls:
            print(f"Found {word} at index {i}: {repr(cls)}")