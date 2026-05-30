# Save classes to file and check hex
import torch

checkpoint = torch.load('models/tsl51_gru_20260412_220010.pt', map_location='cpu', weights_only=False)
classes = checkpoint['classes']

# Save raw to file
with open('results/classes_raw.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(classes):
        f.write(f"{i}: {c} | bytes: {c.encode('utf-8', errors='replace').hex()}\n")

# Try different encodings on a simple string test
test_str = "น้ำ"
print(f"Test string: {test_str}")
print(f"UTF-8 hex: {test_str.encode('utf-8').hex()}")
print(f"CP874 hex: {test_str.encode('cp874').hex()}")