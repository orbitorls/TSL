# Check if some classes are readable in the model
import torch
import logging

logger = logging.getLogger(__name__)

checkpoint = torch.load('models/tsl51_gru_20260412_220010.pt', map_location='cpu', weights_only=False)
classes = checkpoint['classes']

# Print raw bytes
print("Raw class representations:")
for i, c in enumerate(classes[:10]):
    print(f"  {i}: {repr(c)}")

# Check if any class is readable
print("\n\nLooking for readable Thai:")
for i, c in enumerate(classes):
    if 'ก' in c or 'เ' in c or 'า' in c or 'ะ' in c:
        print(f"  Found readable: {i}: {c}")
        break

# Maybe we can decode from Latin1?
print("\n\nTrying latin1 decode:")
for i, c in enumerate(classes[:5]):
    try:
        decoded = c.encode('latin1').decode('utf-8')
        print(f"  {i}: {decoded}")
    except Exception as e:
        # Report failure but continue checking other entries
        print(f"  {i}: failed ({e})")
        logger.exception("Failed to decode class at index %s: %s", i, e)
