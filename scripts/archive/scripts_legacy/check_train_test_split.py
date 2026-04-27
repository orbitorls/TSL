# Check how many user_sign videos are in training vs benchmark
# The benchmark is testing on videos NOT in training!

import numpy as np

# Load training data
data = np.load('.cache/tsl51/user_sign_data.npz', allow_pickle=True)
print("Training data:")
print(f"  X shape: {data['X'].shape}")
print(f"  y shape: {data['y'].shape}")
print(f"  classes: {len(data['classes'])}")

# How many unique samples per class?
unique, counts = np.unique(data['y'], return_counts=True)
print("\nSamples per class in training:")
for i, (u, c) in enumerate(zip(unique[:10], counts[:10])):
    print(f"  Class {u}: {c} samples")

print(f"\nTotal training samples: {len(data['y'])}")
print("Total unique video IDs in benchmark: 638")