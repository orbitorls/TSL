import numpy as np
import os

cache_path = '.cache/tsl51/expert_full_data.npz'

if os.path.exists(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    print("Keys:", list(data.keys()))
    print("\nX shape:", data['X'].shape)
    print("y shape:", data['y'].shape)
    print("\nClasses:")
    classes = data['classes']
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")
    print(f"\nTotal: {len(classes)} classes")
else:
    print(f"File not found: {cache_path}")