import numpy as np

data = np.load('.cache/tsl51/expert_full_data_fixed.npz', allow_pickle=True)
print("Classes in fixed data:")
for i, c in enumerate(data['classes']):
    print(f"  {i}: {c}")
print(f"\nTotal: {len(data['classes'])} classes")