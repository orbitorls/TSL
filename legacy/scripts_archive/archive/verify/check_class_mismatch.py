# Check if training data and benchmark data are from same videos
# The issue might be that benchmark uses DIFFERENT landmark CSVs than training

import numpy as np

# Load training data
train_data = np.load('.cache/tsl51/user_sign_data.npz', allow_pickle=True)
print("Training data classes:", train_data['classes'][:5])

# The training data might have different class ordering than the model expects
# Let's check what's in the model
import torch
checkpoint = torch.load('models/tsl51_gru_20260412_231030.pt', map_location='cpu', weights_only=False)
print("\nModel classes:", checkpoint['classes'][:5])

# Check if they're the same
print("\nAre they the same?")
print("Training:", train_data['classes'].tolist()[:5])
print("Model:", checkpoint['classes'][:5].tolist()[:5])