# Test with one specific word - "น้ำ"
# Extract features from its CSV and see what the model predicts

import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Load model
checkpoint = torch.load('models/tsl51_gru_20260412_220010.pt', map_location='cpu', weights_only=False)
classes = checkpoint['classes']
mean = np.array(checkpoint['mean'])
std = np.array(checkpoint['std'])

print("Model classes (encoded):", classes[:5])

# Create GRU model
class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

model = GRUModel(input_dim=162, num_classes=51, hidden_dim=128, num_layers=2)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Find "น้ำ" video in user_sign
# From user_sign metadata, videos start with video_id like "น้ำ_var_1_10"
# Let's download one "น้ำ" CSV

print("\n[1] Downloading น้ำ landmark CSV...")

# Find files starting with น้ำ
from huggingface_hub import list_repo_files
files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
nam_files = [f for f in files if f.startswith('landmarks/user_sign/') and 'น้ำ' in f]
print(f"Found {len(nam_files)} files with น้ำ")

if nam_files:
    csv_path = hf_hub_download(
        repo_id='Namonpas/thai-sign-language-tsl51',
        filename=nam_files[0],
        repo_type='dataset'
    )
    
    print(f"[2] Loading: {nam_files[0]}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns[:10].tolist()}...")
    
    # Extract features - same as training
    features = []
    
    # Left hand (63)
    from utils.dataset_utils import safe_mean
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    
    # Right hand (63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    
    # Pose (36)
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    
    features = np.array(features, dtype=np.float32)
    print(f"   Features shape: {features.shape}")
    
    # Normalize and predict
    normalized = (features - mean) / std
    tensor = torch.tensor([normalized], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
        # Get top 5 predictions
        top5 = probs.argsort(descending=True)[:5]
        
    print("\n[3] Model predictions:")
    for idx in top5:
        print(f"   Class {idx}: {classes[idx]} (prob: {probs[idx]:.4f})")
    
    # The TRUE label should be "น้ำ" - but we can't read it from encoded classes
    # Let's check what class index "น้ำ" might be by checking the ground truth
    
    print("\n[4] Checking ground truth:")
    # From earlier benchmark output, we saw "น้ำ" predicted as "null_act"
    # So the model is outputting wrong class
