# Let me test a simpler approach - use more benchmark samples
# And check what pattern exists in the predictions

# Also, let's manually test one prediction to see what's happening
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Load model
checkpoint = torch.load('models/tsl51_gru_20260412_230415.pt', map_location='cpu', weights_only=False)
classes = checkpoint['classes']
mean = np.array(checkpoint['mean'])
std = np.array(checkpoint['std'])

# Create model
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

# Test with "น้ำ" - the first test case that gives null_act
# Download the CSV for น้ำ
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
    
    df = pd.read_csv(csv_path)
    print(f"Loaded: {nam_files[0]}")
    print(f"Shape: {df.shape}")
    
    # Extract features
    features = []
    from utils.dataset_utils import safe_mean
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbbow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            features.append(safe_mean(df[col]) if col in df.columns else 0.0)
    
    features = np.array(features, dtype=np.float32)
    
    # Normalize
    normalized = (features - mean) / std
    
    # Check stats
    print("\nFeature stats:")
    print(f"  Min: {normalized.min():.2f}")
    print(f"  Max: {normalized.max():.2f}")
    print(f"  Mean: {normalized.mean():.2f}")
    print(f"  Std: {normalized.std():.2f}")
    
    # Predict
    tensor = torch.tensor([normalized], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
        # Get top 10
        top10 = probs.argsort(descending=True)[:10]
        
    print("\nTop 10 predictions:")
    for idx in top10:
        print(f"  {idx}: {classes[idx]} ({probs[idx]:.4f})")
