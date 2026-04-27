# Check training data for NaN
import numpy as np

train_data = np.load('.cache/tsl51/user_sign_data.npz', allow_pickle=True)
X = train_data['X']

print(f"Training X shape: {X.shape}")
print(f"NaN count: {np.isnan(X).sum()}")
print(f"Inf count: {np.isinf(X).sum()}")
print(f"Min: {X.min()}, Max: {X.max()}")

# Check the benchmark CSV for NaN
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files

files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
csv_files = [f for f in files if f.startswith('landmarks/user_sign/')]

# Load a few and check NaN
for f in csv_files[:5]:
    csv_path = hf_hub_download(
        repo_id='Namonpas/thai-sign-language-tsl51',
        filename=f,
        repo_type='dataset'
    )
    df = pd.read_csv(csv_path)
    nan_count = df.isna().sum().sum()
    print(f"{f.split('/')[-1]}: {df.shape}, NaN: {nan_count}")