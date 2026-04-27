# Check what columns are in the landmark CSV
import pandas as pd
from huggingface_hub import hf_hub_download

# Get a sample CSV
from huggingface_hub import list_repo_files
files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
csv_files = [f for f in files if f.startswith('landmarks/user_sign/') and f.endswith('.csv')][:1]

csv_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename=csv_files[0],
    repo_type='dataset'
)

df = pd.read_csv(csv_path)
print(f"Columns in CSV ({len(df.columns)} total):")
print(df.columns.tolist()[:30])

# Check for NaN
print(f"\nNaN count: {df.isna().sum().sum()}")
print(f"Shape: {df.shape}")