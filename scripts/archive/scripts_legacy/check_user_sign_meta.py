# Check what class labels exist in user_sign metadata
import pandas as pd
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/user_sign_metadata.csv',
    repo_type='dataset'
)
metadata = pd.read_csv(meta_path, encoding='utf-8-sig')
print("Columns:", metadata.columns.tolist())
print("\nFirst 10 rows:")
print(metadata[['video_id', 'sign_clean']].head(10))
print("\nUnique signs (sign_clean):")
unique_signs = metadata['sign_clean'].dropna().unique()
for i, s in enumerate(sorted(unique_signs)[:20]):
    print(f"  {i}: {s}")
print(f"\nTotal unique: {len(unique_signs)}")