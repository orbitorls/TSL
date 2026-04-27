# Try to properly decode Thai metadata using cp874 (Thai Windows)
import pandas as pd
from huggingface_hub import hf_hub_download

# Try cp874 encoding
meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/user_sign_metadata.csv',
    repo_type='dataset'
)

# Read raw bytes
with open(meta_path, 'rb') as f:
    raw = f.read(500)
    print("Raw bytes (first 100):", raw[:100])
    print()

# Try different approach - read with different encodings
print("\nTrying latin1 (which might preserve bytes):")
df_latin = pd.read_csv(meta_path, encoding='latin1')
print("Sample (latin1):")
print(df_latin[['video_id', 'sign_clean']].head(3))

# Try cp874
print("\nTrying cp874 (Thai Windows):")
try:
    df_cp874 = pd.read_csv(meta_path, encoding='cp874')
    print("Sample (cp874):")
    print(df_cp874[['video_id', 'sign_clean']].head(3))
except Exception as e:
    print(f"cp874 failed: {e}")