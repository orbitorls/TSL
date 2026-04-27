# Check what landmark_path looks like in metadata
import pandas as pd
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/user_sign_metadata.csv',
    repo_type='dataset'
)

df = pd.read_csv(meta_path, encoding='utf-8-sig')
print("Sample landmark_path values:")
print(df[['video_id', 'landmark_path', 'sign_clean']].head(10))