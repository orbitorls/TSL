# The Thai IS in the file correctly as UTF-8!
# The issue is just terminal display. Let me verify by using the strings without printing

import pandas as pd
from huggingface_hub import hf_hub_download
import numpy as np

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/expert_metadata.csv',
    repo_type='dataset'
)

df = pd.read_csv(meta_path, encoding='utf-8-sig')

# The Thai IS there - let's use it without printing
# Build mapping and save to npz directly

print("Building mapping from metadata...")

# Build video_id -> sign mapping
video_to_sign = {}
for idx, row in df.iterrows():
    vid = str(row['video_id'])
    sign = str(row['sign_clean'])
    if sign and sign != 'nan':
        video_to_sign[vid] = sign

print(f"Mapped {len(video_to_sign)} videos")

# The issue might be display only - let's try training with user_sign data instead
# which should have the same encoding issue but hopefully will work better

# First, let's check user_sign data
user_sign_path = '.cache/tsl51/user_sign_data.npz'
if not np.load(user_sign_path, allow_pickle=True).files:
    print("Downloading user_sign data...")
else:
    print("User_sign data exists")