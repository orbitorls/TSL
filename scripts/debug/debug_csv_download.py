# Debug the benchmark more carefully
from huggingface_hub import list_repo_files, hf_hub_download
import numpy as np
import pandas as pd

files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
video_files = [f for f in files if f.startswith('videos/user_sign/') and f.endswith('.mp4')]

np.random.seed(42)
selected = video_files[:5]  # Just first 5

def normalize_label(label):
    if not label:
        return None
    if label.startswith('null_'):
        return None
    for suffix in ['_สองมือ_', '_อายุเท่ากันหรือน้อยกว่า_', '_ทำท่ามือถามไปยังผู้นั้น_', 
                   '_เปิดมือสองข้าง_', '_บุลคคลที่สาม_', '_var_']:
        if suffix in label:
            return label.split(suffix)[0]
    return label

# Check a few cases
for vf in selected:
    filename = vf.split('/')[-1].replace('.mp4', '')
    true_label = normalize_label(filename)
    print(f"Filename: {filename[:30]}")
    print(f"  true_label: {true_label}")
    
    csv_file = 'landmarks/user_sign/' + filename + '.csv'
    try:
        csv_path = hf_hub_download(
            repo_id='Namonpas/thai-sign-language-tsl51',
            filename=csv_file,
            repo_type='dataset'
        )
        print(f"  CSV downloaded: {csv_path}")
        
        # Try to read it
        df = pd.read_csv(csv_path)
        print(f"  CSV shape: {df.shape}")
        
    except Exception as e:
        print(f"  ERROR: {e}")