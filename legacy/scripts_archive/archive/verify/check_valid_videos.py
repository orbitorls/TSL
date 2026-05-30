# Check what video files exist and which are valid
from huggingface_hub import list_repo_files
import numpy as np

files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
video_files = [f for f in files if f.startswith('videos/user_sign/') and f.endswith('.mp4')]

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

# Count valid vs invalid
valid = []
invalid = []
for vf in video_files:
    filename = vf.split('/')[-1].replace('.mp4', '')
    result = normalize_label(filename)
    if result:
        valid.append(filename)
    else:
        invalid.append(filename)

print(f"Total videos: {len(video_files)}")
print(f"Valid (non-null): {len(valid)}")
print(f"Invalid (null): {len(invalid)}")
print(f"\nSample valid: {valid[:5]}")

# Try sampling with seed 42
np.random.seed(42)
selected = np.random.choice(video_files, min(10, len(video_files)), replace=False)
print("\nWith seed 42, first 10 selected:")
for s in selected:
    print(f"  {s}")