# Add debug output to benchmark
import sys
sys.path.insert(0, 'D:/TSL')
from benchmark_video import normalize_label

# Test with actual video files from dataset
from huggingface_hub import list_repo_files
files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
video_files = [f for f in files if f.startswith('videos/user_sign/') and f.endswith('.mp4')][:20]

print("Testing with actual video files:")
for vf in video_files:
    filename = vf.split('/')[-1].replace('.mp4', '')
    result = normalize_label(filename)
    print(f"  {filename[:40]} -> {result}")