# Let me read the metadata directly with proper encoding
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/expert_metadata.csv',
    repo_type='dataset'
)

# Read raw and decode manually
with open(meta_path, 'rb') as f:
    raw = f.read()

# Try different encodings
encodings = ['utf-8-sig', 'utf-8', 'cp874', 'iso-8859-11']

for enc in encodings:
    try:
        text = raw.decode(enc)
        print(f"\n=== {enc} ===")
        lines = text.split('\n')[:5]
        for line in lines:
            if ',' in line:
                parts = line.split(',')
                if len(parts) > 4:
                    print(f"  video_id: {parts[0][:30]}, sign_clean: {parts[4][:20]}")
        break
    except Exception as e:
        print(f"{enc}: {str(e)[:50]}")