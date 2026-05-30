# Save raw bytes to see what's really in there
import pandas as pd
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/expert_metadata.csv',
    repo_type='dataset'
)

# Read and save raw
with open(meta_path, 'rb') as f:
    raw = f.read()

# Save raw bytes
with open('results/raw_meta.bin', 'wb') as f:
    f.write(raw[:10000])

# Check first few bytes - should have BOM if UTF-8
print("First 10 bytes:", raw[:10])
print("Bytes as hex:", raw[:10].hex())

# Try reading with pandas different encodings
for enc in ['utf-8-sig', 'utf-8', 'cp874']:
    try:
        df = pd.read_csv(meta_path, encoding=enc)
        print(f"\n{enc}: {len(df)} rows")
        # Check a specific row - the sign_clean column
        print(f"  Sample sign_clean: {repr(df['sign_clean'].iloc[0])}")
    except Exception as e:
        # Log and continue for troubleshooting
        try:
            print(f"{enc}: Error reading metadata - {e}")
        except Exception:
            # Best effort printing failed; ignore
            pass
