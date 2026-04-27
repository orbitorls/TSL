# Check the landmark CSV file encoding
from huggingface_hub import hf_hub_download
import zipfile

# Download expert_scraped zip
zip_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='landmarks/expert_scraped.zip',
    repo_type='dataset'
)

# Extract one file and check its content
with zipfile.ZipFile(zip_path, 'r') as z:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')][:1]
    for csv_file in csv_files:
        with z.open(csv_file) as f:
            raw = f.read(500)
            print(f"File: {csv_file}")
            print(f"First 100 bytes: {raw[:100].hex()}")
            print(f"First 100 bytes decoded (utf-8): {raw[:100].decode('utf-8', errors='replace')[:100]}")