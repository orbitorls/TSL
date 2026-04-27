# Try different encodings for metadata
import pandas as pd
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/user_sign_metadata.csv',
    repo_type='dataset'
)

encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'cp1252', 'iso-8859-11', 'thai']

print("Testing encodings on user_sign_metadata.csv:")
print("-" * 50)

for enc in encodings:
    try:
        df = pd.read_csv(meta_path, encoding=enc)
        print(f"\n{enc}:")
        print(df[['video_id', 'sign_clean']].head(3))
    except Exception as e:
        # Print concise error, don't crash verification script
        try:
            print(f"{enc}: ERROR - {str(e)[:120]}")
        except Exception:
            pass
