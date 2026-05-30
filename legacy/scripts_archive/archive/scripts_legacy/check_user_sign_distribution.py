# Check user_sign metadata - which labels become null_act?
import pandas as pd
from huggingface_hub import hf_hub_download

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/user_sign_metadata.csv',
    repo_type='dataset'
)

df = pd.read_csv(meta_path, encoding='utf-8-sig')

# Count sign_clean distribution
sign_counts = df['sign_clean'].value_counts()
print("Sign distribution in user_sign:")
print(sign_counts)

print("\n=== null_act count ===")
null_count = (df['sign_clean'] == 'null_act').sum()
print(f"null_act: {null_count}")

print("\n=== Words with most samples ===")
print(sign_counts.head(20))