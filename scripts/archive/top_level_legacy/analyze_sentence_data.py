"""Analyze TSL-51 sentence data structure"""
from huggingface_hub import hf_hub_download
import pandas as pd

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/sentence_metadata.csv',
    repo_type='dataset'
)
df = pd.read_csv(meta_path, encoding='utf-8-sig')

print('=== SENTENCE ANALYSIS ===')
print(f'Total videos: {len(df)}')
print(f'Unique sentences (sentence_id): {df["sentence_id"].nunique()}')
print(f'Unique sentence texts (sentence_clean): {df["sentence_clean"].nunique()}')
print()

print('=== SENTENCE LENGTHS ===')
df['word_count'] = df['sentence_clean'].apply(lambda x: len(str(x).split()))
print(f'Words per sentence: min={df["word_count"].min()}, max={df["word_count"].max()}, mean={df["word_count"].mean():.1f}')
print()

print('=== UNIQUE SENTENCES (first 10) ===')
for i, sent in enumerate(df['sentence_clean'].unique()[:10]):
    print(f'{i+1}. {sent}')
print()

print('=== FRAMES PER VIDEO ===')
print(f'Frames: min={df["frames_extracted"].min()}, max={df["frames_extracted"].max()}, mean={df["frames_extracted"].mean():.0f}')
print()

print('=== VARIATIONS ===')
print(df['recording_variation'].value_counts().head())
print()

# Parse sentence structure to understand sign sequence
print('=== SIGN SEQUENCE ANALYSIS ===')
# The sentence_id contains sign information in format: sign1_var_X__sign2_var_Y__...
unique_sentences = df['sentence_id'].unique()
print(f'Total unique sentence patterns: {len(unique_sentences)}')

# Count signs per sentence
signs_per_sentence = []
for sid in unique_sentences:
    # Split by __ to get signs with variations
    parts = str(sid).split('__')
    signs = [p.split('_var_')[0] for p in parts if p]
    signs_per_sentence.append(len(signs))

print(f'Signs per sentence: min={min(signs_per_sentence)}, max={max(signs_per_sentence)}, mean={sum(signs_per_sentence)/len(signs_per_sentence):.1f}')
print()

# Sample landmark file to understand frame-level structure
print('=== SAMPLE LANDMARK FILE ===')
sample_path = df.iloc[0]['landmark_path']
print(f'Sample file: {sample_path}')