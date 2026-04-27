"""Analyze TSL-51 sentence data structure"""
from huggingface_hub import hf_hub_download
import pandas as pd
import os
import json
import csv

meta_path = hf_hub_download(
    repo_id='Namonpas/thai-sign-language-tsl51',
    filename='metadata/sentence_metadata.csv',
    repo_type='dataset'
)
# Use robust read with utf-8-sig and fallback to latin1 if necessary
try:
    df = pd.read_csv(meta_path, encoding='utf-8-sig')
except Exception:
    try:
        df = pd.read_csv(meta_path, encoding='latin1')
    except Exception:
        df = pd.read_csv(meta_path, encoding='utf-8', errors='replace')

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

# Ensure results directory exists and write metrics to JSON and CSV
results_dir = r'D:\\TSL\\results'
os.makedirs(results_dir, exist_ok=True)

metrics = {
    'total_videos': len(df),
    'unique_sentence_ids': int(df['sentence_id'].nunique()),
    'unique_sentence_texts': int(df['sentence_clean'].nunique()),
    'words_per_sentence': {
        'min': int(df['word_count'].min()),
        'max': int(df['word_count'].max()),
        'mean': float(df['word_count'].mean())
    },
    'frames_per_video': {
        'min': int(df['frames_extracted'].min()),
        'max': int(df['frames_extracted'].max()),
        'mean': float(df['frames_extracted'].mean())
    },
    'recording_variation_counts': df['recording_variation'].value_counts().to_dict(),
    'sample_landmark_path': sample_path,
    'signs_per_sentence': {
        'min': int(min(signs_per_sentence)),
        'max': int(max(signs_per_sentence)),
        'mean': float(sum(signs_per_sentence)/len(signs_per_sentence))
    }
}

# Write JSON
json_path = os.path.join(results_dir, 'data_inventory.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# Write CSV
csv_path = os.path.join(results_dir, 'data_inventory.csv')
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['total_videos', metrics['total_videos']])
    writer.writerow(['unique_sentence_ids', metrics['unique_sentence_ids']])
    writer.writerow(['unique_sentence_texts', metrics['unique_sentence_texts']])
    writer.writerow(['words_per_sentence_min', metrics['words_per_sentence']['min']])
    writer.writerow(['words_per_sentence_max', metrics['words_per_sentence']['max']])
    writer.writerow(['words_per_sentence_mean', metrics['words_per_sentence']['mean']])
    writer.writerow(['frames_per_video_min', metrics['frames_per_video']['min']])
    writer.writerow(['frames_per_video_max', metrics['frames_per_video']['max']])
    writer.writerow(['frames_per_video_mean', metrics['frames_per_video']['mean']])
    for k, v in metrics['recording_variation_counts'].items():
        writer.writerow([f'recording_variation_{k}', v])
    writer.writerow(['sample_landmark_path', metrics['sample_landmark_path']])
    writer.writerow(['signs_per_sentence_min', metrics['signs_per_sentence']['min']])
    writer.writerow(['signs_per_sentence_max', metrics['signs_per_sentence']['max']])
    writer.writerow(['signs_per_sentence_mean', metrics['signs_per_sentence']['mean']])

print(f'Wrote JSON: {json_path}')
print(f'Wrote CSV: {csv_path}')
