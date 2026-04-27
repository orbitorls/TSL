# Analyze what makes certain words work and others not
import json
import os
import numpy as np

results_dir = 'results'
json_files = [f for f in os.listdir(results_dir) if f.startswith('video_benchmark_') and f.endswith('.json')]
latest = sorted(json_files)[-1]

with open(f'results/{latest}', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Separate correct vs incorrect
correct = [r for r in data['results'] if r['correct']]
incorrect = [r for r in data['results'] if not r['correct']]

print("=== CORRECT PREDICTIONS ===")
for r in correct:
    print(f"  {r['true']} -> {r['pred']} (conf: {r['confidence']:.2f})")

print("\n=== INCORRECT PREDICTIONS ===")
for r in incorrect:
    print(f"  {r['true']} -> {r['pred']} (conf: {r['confidence']:.2f})")

print("\n=== NULL_ACT cases (as prediction) ===")
null_act = [r for r in data['results'] if r['pred'] == 'null_act']
print(f"Total null_act predictions: {len(null_act)}")
for r in null_act:
    print(f"  True: {r['true']}")

# Check confidence for null_act
print("\n=== Confidence analysis ===")
has_conf = [r for r in data['results'] if r['confidence'] and not np.isnan(r['confidence'])]
no_conf = [r for r in data['results'] if not r['confidence'] or np.isnan(r['confidence'])]
print(f"With confidence: {len(has_conf)}")
print(f"Without (null_act): {len(no_conf)}")