# Analyze benchmark predictions more carefully
import json
import os
from collections import Counter

# Load latest benchmark results
results_dir = 'results'
json_files = [f for f in os.listdir(results_dir) if f.startswith('video_benchmark_') and f.endswith('.json')]
latest = sorted(json_files)[-1]

with open(f'results/{latest}', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== Benchmark Analysis ===\n")

# 1. Check prediction distribution
pred_counts = Counter([r['pred'] for r in data['results']])
print("Prediction distribution:")
for pred, count in pred_counts.most_common(10):
    print(f"  {pred}: {count}")

# 2. Check confidence levels
high_conf = [r for r in data['results'] if r['confidence'] > 0.7]
low_conf = [r for r in data['results'] if r['confidence'] < 0.5]

print(f"\nHigh confidence (>0.7): {len(high_conf)}")
print(f"Low confidence (<0.5): {len(low_conf)}")

# 3. Check null_act cases
null_act = [r for r in data['results'] if r['pred'] == 'null_act']
print(f"\nnull_act predictions: {len(null_act)}")

# 4. Show confidence vs correctness
print("\nConfidence vs Correctness:")
for r in data['results'][:10]:
    status = "[OK]" if r['correct'] else "[X]"
    print(f"  Conf: {r['confidence']:.2f} {status} True: {r['true'][:15]:<15} Pred: {r['pred'][:15]:<15}")