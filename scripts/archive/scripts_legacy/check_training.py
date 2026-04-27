import torch

checkpoint = torch.load('models/tsl51_gru_20260412_220010.pt', map_location='cpu', weights_only=False)

# Check training config
print("Config:", checkpoint.get('config', {}))
print("\nActual parameters:", checkpoint.get('actual_parameters', {}))

# Check JSON results for correct classes
import json
import os
results_dir = 'results'
json_files = [f for f in os.listdir(results_dir) if f.startswith('cv_') and f.endswith('.json')]
if json_files:
    latest = sorted(json_files)[-1]
    with open(f'results/{latest}', 'r', encoding='utf-8') as f:
        cv_results = json.load(f)
    print("\nClasses from CV results:")
    if 'classes' in cv_results:
        for i, c in enumerate(cv_results['classes'][:10]):
            print(f'  {i}: {c}')
    print(f"Total: {len(cv_results.get('classes', []))} classes")