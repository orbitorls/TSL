# Try training on expert data instead - it has more samples (27k)
# This should give better generalization

print("Training on expert data...")
import subprocess
result = subprocess.run([
    'python', 'train_tsl51_v3.py',
    '--dataset', 'local',
    '--data-path', '.cache/tsl51/expert_full_data.npz',
    '--epochs', '30',
    '--folds', '3',
    '--model', 'gru',
    '--hidden', '256',
    '--layers', '3',
    '--dropout', '0.3'
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)