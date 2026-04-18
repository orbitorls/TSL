import torch

checkpoint = torch.load('models/tsl51_gru_20260412_220010.pt', map_location='cpu', weights_only=False)
print('Classes in model:')
for i, c in enumerate(checkpoint['classes']):
    print(f'  {i}: {c}')
print(f'\nTotal: {len(checkpoint["classes"])} classes')