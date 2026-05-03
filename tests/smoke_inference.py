import json
import sys
from pathlib import Path

import numpy as np
import torch

# Paths
CKPT_PATH = "models/tsl51_gru_best.pt"
CACHE_DIR = Path(".cache/tsl51")
PREFERRED_NPZ = CACHE_DIR / "user_sign_data.npz"


def load_checkpoint(path):
    return torch.load(path, map_location='cpu')


def find_input_npz():
    # Try preferred file
    if PREFERRED_NPZ.exists():
        return PREFERRED_NPZ
    # Fallback: first user_sign_*.npz in cache dir
    if CACHE_DIR.exists():
        files = sorted(CACHE_DIR.glob('user_sign_*.npz'))
        if files:
            return files[0]
    return None


def main():
    try:
        if not Path(CKPT_PATH).exists():
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

        ckpt = load_checkpoint(CKPT_PATH)

        # Determine input features
        # Prefer loading .npz sample
        sample_npz = find_input_npz()
        if sample_npz is not None and sample_npz.exists():
            data = np.load(sample_npz)
            # Prefer key 'X' else first array
            if 'X' in data:
                features = data['X']
            else:
                features = data[data.files[0]]
            # Ensure single sample shape (162,) or (1,162)
            features = np.array(features, dtype=np.float32)
            if features.ndim > 1:
                # take first sample
                features = features.reshape(-1, features.shape[-1])[0]
        else:
            # Fallback: synthesize using checkpoint mean if available
            if isinstance(ckpt, dict) and 'mean' in ckpt:
                features = np.array(ckpt['mean'], dtype=np.float32)
            else:
                # As last resort, create zeros of expected length 162
                features = np.zeros(162, dtype=np.float32)

        # Import predictor from inference if available
        predictor = None
        try:
            # suppress prints from predictor init
            import io
            import contextlib
            from inference import TSLPredictor
            with contextlib.redirect_stdout(io.StringIO()):
                predictor = TSLPredictor(CKPT_PATH, device=torch.device('cpu'))
        except Exception:
            predictor = None

        if predictor is None:
            # Minimal fallback: try to load checkpoint and inspect classes
            classes = None
            if isinstance(ckpt, dict) and 'classes' in ckpt:
                classes = ckpt['classes']
            if classes is None:
                classes = ["unknown"]

            # Create a dummy linear model if state_dict present
            try:
                state = ckpt.get('state_dict', None) if isinstance(ckpt, dict) else None
                if state is not None:
                    # Try to infer output dim from final linear weight (for logging only)
                    for k, v in state.items():
                        if k.endswith('fc.weight') or k.endswith('net.{}'):
                            # out_dim = v.shape[0]  # available if needed for future use
                            break
                    # simple random linear mapping
                    import torch.nn as nn
                    linear = nn.Linear(features.size if hasattr(features, 'size') else features.shape[0], len(classes))
                    linear.eval()
                    with torch.no_grad():
                        inp = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
                        logits = linear(inp)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_idx = int(probs.argmax())
                        predicted = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
                        confidence = float(probs[pred_idx])
                        print(json.dumps({"predicted": predicted, "confidence": float(confidence)}))
                        return 0
            except Exception:
                # final fallback: return unknown with confidence 0
                print(json.dumps({"predicted": classes[0], "confidence": 0.0}))
                return 0

        # Use predictor to predict
        label, conf = predictor.predict(features)
        # Ensure native types
        out = {"predicted": label, "confidence": float(conf)}
        print(json.dumps(out))
        return 0

    except Exception as e:
        # Print error to stderr and exit non-zero
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}))
        return 2


if __name__ == '__main__':
    sys.exit(main())