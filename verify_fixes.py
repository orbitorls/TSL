"""
Verify TSL Accuracy Fixes

This script verifies that all fixes have been applied correctly:
1. Feature extraction matches training
2. Sequence length is correct (100)
3. Minimum frames is correct (60)
4. Normalization threshold matches training (<1e-8)
"""

import math
import sys
import torch
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_DIR = Path(__file__).resolve().parent


def test_feature_extraction():
    """Test that feature extraction matches training exactly."""
    print("="*60)
    print("TEST 1: Feature Extraction")
    print("="*60)
    
    from tsl_tasks_extractor import extract_features as camera_extract
    
    # Define training extraction (from train_cv.py)
    def train_extract(seq):
        """Training version of extract_features."""
        n = len(seq)
        m = len(seq[0])
        
        mean = [sum(seq[i][j] for i in range(n)) / n for j in range(m)]
        std = [math.sqrt(sum((seq[i][j] - mean[j])**2 for i in range(n)) / n) for j in range(m)]
        mn = [min(seq[i][j] for i in range(n)) for j in range(m)]
        mx = [max(seq[i][j] for i in range(n)) for j in range(m)]
        
        if n > 1:
            delta = [sum(abs(seq[i+1][j] - seq[i][j]) for i in range(n-1)) / (n-1) for j in range(m)]
        else:
            delta = [0.0] * m
        
        return mean + std + mn + mx + delta
    
    # Create test data (100 frames, 174 values each)
    import random
    random.seed(42)
    test_frames = [[random.random() for _ in range(174)] for _ in range(100)]
    
    # Extract features
    camera_features = camera_extract(test_frames)
    train_features = train_extract(test_frames)
    
    # Compare
    if camera_features is None:
        print("[FAIL]: camera_extract returned None")
        return False
    
    if len(camera_features) != len(train_features):
        print(f"[FAIL]: Length mismatch - camera: {len(camera_features)}, train: {len(train_features)}")
        return False
    
    # Check if values match
    max_diff = max(abs(c - t) for c, t in zip(camera_features, train_features))
    
    if max_diff < 1e-10:
        print(f"[PASS] Feature extraction matches training (max diff: {max_diff:.2e})")
        print(f"   Feature dimension: {len(camera_features)}")
        return True
    else:
        print(f"[FAIL] Feature values differ (max diff: {max_diff:.2e})")
        return False


def test_sequence_length():
    """Test that the ThaiSignTranslator initialises sequence_buffer with maxlen=100."""
    print("\n" + "="*60)
    print("TEST 2: Sequence Length")
    print("="*60)

    content = Path("D:/TSL/camera_translate.py").read_text(encoding="utf-8")

    if "deque(maxlen=100)" in content:
        print("[PASS]: sequence_buffer initialised with maxlen=100 (matches training 89-121 frames)")
        return True
    else:
        print("[FAIL]: Could not find 'deque(maxlen=100)' in camera_translate.py")
        return False


def test_minimum_frames():
    """Test that minimum frames is correct."""
    print("\n" + "="*60)
    print("TEST 3: Minimum Frames")
    print("="*60)
    
    content = Path("D:/TSL/camera_translate.py").read_text(encoding="utf-8")
    
    # Check for "len(self.sequence_buffer) < 60"
    if "len(self.sequence_buffer) < 60" in content:
        print("[PASS]: Minimum frames = 60 (matches training range)")
        return True
    elif "len(self.sequence_buffer) < 10" in content:
        print("[FAIL]: Still using minimum frames = 10 (should be 60)")
        return False
    else:
        print("[WARNING]: Could not verify minimum frames")
        return False


def test_normalization_threshold():
    """Test that normalization threshold matches training (1e-8 floor from tsl_tasks_extractor)."""
    print("\n" + "="*60)
    print("TEST 4: Normalization Threshold")
    print("="*60)

    # Normalization lives in tsl_tasks_extractor.normalize_features; verify the constant there.
    try:
        from tsl_tasks_extractor import NORMALIZATION_STD_FLOOR, normalize_features
        if NORMALIZATION_STD_FLOOR == 1e-8:
            print(f"[PASS]: NORMALIZATION_STD_FLOOR = {NORMALIZATION_STD_FLOOR} (matches training 1e-8)")
            # Also verify normalize_features uses the constant, not a hard-coded fallback
            import inspect
            src = inspect.getsource(normalize_features)
            if "NORMALIZATION_STD_FLOOR" in src:
                print("   normalize_features references NORMALIZATION_STD_FLOOR correctly")
            else:
                print("   WARNING: normalize_features does not reference NORMALIZATION_STD_FLOOR")
            return True
        else:
            print(f"[FAIL]: NORMALIZATION_STD_FLOOR = {NORMALIZATION_STD_FLOOR} (expected 1e-8)")
            return False
    except ImportError as exc:
        print(f"[FAIL]: Could not import from tsl_tasks_extractor: {exc}")
        return False


def test_dimension_assertion():
    """Test that dimension assertion is present."""
    print("\n" + "="*60)
    print("TEST 5: Dimension Assertion")
    print("="*60)
    
    content = Path("D:/TSL/camera_translate.py").read_text(encoding="utf-8")
    
    # Check for dimension assertion
    if "len(features) != target_len" in content and "ERROR: Feature dimension mismatch" in content:
        print("[PASS]: Dimension assertion present (will error on mismatch)")
        return True
    else:
        print("[FAIL]: No dimension assertion found (silent failures possible)")
        return False


def test_quality_gate():
    """Test that quality gate is present."""
    print("\n" + "="*60)
    print("TEST 6: Quality Gate")
    print("="*60)
    
    content = Path("D:/TSL/camera_translate.py").read_text(encoding="utf-8")
    
    # Check for confidence threshold
    if "confidence < 0.70" in content or "confidence < 0.7" in content:
        print("[PASS]: Quality gate present (70% confidence threshold)")
        return True
    else:
        print("[FAIL]: No quality gate found (should reject low confidence)")
        return False


def test_consecutive_missing():
    """Test that consecutive missing counter is present."""
    print("\n" + "="*60)
    print("TEST 7: Consecutive Missing Counter")
    print("="*60)
    
    content = Path("D:/TSL/camera_translate.py").read_text(encoding="utf-8")
    
    # Check for consecutive missing counter
    if "consecutive_missing" in content and "self.consecutive_missing > 10" in content:
        print("[PASS]: Consecutive missing counter present (buffer cleared after 10 missing frames)")
        return True
    else:
        print("[FAIL]: No consecutive missing counter found")
        return False


def main():
    print("\n" + "="*60)
    print("TSL ACCURACY FIXES VERIFICATION")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Sequence Length", test_sequence_length()))
    results.append(("Minimum Frames", test_minimum_frames()))
    results.append(("Normalization Threshold", test_normalization_threshold()))
    results.append(("Dimension Assertion", test_dimension_assertion()))
    results.append(("Quality Gate", test_quality_gate()))
    results.append(("Consecutive Missing", test_consecutive_missing()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name:25} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED! Fixes applied correctly.")
    else:
        print(f"\n[!]  {total - passed} test(s) failed. Please check the fixes.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())