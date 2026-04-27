"""
Test TSL Model with Real Videos - Check Label Match

This script tests the fixed model with videos and checks label overlap.
"""

import os
import sys
import cv2
import torch
from pathlib import Path
from collections import Counter

def configure_stdout_encoding():
    """Enable UTF-8 console output for Thai text when run as a script."""
    if sys.platform == 'win32' and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

PROJECT_DIR = Path(__file__).resolve().parent

# Import model and extraction
from camera_translate import ThaiSignTranslator
from tsl_tasks_extractor import LANDMARK_VECTOR_DIM, extract_features, normalize_features


def get_label_from_filename(filename):
    """Extract label from filename."""
    name_without_ext = filename.replace('.mp4', '')
    parts = name_without_ext.split('_var_')
    if len(parts) >= 1:
        label = parts[0]
        if label and label.lower() not in ['null', '']:
            return label
    return None


def check_label_overlap():
    """Check if video labels match model labels."""
    
    print("Loading model...")
    model_path = PROJECT_DIR / "models" / "tsl_model.pt"
    translator = ThaiSignTranslator(model_path=str(model_path))
    
    print(f"Model labels: {translator.labels}")
    
    # Get videos
    video_dir = PROJECT_DIR / "data" / "tsl51_full" / "videos" / "user_sign"
    video_files = list(video_dir.glob("*.mp4"))
    
    # Get unique labels from videos
    video_labels = set()
    for v in video_files:
        label = get_label_from_filename(v.name)
        if label:
            video_labels.add(label)
    
    print(f"\nUnique labels from videos: {len(video_labels)}")
    print(f"Sample video labels: {list(video_labels)[:10]}")
    
    # Check overlap
    model_labels_set = set(translator.labels)
    overlap = video_labels & model_labels_set
    
    print(f"\nOverlap: {len(overlap)} labels")
    print(f"Matching labels: {overlap}")
    
    if not overlap:
        print("\n[!] CRITICAL: No overlap between video labels and model labels!")
        print("[!] This explains the poor accuracy")
        print("[!] The model was trained on different data")
        
        # Check what training data was used
        train_dir = PROJECT_DIR / "data" / "tsl51_full_processed" / "train"
        if train_dir.exists():
            train_files = list(train_dir.glob("*.json"))[:5]
            print(f"\nTraining data files:")
            for f in train_files:
                print(f"  {f.name}")
        
        return False
    
    return True


def test_with_matching_videos(num_videos=20):
    """Test with videos that have matching labels."""
    
    model_path = PROJECT_DIR / "models" / "tsl_model.pt"
    translator = ThaiSignTranslator(model_path=str(model_path))
    model_labels_set = set(translator.labels)
    
    video_dir = PROJECT_DIR / "data" / "tsl51_full" / "videos" / "user_sign"
    video_files = list(video_dir.glob("*.mp4"))
    
    # Filter to matching labels
    testable = []
    for v in video_files:
        label = get_label_from_filename(v.name)
        if label and label in model_labels_set:
            testable.append((v, label))
    
    print(f"\nTestable videos: {len(testable)}")
    
    if not testable:
        print("No testable videos found!")
        return None
    
    results = []
    correct = 0
    total = 0
    
    print(f"\nTesting {min(num_videos, len(testable))} videos...\n")
    
    for i, (video_path, true_label) in enumerate(testable[:num_videos]):
        print(f"[{i+1}/{min(num_videos, len(testable))}] {video_path.name}")
        print(f"  True: {true_label}")
        
        pred, conf = process_video(video_path, translator)
        
        if pred:
            is_correct = pred == true_label
            status = "[OK]" if is_correct else "[X]"
            print(f"  Pred: {pred} ({conf:.1%}) {status}")
            results.append({'true': true_label, 'pred': pred, 'correct': is_correct})
            if is_correct:
                correct += 1
            total += 1
        else:
            print(f"  Pred: None")
            results.append({'true': true_label, 'pred': None, 'correct': False})
        
        print()
    
    # Summary
    print("="*60)
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "No predictions")
    
    return results


def process_video(video_path, translator, min_frames=60):
    """Process a single video and return prediction."""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0
    
    landmarks_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        _, landmarks = translator.process_frame(frame)
        
        if landmarks and len(landmarks) == LANDMARK_VECTOR_DIM:
            landmarks_list.append(landmarks)
        
        if len(landmarks_list) >= 100:
            break
    
    cap.release()
    
    print(f"  Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f}, Valid: {len(landmarks_list)}")
    
    if len(landmarks_list) < min_frames:
        return None, 0.0
    
    frames = landmarks_list[-100:]
    features = extract_features(frames)
    
    if features is None or len(features) != len(translator.mean):
        return None, 0.0

    normalized = normalize_features(features, translator.mean, translator.std)
    if normalized is None:
        return None, 0.0
    tensor = torch.tensor([normalized], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = translator.model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
    
    if confidence < 0.70:
        return None, confidence
    
    return translator.idx_to_label[top_idx], confidence


if __name__ == "__main__":
    configure_stdout_encoding()
    print("="*60)
    print("CHECKING LABEL OVERLAP")
    print("="*60 + "\n")
    
    has_overlap = check_label_overlap()
    
    if has_overlap:
        print("\n" + "="*60)
        print("TESTING WITH MATCHING VIDEOS")
        print("="*60 + "\n")
        
        test_with_matching_videos(20)
    else:
        print("\n[!] Cannot test - labels don't match!")
        print("[!] Model was trained on different dataset")
        print("[!] Need to retrain with correct data")
