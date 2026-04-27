"""
Thai Sign Language Video Benchmark Script
============================================
ทดสอบโมเดลกับวิดีโอ + WER/BLEU/ROUGE Metrics

Usage:
    python tools/benchmark_video.py --samples 10
    python tools/benchmark_video.py --samples 50 --output results/
"""

import argparse
import sys
import json
import time
from pathlib import Path
import math

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset_utils import safe_mean

# Module logger
logger = logging.getLogger(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())

# =============================================================================
# EVALUATION METRICS (WER, BLEU, ROUGE)
# =============================================================================
def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_wer(reference, hypothesis):
    """Word Error Rate - เป็น % ยิ่งต่ำยิ่งดี"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    distance = levenshtein_distance(ref_chars, hyp_chars)
    wer = (distance / max(len(ref_chars), 1)) * 100
    
    return wer


def calculate_bleu(reference, hypothesis, n=4):
    """BLEU Score - เป็น % ยิ่งสูงยิ่งดี"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if not hyp_tokens or not ref_tokens:
        return 0.0
    
    # Calculate n-gram precision
    precisions = []
    for n_gram in range(1, min(n + 1, len(hyp_tokens) + 1)):
        ref_ngrams = [tuple(ref_tokens[i:i+n_gram]) for i in range(len(ref_tokens)-n_gram+1)]
        hyp_ngrams = [tuple(hyp_tokens[i:i+n_gram]) for i in range(len(hyp_tokens)-n_gram+1)]
        
        if not hyp_ngrams:
            precisions.append(0)
            continue
        
        matches = sum(1 for ng in hyp_ngrams if ng in ref_ngrams)
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)
    
    if not precisions or max(precisions) == 0:
        return 0.0
    
    # Geometric mean with brevity penalty
    log_precision = sum(math.log(p + 1e-10) for p in precisions) / len(precisions)
    brevity_penalty = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    
    bleu = brevity_penalty * math.exp(log_precision) * 100
    
    return bleu


def calculate_rouge(reference, hypothesis):
    """ROUGE Score - เป็น % ยิ่งสูงยิ่งดี"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if not hyp_tokens or not ref_tokens:
        return {'rouge_1': 0, 'rouge_2': 0, 'rouge_l': 0}
    
    # ROUGE-1
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    overlap = len(ref_set & hyp_set)
    rouge_1 = (overlap / len(ref_set)) * 100 if ref_set else 0
    
    # ROUGE-2
    ref_bigrams = set(tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens)-1))
    hyp_bigrams = set(tuple(hyp_tokens[i:i+2]) for i in range(len(hyp_tokens)-1))
    
    if ref_bigrams:
        overlap_2 = len(ref_bigrams & hyp_bigrams)
        rouge_2 = (overlap_2 / len(ref_bigrams)) * 100
    else:
        rouge_2 = 0
    
    # ROUGE-L
    rouge_l = calculate_lcs(reference, hypothesis) * 100
    
    return {'rouge_1': rouge_1, 'rouge_2': rouge_2, 'rouge_l': rouge_l}


def calculate_lcs(s1, s2):
    """Longest Common Subsequence ratio"""
    s1_tokens = s1.split()
    s2_tokens = s2.split()
    
    m, n = len(s1_tokens), len(s2_tokens)
    if m == 0 or n == 0:
        return 0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1_tokens[i-1] == s2_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n] / m


# =============================================================================
# MODEL
# =============================================================================
class GRUModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)


def load_model(model_path):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    classes = checkpoint['classes']
    mean = np.array(checkpoint['mean'])
    std = np.array(checkpoint['std'])
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    config = checkpoint.get('config', {})
    
    model = GRUModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=config.get('hidden', 128),
        num_layers=config.get('layers', 2),
        dropout=config.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    idx_to_label = {i: c for i, c in enumerate(classes)}
    
    return model, idx_to_label, mean, std


# =============================================================================
# FEATURE EXTRACTION - from offline CSV
# =============================================================================
def normalize_label(label):
    """Normalize label by removing suffixes like _สองมือ_6, _อายุเท่ากันหรือน้อยกว่า_5, etc."""
    if not label:
        return None
    
    # Filter out null samples
    if label.startswith('null_'):
        return None
    
    # Split on common suffixes and get base word
    for suffix in ['_สองมือ_', '_อายุเท่ากันหรือน้อยกว่า_', '_ทำท่ามือถามไปยังผู้นั้น_', 
                   '_เปิดมือสองข้าง_', '_บุลคคลที่สาม_', '_var_']:
        if suffix in label:
            return label.split(suffix)[0]
    
    return label


def extract_features_from_csv(csv_path):
    """Extract features from offline landmark CSV"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Handle NaN in raw data first
    df = df.fillna(0)
    
    features = []
    
    # Static features (average) - Left hand (63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'lh_{c}{i}'
            if col in df.columns:
                features.append(safe_mean(df[col]))
            else:
                features.append(0.0)
    
    # Static features - Right hand (63)
    for i in range(21):
        for c in ['x', 'y', 'z']:
            col = f'rh_{c}{i}'
            if col in df.columns:
                features.append(safe_mean(df[col]))
            else:
                features.append(0.0)
    
    # Static features - Pose (36)
    pose_cols = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                'lbrow_outer', 'lbrow_inner', 'rbrow_inner', 'rbrow_outer',
                'mouth_right', 'mouth_left']
    for base in pose_cols:
        for c in ['x', 'y', 'z']:
            col = f'{base}_{c}'
            if col in df.columns:
                features.append(safe_mean(df[col]))
            else:
                features.append(0.0)
    
    return np.nan_to_num(np.array(features, dtype=np.float32))


def predict_from_csv(csv_path, model, idx_to_label, mean, std):
    """Predict from offline CSV"""
    try:
        features = extract_features_from_csv(csv_path)
    except Exception as e:
        # Log the exception for debugging and return a safe empty result
        logger.debug("Failed extracting features from %s: %s", csv_path, e, exc_info=True)
        return None, 0.0, []
    
    if features is None or len(features) != len(mean):
        return None, 0.0, []
    
    normalized = (features - mean) / std
    tensor = torch.tensor([normalized], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
        top_k = 5
        top_indices = probs.argsort(descending=True)[:top_k].tolist()
        predictions = [(idx_to_label[idx], probs[idx].item()) for idx in top_indices]
    
    return predictions[0][0], predictions[0][1], predictions


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='TSL Video Benchmark with WER/BLEU/ROUGE')
    parser.add_argument('--model', type=str, default='models/tsl51_gru_20260412_220010.pt')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', type=str, default='results/')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TSL-51 Thai Sign Language - Video Benchmark")
    print("         Metrics: Accuracy, WER, BLEU, ROUGE")
    print("=" * 70)
    
    # 1. Load model
    print(f"\n[1] Loading model: {args.model}")
    model, idx_to_label, mean, std = load_model(args.model)
    print(f"    Classes: {len(idx_to_label)}, Input dim: {len(mean)}")
    
    # 2. Find user_sign videos (single word clips)
    print("\n[2] Finding user_sign videos from dataset...")
    from huggingface_hub import hf_hub_download, list_repo_files
    
    files = list(list_repo_files('Namonpas/thai-sign-language-tsl51', repo_type='dataset'))
    
    # Get videos from user_sign folder (single word clips)
    video_files = [f for f in files if f.startswith('videos/user_sign/') and f.endswith('.mp4')]
    
    print(f"Found {len(video_files)} user_sign videos")
    
    # Filter out null samples BEFORE sampling
    valid_video_files = []
    for vf in video_files:
        filename = vf.split('/')[-1].replace('.mp4', '')
        if normalize_label(filename) is not None:
            valid_video_files.append(vf)
    
    print(f"Valid videos (non-null): {len(valid_video_files)}")
    
    # Shuffle and sample from valid videos
    np.random.seed(42)
    selected = np.random.choice(valid_video_files, min(args.samples, len(valid_video_files)), replace=False)
    
    results = []
    metrics_sum = {'accuracy': 0, 'wer': 0, 'bleu': 0, 'rouge_1': 0, 'rouge_2': 0, 'rouge_l': 0}
    total_valid = 0
    
    print(f"\n[3] Processing {len(selected)} landmark CSVs...")
    print("-" * 70)
    print(f"{'#':<3} {'True':<12} {'Predicted':<12} {'Conf':<6} {'WER':<7} {'BLEU':<7} {'R-1':<7} {'Stat':<5}")
    print("-" * 70)
    
    for i, video_file in enumerate(selected):
        filename = video_file.split('/')[-1].replace('.mp4', '')
        
        # Normalize label - remove suffixes and filter null samples
        true_label = normalize_label(filename)
        if true_label is None:
            print(f"Skipping (null): {filename[:30]}")
            continue  # Skip null samples
        
        # Map video file to corresponding CSV in landmarks
        # videos/user_sign/xxx.mp4 -> landmarks/user_sign/xxx.csv
        csv_file = 'landmarks/user_sign/' + filename + '.csv'
        
        try:
            csv_path = hf_hub_download(
                repo_id='Namonpas/thai-sign-language-tsl51',
                filename=csv_file,
                repo_type='dataset'
            )
        except Exception as e:
            # Log and print a concise message
            logger.debug("Download error for %s: %s", filename, e, exc_info=True)
            print(f"Skipping (download error): {filename[:30]} - {e}")
            continue
        
        pred_word, conf, preds = predict_from_csv(csv_path, model, idx_to_label, mean, std)
        
        if pred_word:
            total_valid += 1
            
            is_correct = true_label == pred_word or true_label in pred_word
            
            wer = calculate_wer(true_label, pred_word)
            bleu = calculate_bleu(true_label, pred_word)
            rouge = calculate_rouge(true_label, pred_word)
            
            metrics_sum['accuracy'] += (1 if is_correct else 0)
            metrics_sum['wer'] += wer
            metrics_sum['bleu'] += bleu
            metrics_sum['rouge_1'] += rouge['rouge_1']
            metrics_sum['rouge_2'] += rouge['rouge_2']
            metrics_sum['rouge_l'] += rouge['rouge_l']
            
            status = "[OK]" if is_correct else "[X]"
            print(f"{i+1:<3} {true_label:<12} {pred_word:<12} {conf:.2f}   {wer:>5.1f}% {bleu:>5.1f}% {rouge['rouge_1']:>5.1f}%   {status}")
            
            results.append({
                'video': filename[:30],
                'true': true_label,
                'pred': pred_word,
                'confidence': conf,
                'wer': wer,
                'bleu': bleu,
                'rouge': rouge,
                'correct': is_correct
            })
    
    print("-" * 70)
    
    # 4. Summary
    if total_valid > 0:
        avg_acc = (metrics_sum['accuracy'] / total_valid) * 100
        avg_wer = metrics_sum['wer'] / total_valid
        avg_bleu = metrics_sum['bleu'] / total_valid
        avg_r1 = metrics_sum['rouge_1'] / total_valid
        avg_r2 = metrics_sum['rouge_2'] / total_valid
        avg_rl = metrics_sum['rouge_l'] / total_valid
    else:
        avg_acc = avg_wer = avg_bleu = avg_r1 = avg_r2 = avg_rl = 0
    
    print("\n[4] BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Videos processed: {len(results)}, Valid: {total_valid}")
    print()
    print("┌────────────────────────────────────────────────────────────┐")
    print("│                    METRICS SUMMARY                       │")
    print("├────────────────────────────────────────────────────────────┤")
    print(f"│  Accuracy:    {avg_acc:>6.2f}%  (Classification correct)        │")
    print(f"│  WER:         {avg_wer:>6.2f}%  (Word Error Rate - lower OK) │")
    print(f"│  BLEU:        {avg_bleu:>6.2f}%  (higher is better)           │")
    print(f"│  ROUGE-1:     {avg_r1:>6.2f}%  (unigram overlap)             │")
    print(f"│  ROUGE-2:     {avg_r2:>6.2f}%  (bigram overlap)              │")
    print(f"│  ROUGE-L:     {avg_rl:>6.2f}%  (LCS overlap)                 │")
    print("└────────────────────────────────────────────────────────────┘")
    
    # 5. Save
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    benchmark_result = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'total_videos': len(results),
        'valid_predictions': total_valid,
        'metrics': {
            'accuracy': avg_acc,
            'wer': avg_wer,
            'bleu': avg_bleu,
            'rouge_1': avg_r1,
            'rouge_2': avg_r2,
            'rouge_l': avg_rl
        },
        'results': results
    }
    
    output_file = output_dir / f"video_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_result, f, indent=2, ensure_ascii=False)
    
    # ============================================================
    # Generate PNG Report
    # ============================================================
    print("\n[5] Generating PNG Report...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Title
        fig.suptitle('TSL-51 Thai Sign Language - Video Benchmark Report', 
                     fontsize=22, fontweight='bold', color='#2c3e50', y=0.98)
        
        colors = {
            'primary': '#3498db',
            'success': '#2ecc71', 
            'warning': '#f39c12',
            'secondary': '#e74c3c',
            'dark': '#34495e',
            'light': '#ecf0f1'
        }
        
        # 1. Metrics Bar Chart (Top Left)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor('#ffffff')
        metric_names = ['Accuracy', 'BLEU', 'ROUGE-1', 'ROUGE-L']
        metric_values = [avg_acc, avg_bleu, avg_r1, avg_rl]
        bar_colors = [colors['success'], colors['primary'], colors['warning'], colors['primary']]
        bars = ax1.bar(metric_names, metric_values, color=bar_colors, edgecolor=colors['dark'], linewidth=2)
        for bar, v in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2., v + 1, f'{v:.1f}%', 
                    ha='center', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Metrics', fontsize=16, fontweight='bold', pad=10)
        ax1.set_ylim([0, 110])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 2. WER Gauge (Top Right)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor('#ffffff')
        wer_colors = colors['secondary'] if avg_wer > 50 else colors['warning'] if avg_wer > 20 else colors['success']
        ax2.barh(['WER'], [avg_wer], color=wer_colors, edgecolor=colors['dark'], height=0.4)
        ax2.text(avg_wer + 2, 0, f'{avg_wer:.1f}%', va='center', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 150])
        ax2.set_xlabel('Word Error Rate (%) - Lower is Better', fontsize=11)
        ax2.set_title('WER (Word Error Rate)', fontsize=16, fontweight='bold', pad=10)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. Per-Sample Results (Bottom Left)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor('#ffffff')
        ax3.axis('off')
        
        result_text = "SAMPLE PREDICTIONS\n" + "="*40 + "\n\n"
        for r in results[:10]:  # Show first 10
            status = "[OK]" if r['correct'] else "[X]"
            result_text += f"True: {r['true']:<10} Pred: {r['pred']:<10} {status}\n"
            result_text += f"   Conf: {r['confidence']:.2f} | WER: {r['wer']:.1f}% | BLEU: {r['bleu']:.1f}%\n\n"
        
        ax3.text(0.02, 0.98, result_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, 
                         edgecolor=colors['dark'], linewidth=1))
        
        # 4. Summary Info (Bottom Right)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor('#ffffff')
        ax4.axis('off')
        
        summary_text = f'''BENCHMARK SUMMARY
{'='*40}

Model: {args.model.split('/')[-1]}
Samples: {total_valid} user_sign videos

METRICS:
------------------------------------------
Accuracy:   {avg_acc:>6.2f}%
WER:        {avg_wer:>6.2f}% (lower OK)
BLEU:       {avg_bleu:>6.2f}%
ROUGE-1:    {avg_r1:>6.2f}%
ROUGE-2:    {avg_r2:>6.2f}%
ROUGE-L:    {avg_rl:>6.2f}%

Correct:    {metrics_sum['accuracy']}/{total_valid} ({avg_acc:.1f}%)
'''
        
        ax4.text(0.02, 0.98, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', alpha=0.9, 
                         edgecolor=colors['success'], linewidth=2))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        png_file = output_dir / f"video_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(png_file, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        print(f"Saved PNG: {png_file}")
        
    except Exception as e:
        # Log full traceback and print short warning
        logger.debug("Could not generate PNG report: %s", e, exc_info=True)
        print(f"Warning: Could not generate PNG: {e}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
