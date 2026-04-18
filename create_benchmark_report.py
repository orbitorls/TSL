import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Find Thai-compatible font
thai_fonts = [f.name for f in fm.fontManager.ttflist if any(x in f.name.lower() for x in ['noto', 'arial', 'dejavu', 'loma'])]
if thai_fonts:
    plt.rcParams['font.family'] = thai_fonts[0]
    print(f"Using font: {thai_fonts[0]}")

# Load benchmark results
with open('results/benchmark_20260412.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# Create beautiful report
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#f8f9fa')

# Title
fig.suptitle('TSL-51 Thai Sign Language - Benchmark Report', 
             fontsize=28, fontweight='bold', color='#2c3e50', y=0.98)

# Colors
colors = {
    'primary': '#3498db',
    'secondary': '#e74c3c', 
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6',
    'dark': '#34495e',
    'light': '#ecf0f1'
}

# === 1. Main Metrics (Top Left) ===
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#ffffff')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
bar_colors = [colors['success'], colors['primary'], colors['warning'], colors['info']]
bars = ax1.bar(metrics, values, color=bar_colors, edgecolor=colors['dark'], linewidth=2, width=0.6)
for bar, v in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2., v + 0.5,
             f'{v:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax1.set_title('Overall Performance Metrics', fontsize=18, fontweight='bold', pad=15)
ax1.set_ylim([0, 110])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# === 2. Model Info (Top Right) ===
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#ffffff')
test_samples = results['test_samples']
inf_time = results['inference_time_seconds']
inf_speed = results['inferences_per_second']

model_info = f'''MODEL INFORMATION
==============================================
Model:        tsl51_gru_20260412_220010.pt
Architecture: Bidirectional GRU (2 layers)
Hidden Dim:  128
Parameters:  534,323 (~0.53M)
Input Dim:   162 features
Classes:     51 Thai signs

BENCHMARK SETUP
==============================================
Test Samples: {test_samples}
Split:       80% train / 20% test
Stratified:  Yes

INFERENCE SPEED
==============================================
Total Time:   {inf_time:.2f} seconds
Speed:       {inf_speed:.0f} inferences/sec'''

ax2.text(0.5, 0.95, model_info, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace', ha='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', alpha=0.9, 
                  edgecolor='#1abc9c', linewidth=2))
ax2.axis('off')

# === 3. Top Misclassifications (Bottom Left) ===
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#ffffff')
errors = results['top_errors']
error_labels = list(errors.keys())[:10]
error_values = list(errors.values())[:10]
bars3 = ax3.barh(range(len(error_labels)), error_values, color=colors['secondary'], edgecolor=colors['dark'])
ax3.set_yticks(range(len(error_labels)))
ax3.set_yticklabels(error_labels, fontsize=10)
ax3.set_xlabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Misclassifications', fontsize=16, fontweight='bold', pad=15)
ax3.invert_yaxis()
for bar, v in zip(bars3, error_values):
    ax3.text(v + 0.1, bar.get_y() + bar.get_height()/2., str(v), va='center', fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# === 4. Per-Class Accuracy Distribution (Bottom Right) ===
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#ffffff')
per_class = results['per_class_accuracy']
accs = sorted(per_class.values())
low_acc = [k for k, v in per_class.items() if v < 80]
high_acc = [k for k, v in per_class.items() if v >= 100]
mid_acc = [k for k, v in per_class.items() if 80 <= v < 100]

n, bins, patches = ax4.hist(accs, bins=10, edgecolor=colors['dark'], alpha=0.8)
for i, patch in enumerate(patches):
    if bins[i] >= 90:
        patch.set_facecolor(colors['success'])
    elif bins[i] >= 80:
        patch.set_facecolor(colors['warning'])
    else:
        patch.set_facecolor(colors['secondary'])

overall_acc = results['accuracy']
ax4.axvline(x=overall_acc, color=colors['primary'], linestyle='--', linewidth=2, 
            label=f'Overall: {overall_acc:.2f}%')
ax4.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
ax4.set_title('Per-Class Accuracy Distribution', fontsize=16, fontweight='bold', pad=15)
ax4.legend(fontsize=11)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

stats_text = f'100% accuracy: {len(high_acc)} classes\n80-99%: {len(mid_acc)} classes\n<80%: {len(low_acc)} classes'
ax4.text(0.98, 0.05, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='bottom', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/benchmark_report_20260412.png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

print('Saved: results/benchmark_report_20260412.png')