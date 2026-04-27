"""Training visualizations and report generation."""
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from src.train.compat import HAS_MATPLOTLIB, plt
from src.train.utils import estimate_params


def _to_float(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def save_visualizations(results, fold_results, output_dir, args):
    """Save comprehensive training visualizations."""
    if not HAS_MATPLOTLIB:
        return
    if plt is None:
        return

    for k in [
        "average_accuracy",
        "std_accuracy",
        "overall_accuracy",
        "precision",
        "recall",
        "f1_score",
        "num_classes",
        "num_samples",
        "test_samples",
        "input_dim",
    ]:
        if k in results:
            results[k] = _to_float(results[k])
    if "test_results" in results and isinstance(results["test_results"], dict):
        for k in results["test_results"]:
            results["test_results"][k] = _to_float(results["test_results"][k])
    for fr in fold_results:
        if "accuracy" in fr:
            fr["accuracy"] = _to_float(fr["accuracy"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("TSL-51 Thai Sign Language Training Report", fontsize=24, fontweight="bold", color="#2c3e50", y=0.98)

    all_accs = [float(r["accuracy"]) * 100 for r in fold_results]
    all_metrics = [
        float(results["overall_accuracy"]) * 100,
        float(results["precision"]) * 100,
        float(results["recall"]) * 100,
        float(results["f1_score"]) * 100,
    ]
    min_acc = min(min(all_accs), min(all_metrics)) - 15
    max_acc = max(max(all_accs), max(all_metrics)) + 10
    y_min = max(0, min_acc)
    y_max = min(100, max_acc)

    colors = {
        "primary": "#3498db",
        "secondary": "#e74c3c",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "info": "#9b59b6",
        "dark": "#34495e",
        "light": "#ecf0f1",
    }

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#ffffff")
    folds = [r["fold"] for r in fold_results]
    accs = [r["accuracy"] * 100 for r in fold_results]
    bars = ax1.bar(folds, accs, color=colors["primary"], edgecolor=colors["dark"], linewidth=2, width=0.7, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1, f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color=colors["dark"])
    ax1.axhline(y=results["average_accuracy"] * 100, color=colors["secondary"], linestyle="--", linewidth=3, label=f"CV Average: {results['average_accuracy'] * 100:.2f}%", alpha=0.8)
    ax1.set_xlabel("Fold", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax1.set_title("Cross-Validation Accuracy by Fold", fontsize=16, fontweight="bold", pad=15)
    ax1.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax1.set_ylim((y_min, y_max))
    ax1.grid(axis="y", alpha=0.4, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#ffffff")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [
        results["overall_accuracy"],
        results["precision"],
        results["recall"],
        results["f1_score"],
    ]
    metric_colors = [colors["primary"], colors["warning"], colors["success"], colors["info"]]
    bars2 = ax2.bar(metrics, [v * 100 for v in values], color=metric_colors, edgecolor=colors["dark"], linewidth=2, width=0.65, alpha=0.85)
    for bar, v in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color=colors["dark"])
    ax2.set_ylabel("Score (%)", fontsize=14, fontweight="bold")
    ax2.set_title("Overall Performance Metrics", fontsize=16, fontweight="bold", pad=15)
    ax2.set_ylim((y_min, y_max))
    ax2.grid(axis="y", alpha=0.4, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#ffffff")
    num_classes = results.get("num_classes", 51)
    num_samples = results.get("num_samples", 0)
    avg_per_class = num_samples / num_classes if num_classes > 0 else 0
    fold_table = "PER-FOLD RESULTS\n" + "-" * 40 + "\n"
    for fold in fold_results:
        fold_table += f"  Fold {fold['fold']}: {fold['accuracy'] * 100:>6.2f}%\n"
    fold_table += "-" * 40 + "\n"
    fold_table += f"  Best Fold: Fold {np.argmax([r['accuracy'] for r in fold_results]) + 1}\n"
    fold_table += f"  Std Dev:   {results['std_accuracy'] * 100:>6.2f}%\n"
    fold_table += f"  Best Acc:  {max([r['accuracy'] for r in fold_results]) * 100:>6.2f}%\n"
    fold_table += f"  Worst Acc: {min([r['accuracy'] for r in fold_results]) * 100:>6.2f}%\n"
    fold_table += "-" * 40 + "\n"
    fold_table += f"  Classes:   {num_classes} signs\n"
    fold_table += f"  Avg/Class: {avg_per_class:>6.1f} samples\n"
    ax3.text(0.5, 0.95, fold_table, transform=ax3.transAxes, fontsize=11, verticalalignment="top", fontfamily="monospace", ha="center", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f6f3", alpha=0.9, edgecolor="#1abc9c", linewidth=2))
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#ffffff")
    augment_val = args.augment if hasattr(args, "augment") and args.augment else 1
    test_split_val = args.test_split if hasattr(args, "test_split") else 0
    input_dim = results.get("input_dim", 162)
    if "actual_parameters" in results:
        params = float(results["actual_parameters"])
        params_note = " (actual)"
    else:
        params = estimate_params(args.model, args.hidden, args.layers, input_dim, results.get("num_classes", 51))
        params_note = " (estimated)"
    config_text = (
        f"MODEL CONFIGURATION\n{'-' * 40}\n"
        f"  Model Type:        {args.model.upper()} (Bidirectional={args.model == 'gru'})\n"
        f"  Hidden Dimensions: {args.hidden}\n"
        f"  Number of Layers:  {args.layers}\n"
        f"  Dropout Rate:      {args.dropout}\n"
        f"  Input Features:    {input_dim}\n"
        f"  Output Classes:    {results.get('num_classes', 51)}\n"
        f"  Est. Parameters:   ~{params / 1e6:.2f}M{params_note}\n"
        f"{'-' * 40}\n"
        f"  Optimizer: AdamW\n"
        f"  Learning Rate:     {args.lr}\n"
        f"  Weight Decay: 1e-4\n"
        f"  LR Scheduler: OneCycleLR\n"
        f"{'-' * 40}\n"
        f"  Batch Size:  {args.batch}\n"
        f"  Max Epochs:  {args.epochs}\n"
        f"  Early Stop:  patience={args.patience}\n"
        f"  Random Seed: {args.seed}\n"
        f"  CV Folds:    {args.folds}\n"
        f"  Augmentation: {augment_val}x\n"
        f"  Test Split: {test_split_val * 100:.0f}%\n"
    )
    ax4.text(0.02, 0.98, config_text, transform=ax4.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", alpha=0.9, edgecolor="#f39c12", linewidth=2))
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#ffffff")
    test_samples_line = ""
    if results.get("test_samples", 0) > 0:
        test_samples_line = f"  Test Samples: {results.get('test_samples', 0)}\n"
    dataset_text = (
        f"DATASET INFORMATION\n{'-' * 40}\n"
        f"  Dataset:          tsl51_user_sign\n"
        f"  Total Samples:    {results['num_samples']}\n"
        f"  Train Samples:    {results['num_samples'] - results.get('test_samples', 0)}\n"
        f"{test_samples_line}"
        f"  Number of Classes: {results['num_classes']}\n"
        f"  Feature Dim:       {results['input_dim']}\n"
        f"{'-' * 40}\n"
        f"  Feature Breakdown:\n"
        f"    Left Hand:  21 pts x 3 = 63\n"
        f"    Right Hand: 21 pts x 3 = 63\n"
        f"    Pose:       12 pts x 3 = 36\n"
        f"    Total:               = 162\n"
        f"{'-' * 40}\n"
        f"  Augmentation Methods:\n"
        f"    1. Gaussian Noise (sigma=0.01)\n"
        f"    2. Random Scale (0.95-1.05)\n"
        f"    3. Left-Right Hand Flip\n"
    )
    ax5.text(0.02, 0.98, dataset_text, transform=ax5.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f6f3", alpha=0.9, edgecolor="#1abc9c", linewidth=2))
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#ffffff")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    test_line = ""
    if "test_results" in results:
        tr = results["test_results"]
        test_line = (
            f"  Test Accuracy:  {tr['test_accuracy'] * 100:>6.2f}%\n"
            f"  Test Precision: {tr['test_precision'] * 100:>6.2f}%\n"
            f"  Test Recall:    {tr['test_recall'] * 100:>6.2f}%\n"
            f"  Test F1-Score:  {tr['test_f1_score'] * 100:>6.2f}%\n"
            f"{'-' * 40}\n"
        )
    results_text = (
        f"FINAL RESULTS\n{'-' * 40}\n"
        f"  CV Average:       {results['average_accuracy'] * 100:>6.2f}% (+/-{results['std_accuracy'] * 100:.2f}%)\n"
        f"  Overall Accuracy: {results['overall_accuracy'] * 100:>6.2f}%\n"
        f"  Precision:        {results['precision'] * 100:>6.2f}%\n"
        f"  Recall:           {results['recall'] * 100:>6.2f}%\n"
        f"  F1-Score:         {results['f1_score'] * 100:>6.2f}%\n"
        f"{test_line}"
        f"{'-' * 40}\n"
        f"  TRAINING ENVIRONMENT\n"
        f"{'-' * 40}\n"
        f"  Date:     {results['timestamp'][:19]}\n"
        f"  Device:   {gpu_name}\n"
        f"{'-' * 40}\n"
        f"  OUTPUT FILES\n"
        f"{'-' * 40}\n"
        f"  Model:    tsl51_{args.model}_{results['timestamp'].replace('-', '').replace(':', '')[:12]}.pt\n"
        f"  Report:   cv_{results['timestamp'].replace('-', '').replace(':', '')[:12]}.json\n"
    )
    ax6.text(0.02, 0.98, results_text, transform=ax6.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdedec", alpha=0.9, edgecolor="#e74c3c", linewidth=2))
    ax6.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        output_dir / f"results_{timestamp}.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#f8f9fa",
        edgecolor="none",
    )
    plt.close()

    text_report_path = output_dir / f"report_{timestamp}.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TSL-51 THAI SIGN LANGUAGE TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
        f.write("\nMODEL PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Model Type: {args.model.upper()}\n")
        f.write(f"  Hidden Dimensions: {args.hidden}\n")
        f.write(f"  Number of Layers: {args.layers}\n")
        f.write(f"  Dropout Rate: {args.dropout}\n")
        f.write(f"  Input Features: {results['input_dim']}\n")
        f.write(f"  Output Classes: {results['num_classes']}\n")
        params = estimate_params(args.model, args.hidden, args.layers, results["input_dim"], results["num_classes"])
        f.write(f"  Est. Parameters: ~{params / 1e6:.2f}M\n")
        f.write("\nTRAINING CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write("  Optimizer: AdamW\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write("  Weight Decay: 1e-4\n")
        f.write("  LR Scheduler: OneCycleLR\n")
        f.write(f"  Batch Size: {args.batch}\n")
        f.write(f"  Max Epochs: {args.epochs}\n")
        f.write(f"  Early Stopping: patience={args.patience}\n")
        f.write(f"  CV Folds: {args.folds}\n")
        f.write(f"  Augmentation: {args.augment if args.augment else 1}x\n")
        f.write(f"  Test Split: {args.test_split * 100:.0f}%\n")
        f.write("\nDATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total Samples: {results['num_samples']}\n")
        f.write(f"  Number of Classes: {results['num_classes']}\n")
        f.write(f"  Feature Dimension: {results['input_dim']}\n")
        if results.get("test_samples", 0) > 0:
            f.write(f"  Train Samples: {results['num_samples'] - results['test_samples']}\n")
            f.write(f"  Test Samples: {results['test_samples']}\n")
        f.write("\nPER-FOLD RESULTS\n")
        f.write("-" * 70 + "\n")
        for fold in fold_results:
            f.write(f"  Fold {fold['fold']}: {fold['accuracy'] * 100:.2f}%\n")
        f.write(f"\n  Best Fold: Fold {np.argmax([r['accuracy'] for r in fold_results]) + 1}\n")
        f.write(f"  Std Dev: {results['std_accuracy'] * 100:.2f}%\n")
        f.write("\nOVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  CV Average: {results['average_accuracy'] * 100:.2f}% (±{results['std_accuracy'] * 100:.2f}%)\n")
        f.write(f"  Overall Accuracy: {results['overall_accuracy'] * 100:.2f}%\n")
        f.write(f"  Precision: {results['precision'] * 100:.2f}%\n")
        f.write(f"  Recall: {results['recall'] * 100:.2f}%\n")
        f.write(f"  F1-Score: {results['f1_score'] * 100:.2f}%\n")
        if "test_results" in results:
            f.write("\nTEST SET RESULTS\n")
            f.write("-" * 70 + "\n")
            tr = results["test_results"]
            f.write(f"  Test Accuracy: {tr['test_accuracy'] * 100:.2f}%\n")
            f.write(f"  Test Precision: {tr['test_precision'] * 100:.2f}%\n")
            f.write(f"  Test Recall: {tr['test_recall'] * 100:.2f}%\n")
            f.write(f"  Test F1-Score: {tr['test_f1_score'] * 100:.2f}%\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"Text Report: {text_report_path}")
