"""Evaluate a trained TSL model checkpoint against a test NPZ file.

Usage:
    python eval_model.py --model models/tsl51_gru_20260518.pt --data .cache/tsl51/expert_all_45k.npz
    python eval_model.py --model model.pt --data test.npz --top-n 40 --worst-n 20 --no-plots
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Models (mirrors notebook Cell 3)
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    def __init__(self, input_dim=162, num_classes=51, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(self.dropout(self.norm(out[:, -1, :])))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class SmallGRUModel(nn.Module):
    def __init__(self, input_dim=162, num_classes=51, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        return self.fc(self.dropout(self.norm(out[:, -1, :])))

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPModel(nn.Module):
    def __init__(self, input_dim=162, num_classes=51, hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


MODEL_REGISTRY = {"gru": GRUModel, "gru_small": SmallGRUModel, "mlp": MLPModel}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_model(pt_path: str, device: torch.device):
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    mdl  = MODEL_REGISTRY[ckpt["model"]](
        ckpt["input_dim"], ckpt["num_classes"],
        cfg["hidden"], cfg["layers"], cfg["dropout"],
    ).to(device)
    mdl.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
    mdl.train(False)
    return mdl, ckpt


def load_npz(npz_path: str):
    raw = np.load(npz_path)
    return raw["X"].astype(np.float32), raw["y"].astype(np.int64), raw["classes"]


def run_inference(model, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 256):
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=batch_size, shuffle=False,
    )
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            prob = torch.softmax(model(Xb.to(device)), dim=1)
            preds.extend(prob.argmax(1).cpu().numpy())
            targets.extend(yb.numpy())
            probs.extend(prob.cpu().numpy())
    return np.array(preds), np.array(targets), np.array(probs)


def benchmark_latency(model, X: np.ndarray, device: torch.device):
    single = torch.tensor(X[:1]).to(device)
    for _ in range(20):                          # warmup
        with torch.no_grad():
            model(single)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(single)
        times.append((time.perf_counter() - t0) * 1000)
    times = times[20:]

    batch = torch.tensor(X[:min(512, len(X))]).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(batch)
    batch_ms = (time.perf_counter() - t0) * 1000

    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_std_ms":  float(np.std(times)),
        "batch512_ms":     batch_ms,
        "throughput_sps":  len(batch) / (batch_ms / 1000),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(preds, targets, probs, n_cls, cls_names, model_info, bench):
    acc  = accuracy_score(targets, preds) * 100
    f1   = f1_score(targets, preds, average="weighted", zero_division=0) * 100
    prec = precision_score(targets, preds, average="weighted", zero_division=0) * 100
    rec  = recall_score(targets, preds, average="weighted", zero_division=0) * 100
    top3 = top_k_accuracy_score(targets, probs, k=min(3, n_cls), labels=np.arange(n_cls)) * 100
    top5 = top_k_accuracy_score(targets, probs, k=min(5, n_cls), labels=np.arange(n_cls)) * 100

    model_mb = sum(p.numel() * p.element_size() for p in model_info["model"].parameters()) / 1e6

    print(f"\n{'='*60}")
    print(f"  MODEL  : {model_info['type'].upper()} | {model_info['params']:,} params | {model_mb:.2f} MB")
    print(f"  SAMPLES: {len(targets):,} | CLASSES: {n_cls}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Top-3 Acc : {top3:.2f}%")
    print(f"  Top-5 Acc : {top5:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1-Score  : {f1:.2f}%")
    print(f"{'─'*60}")
    print(f"  Latency (single) : {bench['latency_mean_ms']:.2f} ± {bench['latency_std_ms']:.2f} ms")
    print(f"  Batch-512        : {bench['batch512_ms']:.1f} ms")
    print(f"  Throughput       : {bench['throughput_sps']:,.0f} samples/sec")
    print(f"{'='*60}\n")
    return acc, f1, prec, rec, top3, top5


def print_per_class(preds, targets, n_cls, cls_names, cm, worst_n):
    pc_prec = precision_score(targets, preds, average=None, zero_division=0, labels=np.arange(n_cls))
    pc_rec  = recall_score(targets, preds, average=None, zero_division=0, labels=np.arange(n_cls))
    pc_f1   = f1_score(targets, preds, average=None, zero_division=0, labels=np.arange(n_cls))
    pc_n    = np.bincount(targets, minlength=n_cls)
    with np.errstate(divide="ignore", invalid="ignore"):
        pc_acc = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), 0.0)

    worst_idx = np.argsort(pc_f1)[:worst_n]
    print(f"{'─'*74}")
    print(f"  Worst {worst_n} Classes by F1")
    print(f"{'─'*74}")
    print(f"  {'Class':<26} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"{'─'*74}")
    for idx in worst_idx:
        print(f"  {cls_names[idx]:<26} {pc_n[idx]:>5} "
              f"{pc_acc[idx]*100:>6.1f}% {pc_prec[idx]*100:>6.1f}% "
              f"{pc_rec[idx]*100:>6.1f}% {pc_f1[idx]*100:>6.1f}%")

    confused = sorted(
        [(cm[i, j], i, j) for i in range(n_cls) for j in range(n_cls) if i != j and cm[i, j] > 0],
        reverse=True,
    )
    print(f"\n{'─'*58}")
    print("  Top-15 Most Confused Pairs")
    print(f"{'─'*58}")
    print(f"  {'True':<24} {'Predicted':<24} {'Count':>6}")
    print(f"{'─'*58}")
    for cnt, i, j in confused[:15]:
        print(f"  {cls_names[i]:<24} {cls_names[j]:<24} {cnt:>6}")
    print()

    return pc_f1, pc_acc, pc_n


def save_plots(preds, targets, probs, n_cls, cls_names, cm, pc_f1, pc_acc, pc_n,
               overall_acc, overall_f1, out_dir: Path, top_n_cm: int, worst_n: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Confusion matrix ──────────────────────────────────────────
    top_idx = np.argsort(pc_n)[::-1][:top_n_cm]
    cm_sub  = cm[np.ix_(top_idx, top_idx)].astype(float)
    row_sum = cm_sub.sum(1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_sub / row_sum, 0)

    fig1, ax1 = plt.subplots(figsize=(max(12, top_n_cm // 2), max(10, top_n_cm // 2)))
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, fraction=0.03)
    tick_lbl = [cls_names[i][:14] for i in top_idx]
    ax1.set_xticks(range(top_n_cm)); ax1.set_xticklabels(tick_lbl, rotation=90, fontsize=7)
    ax1.set_yticks(range(top_n_cm)); ax1.set_yticklabels(tick_lbl, fontsize=7)
    ax1.set_title(f"Confusion Matrix — Top {top_n_cm} classes (normalized)", fontsize=13)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    for i in range(top_n_cm):
        for j in range(top_n_cm):
            v = cm_norm[i, j]
            if v > 0.05:
                ax1.text(j, i, f"{v:.0%}", ha="center", va="center",
                         fontsize=5, color="white" if v > 0.55 else "black")
    plt.tight_layout()
    p = out_dir / "confusion_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    # ── 2. Per-class analysis ────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Per-Class Analysis", fontsize=14, fontweight="bold")

    worst_idx = np.argsort(pc_f1)[:worst_n]
    colors = plt.cm.RdYlGn(pc_f1[worst_idx])
    bars = axes[0].barh([cls_names[i][:20] for i in worst_idx],
                         pc_f1[worst_idx] * 100, color=colors, edgecolor="grey", height=0.7)
    axes[0].axvline(overall_f1, color="red", linestyle="--", linewidth=1.5,
                    label=f"Overall F1 {overall_f1:.1f}%")
    axes[0].set_xlabel("F1-Score (%)"); axes[0].set_title(f"Worst {worst_n} Classes by F1")
    axes[0].legend(fontsize=8)
    for bar, idx in zip(bars, worst_idx):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"N={pc_n[idx]}", va="center", fontsize=7)

    valid = pc_n > 0
    sc = axes[1].scatter(pc_n[valid], pc_acc[valid] * 100,
                          c=pc_f1[valid], cmap="RdYlGn", alpha=0.7, s=35, vmin=0, vmax=1)
    plt.colorbar(sc, ax=axes[1], label="F1")
    axes[1].axhline(overall_acc, color="red", linestyle="--", linewidth=1,
                    label=f"Overall {overall_acc:.1f}%")
    axes[1].set_xlabel("Sample count (N)"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy vs Sample Count per Class"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    p = out_dir / "per_class_analysis.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a TSL model checkpoint")
    parser.add_argument("--model",    required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data",     required=True, help="Path to test .npz file")
    parser.add_argument("--top-n",    type=int, default=30,  help="Top-N classes in confusion matrix")
    parser.add_argument("--worst-n",  type=int, default=15,  help="Worst-N classes to list")
    parser.add_argument("--batch",    type=int, default=256,  help="Inference batch size")
    parser.add_argument("--no-plots", action="store_true",    help="Skip saving plots")
    parser.add_argument("--out-dir",  default="results/eval", help="Output directory for plots")
    parser.add_argument("--test-split", type=float, default=0.0,
                        help="If > 0, split data and evaluate only on held-out portion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model, ckpt = load_model(args.model, device)
    cls_names = np.array(ckpt["classes"])
    n_cls     = ckpt["num_classes"]
    norm_mean = np.array(ckpt["mean"], dtype=np.float32)
    norm_std  = np.array(ckpt["std"],  dtype=np.float32)

    # ── Load data ───────────────────────────────────────────────────
    print(f"Loading data : {args.data}")
    X, y, cls_data = load_npz(args.data)

    # optional: evaluate on held-out split only
    if args.test_split > 0:
        _, X, _, y = train_test_split(X, y, test_size=args.test_split,
                                       stratify=y, random_state=42)
        print(f"Using held-out {args.test_split*100:.0f}%: {len(X):,} samples")

    # filter classes unknown to the model
    known = {str(c): i for i, c in enumerate(cls_names)}
    mask  = np.array([i < len(cls_data) and str(cls_data[y[i]]) in known
                      for i in range(len(y))], dtype=bool)
    if not mask.all():
        print(f"⚠  Dropping {(~mask).sum()} samples with unknown classes")
        X, y = X[mask], y[mask]

    X = np.nan_to_num((X - norm_mean) / norm_std)

    # ── Inference ───────────────────────────────────────────────────
    print(f"Running inference on {len(X):,} samples...")
    preds, targets, probs = run_inference(model, X, y, device, args.batch)

    # ── Latency benchmark ───────────────────────────────────────────
    bench = benchmark_latency(model, X, device)

    # ── Summary ─────────────────────────────────────────────────────
    model_info = {"model": model, "type": ckpt["model"], "params": model.num_params}
    acc, f1, prec, rec, top3, top5 = print_summary(preds, targets, probs, n_cls,
                                                     cls_names, model_info, bench)

    # ── Per-class / confusion ────────────────────────────────────────
    cm = confusion_matrix(targets, preds, labels=np.arange(n_cls))
    pc_f1, pc_acc, pc_n = print_per_class(preds, targets, n_cls, cls_names, cm, args.worst_n)

    # ── Plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Saving plots...")
        save_plots(preds, targets, probs, n_cls, cls_names, cm,
                   pc_f1, pc_acc, pc_n, acc, f1,
                   Path(args.out_dir), args.top_n, args.worst_n)
    else:
        print("Plots skipped (--no-plots)")


if __name__ == "__main__":
    main()
