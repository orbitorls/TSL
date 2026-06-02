from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FS_RUN = ROOT / ".tools" / "train_runs_25690531-131843" / "fs_e80_b64"
TSL_RUN = ROOT / ".tools" / "train_runs_25690531-175818" / "tsl51_e40_b32_s12000"


@dataclass
class BenchResult:
    name: str
    test_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    ece: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_sps_batch1: float
    throughput_sps_batch16: float
    model_size_mb: float
    load_time_ms: float
    num_classes: int
    robustness: dict[str, float]


def ece_score(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if not np.any(m):
            continue
        ece += float(np.abs(acc[m].mean() - conf[m].mean()) * (m.sum() / len(y_true)))
    return ece


def latency_stats(model, sample: np.ndarray, loops: int = 220) -> tuple[float, float, float, float]:
    for _ in range(20):
        _ = model.predict(sample, verbose=0)
    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        _ = model.predict(sample, verbose=0)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    mean_ms = float(arr.mean())
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99)), 1000.0 / mean_ms


def throughput_batch(model, sample: np.ndarray, loops: int = 100) -> float:
    for _ in range(8):
        _ = model.predict(sample, verbose=0)
    t0 = time.perf_counter()
    for _ in range(loops):
        _ = model.predict(sample, verbose=0)
    sec = time.perf_counter() - t0
    return (loops * sample.shape[0]) / sec


def robust_eval_fs(model, X_test_s: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    out = {}
    p = model.predict(X_test_s, verbose=0)
    out["base_acc"] = float(accuracy_score(y_test, p.argmax(axis=1)))

    noise = X_test_s + np.random.normal(0, 0.03, X_test_s.shape).astype(np.float32)
    p_noise = model.predict(noise, verbose=0)
    out["gaussian_noise_acc"] = float(accuracy_score(y_test, p_noise.argmax(axis=1)))

    scale = X_test_s * 0.92
    p_scale = model.predict(scale, verbose=0)
    out["feature_scale_0_92_acc"] = float(accuracy_score(y_test, p_scale.argmax(axis=1)))

    drop = X_test_s.copy()
    drop[:, ::7] = 0.0
    p_drop = model.predict(drop, verbose=0)
    out["feature_dropout_acc"] = float(accuracy_score(y_test, p_drop.argmax(axis=1)))
    return out


def robust_eval_tsl(model, X_test_s: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    out = {}
    p = model.predict(X_test_s, verbose=0)
    out["base_acc"] = float(accuracy_score(y_test, p.argmax(axis=1)))

    noise = X_test_s + np.random.normal(0, 0.02, X_test_s.shape).astype(np.float32)
    p_noise = model.predict(noise, verbose=0)
    out["gaussian_noise_acc"] = float(accuracy_score(y_test, p_noise.argmax(axis=1)))

    frame_drop = X_test_s.copy()
    frame_drop[:, ::6, :] = 0.0
    p_drop = model.predict(frame_drop, verbose=0)
    out["frame_dropout_acc"] = float(accuracy_score(y_test, p_drop.argmax(axis=1)))

    scale = X_test_s * 0.95
    p_scale = model.predict(scale, verbose=0)
    out["feature_scale_0_95_acc"] = float(accuracy_score(y_test, p_scale.argmax(axis=1)))
    return out


def benchmark_fingerspelling() -> tuple[BenchResult, np.ndarray]:
    cache = FS_RUN / "work" / "features" / "keypoints_cache.npz"
    data = np.load(cache, allow_pickle=True)
    X_train_raw, y_train_raw = data["X_train"], data["y_train"]
    X_test_raw, y_test = data["X_test"], data["y_test"]
    class_names = list(data["class_names"])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw, y_train_raw, test_size=0.12, stratify=y_train_raw, random_state=42
    )

    scaler = joblib.load(FS_RUN / "artifacts" / "fingerspelling" / "scaler.pkl")
    X_test_s = scaler.transform(X_test_raw).astype(np.float32)

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(FS_RUN / "artifacts" / "fingerspelling" / "model.keras")
    load_ms = (time.perf_counter() - t0) * 1000.0

    probs = model.predict(X_test_s, verbose=0)
    y_pred = probs.argmax(axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prw, rcw, f1w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    p50, p95, p99, sps1 = latency_stats(model, X_test_s[:1])
    sps16 = throughput_batch(model, np.repeat(X_test_s[:1], 16, axis=0))

    result = BenchResult(
        name="Fingerspelling",
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(pr),
        recall_macro=float(rc),
        f1_macro=float(f1),
        precision_weighted=float(prw),
        recall_weighted=float(rcw),
        f1_weighted=float(f1w),
        ece=ece_score(probs, y_test),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        throughput_sps_batch1=sps1,
        throughput_sps_batch16=sps16,
        model_size_mb=(FS_RUN / "artifacts" / "fingerspelling" / "model.keras").stat().st_size / (1024 * 1024),
        load_time_ms=load_ms,
        num_classes=len(class_names),
        robustness=robust_eval_fs(model, X_test_s, y_test),
    )
    cm = confusion_matrix(y_test, y_pred)
    return result, cm


def benchmark_tsl51() -> tuple[BenchResult, np.ndarray]:
    cache = TSL_RUN / "work" / "features" / "tsl51_features.npz"
    data = np.load(cache, allow_pickle=True)
    X_all, y_all = data["X"], data["y"]
    class_names = list(data["class_names"])

    cls_ids, cls_counts = np.unique(y_all, return_counts=True)
    keep_ids = {int(i) for i, c in zip(cls_ids, cls_counts) if int(c) >= 2}
    if len(keep_ids) < len(cls_ids):
        mask = np.asarray([int(v) in keep_ids for v in y_all], dtype=bool)
        X_all = X_all[mask]
        y_all = y_all[mask]
        remap = {old: new for new, old in enumerate(sorted(keep_ids))}
        y_all = np.asarray([remap[int(v)] for v in y_all], dtype=np.int32)
        class_names = [class_names[i] for i in sorted(keep_ids)]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.10, stratify=y_all, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.111, stratify=y_tmp, random_state=42
    )

    scaler = joblib.load(TSL_RUN / "artifacts" / "tsl51" / "tsl51_scaler.pkl")
    X_test_s = scaler.transform(X_test.reshape(-1, 162)).reshape(-1, 60, 162).astype(np.float32)

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(TSL_RUN / "artifacts" / "tsl51" / "tsl51_model.keras")
    load_ms = (time.perf_counter() - t0) * 1000.0

    probs = model.predict(X_test_s, verbose=0)
    y_pred = probs.argmax(axis=1)

    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    prw, rcw, f1w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    p50, p95, p99, sps1 = latency_stats(model, X_test_s[:1])
    sps16 = throughput_batch(model, np.repeat(X_test_s[:1], 16, axis=0))

    result = BenchResult(
        name="TSL-51",
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(pr),
        recall_macro=float(rc),
        f1_macro=float(f1),
        precision_weighted=float(prw),
        recall_weighted=float(rcw),
        f1_weighted=float(f1w),
        ece=ece_score(probs, y_test),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        throughput_sps_batch1=sps1,
        throughput_sps_batch16=sps16,
        model_size_mb=(TSL_RUN / "artifacts" / "tsl51" / "tsl51_model.keras").stat().st_size / (1024 * 1024),
        load_time_ms=load_ms,
        num_classes=len(class_names),
        robustness=robust_eval_tsl(model, X_test_s, y_test),
    )
    cm = confusion_matrix(y_test, y_pred)
    return result, cm


def write_outputs(fs: BenchResult, tsl: BenchResult, fs_cm: np.ndarray, tsl_cm: np.ndarray) -> None:
    out_json = REPORT_DIR / "benchmark-standard-report.json"
    out_csv = REPORT_DIR / "benchmark-standard-summary.csv"
    out_html = REPORT_DIR / "benchmark-standard-report.html"
    out_model_html = REPORT_DIR / "model-report-fs-tsl51.html"

    payload = {
        "fingerspelling": fs.__dict__,
        "tsl51": tsl.__dict__,
        "confusion_matrices": {
            "fingerspelling_shape": list(fs_cm.shape),
            "tsl51_shape": list(tsl_cm.shape),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","test_acc","f1_macro","f1_weighted","ece","p50_ms","p95_ms","p99_ms","sps_batch1","sps_batch16","model_size_mb","load_time_ms","num_classes"])
        for r in (fs, tsl):
            w.writerow([
                r.name,
                f"{r.test_accuracy:.6f}",
                f"{r.f1_macro:.6f}",
                f"{r.f1_weighted:.6f}",
                f"{r.ece:.6f}",
                f"{r.latency_p50_ms:.3f}",
                f"{r.latency_p95_ms:.3f}",
                f"{r.latency_p99_ms:.3f}",
                f"{r.throughput_sps_batch1:.2f}",
                f"{r.throughput_sps_batch16:.2f}",
                f"{r.model_size_mb:.2f}",
                f"{r.load_time_ms:.2f}",
                r.num_classes,
            ])

    def robust_rows(r: BenchResult) -> str:
        return "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in r.robustness.items())

    metric_rows = [
        ("Test Accuracy", fs.test_accuracy, tsl.test_accuracy, "higher"),
        ("Macro F1", fs.f1_macro, tsl.f1_macro, "higher"),
        ("Weighted F1", fs.f1_weighted, tsl.f1_weighted, "higher"),
        ("Calibration ECE", fs.ece, tsl.ece, "lower"),
        ("Latency p50 (ms)", fs.latency_p50_ms, tsl.latency_p50_ms, "lower"),
        ("Latency p95 (ms)", fs.latency_p95_ms, tsl.latency_p95_ms, "lower"),
        ("Latency p99 (ms)", fs.latency_p99_ms, tsl.latency_p99_ms, "lower"),
        ("Throughput batch1", fs.throughput_sps_batch1, tsl.throughput_sps_batch1, "higher"),
        ("Throughput batch16", fs.throughput_sps_batch16, tsl.throughput_sps_batch16, "higher"),
        ("Model Size (MB)", fs.model_size_mb, tsl.model_size_mb, "lower"),
        ("Load Time (ms)", fs.load_time_ms, tsl.load_time_ms, "lower"),
    ]

    compare_rows = []
    compare_bars = []
    for label, a, b, direction in metric_rows:
        fs_win = (a >= b) if direction == "higher" else (a <= b)
        winner = "Fingerspelling" if fs_win else "TSL-51"
        compare_rows.append(
            f"<tr><td>{label}</td><td>{a:.4f}</td><td>{b:.4f}</td><td><strong>{winner}</strong></td></tr>"
        )
        m = max(a, b) or 1.0
        fs_pct = (a / m) * 100.0
        tsl_pct = (b / m) * 100.0
        compare_bars.append(
            f"""
            <div class='bar-block'>
              <div class='bar-title'>{label}</div>
              <div class='bar-pair'>
                <span class='tag fs'>FS</span><div class='bar-track'><div class='bar-fill fs' style='width:{fs_pct:.1f}%'></div></div><span class='bar-val'>{a:.4f}</span>
              </div>
              <div class='bar-pair'>
                <span class='tag tsl'>TSL</span><div class='bar-track'><div class='bar-fill tsl' style='width:{tsl_pct:.1f}%'></div></div><span class='bar-val'>{b:.4f}</span>
              </div>
            </div>
            """
        )

    html = f"""<!doctype html>
<html lang='th'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <title>TSL Unified Model + Benchmark Report</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --ink: #1a1712;
      --muted: #5f584d;
      --line: #d6cdbd;
      --card: #fffcf6;
      --teal: #0f766e;
      --teal-dark: #0a4f4a;
      --amber: #b45309;
      --blue: #1d4ed8;
      --radius: 16px;
      --shadow: 0 8px 30px rgba(41, 33, 24, 0.09);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Aptos", "Bahnschrift", "Trebuchet MS", sans-serif;
      background:
        radial-gradient(1000px 480px at -10% -10%, #d9efe8 0%, transparent 58%),
        radial-gradient(840px 420px at 110% -8%, #fde9d3 0%, transparent 60%),
        var(--bg);
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 16px 42px; }}
    .hero {{
      border-radius: 24px;
      padding: 26px 22px;
      background: linear-gradient(130deg, var(--teal), var(--teal-dark));
      color: #edfffc;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{ margin: 0; font-family: "Cambria", "Palatino Linotype", serif; font-size: clamp(1.6rem, 3.2vw, 2.5rem); }}
    .hero p {{ margin: 10px 0 0; color: #c7f5ed; max-width: 980px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .chip {{ font-size: .88rem; border: 1px solid rgba(255,255,255,.28); background: rgba(255,255,255,.12); border-radius: 999px; padding: 5px 10px; }}
    .grid {{ display: grid; gap: 12px; }}
    .kpi {{ margin-top: 14px; grid-template-columns: repeat(4, minmax(0,1fr)); }}
    .card {{
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: var(--card);
      box-shadow: var(--shadow);
      padding: 14px;
    }}
    .kpi .card .t {{ color: var(--muted); font-size: .83rem; }}
    .kpi .card .v {{ margin-top: 5px; font-size: 1.25rem; font-weight: 700; color: var(--teal); }}
    .split-2 {{ margin-top: 14px; grid-template-columns: 1fr 1fr; }}
    h2 {{ margin: 0; font-size: 1.18rem; font-family: "Cambria", "Palatino Linotype", serif; }}
    .head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }}
    .badge {{ font-size: .8rem; border-radius: 999px; border: 1px solid; padding: 5px 10px; }}
    .good {{ color: #166534; border-color: #86efac; background: #f0fdf4; }}
    .warn {{ color: #92400e; border-color: #fbd38d; background: #fff7ed; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .94rem; }}
    td, th {{ border-bottom: 1px dashed var(--line); padding: 8px 4px; text-align: left; vertical-align: top; }}
    td:first-child {{ width: 210px; color: var(--muted); font-weight: 600; }}
    .mono {{ font-family: "Consolas", "Courier New", monospace; font-size: .84rem; word-break: break-all; }}
    .bars {{ margin-top: 14px; }}
    .bar-block {{ margin-bottom: 12px; }}
    .bar-title {{ font-weight: 700; font-size: .9rem; margin-bottom: 5px; }}
    .bar-pair {{ display: grid; grid-template-columns: 32px 1fr 88px; align-items: center; gap: 8px; margin-bottom: 5px; }}
    .tag {{ text-align: center; font-weight: 700; font-size: .76rem; border-radius: 6px; padding: 2px 4px; color: #fff; }}
    .tag.fs {{ background: var(--teal); }}
    .tag.tsl {{ background: var(--amber); }}
    .bar-track {{ height: 10px; background: #ece4d6; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .bar-fill.fs {{ background: linear-gradient(90deg, #159987, var(--teal)); }}
    .bar-fill.tsl {{ background: linear-gradient(90deg, #cf6f18, var(--amber)); }}
    .bar-val {{ text-align: right; font-family: "Consolas", monospace; font-size: .84rem; color: var(--muted); }}
    .foot {{ margin-top: 14px; color: var(--muted); font-size: .9rem; }}
    @media (max-width: 980px) {{
      .kpi {{ grid-template-columns: 1fr 1fr; }}
      .split-2 {{ grid-template-columns: 1fr; }}
      td:first-child {{ width: 150px; }}
    }}
    @media (max-width: 640px) {{
      .kpi {{ grid-template-columns: 1fr; }}
      .bar-pair {{ grid-template-columns: 28px 1fr 74px; }}
    }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='hero'>
      <h1>Thai Sign Language Unified Report</h1>
      <p>รวมรายงานโมเดลและ benchmark มาตรฐานในหน้าเดียว: คุณภาพการจำแนก, calibration, latency/throughput, footprint และ robustness ของทั้ง Fingerspelling และ TSL-51</p>
      <div class='chips'>
        <span class='chip'>Generated: 2026-05-31</span>
        <span class='chip'>Pipeline: scripts/benchmark_standard_suite.py</span>
        <span class='chip'>Confusion matrix: {fs_cm.shape[0]}x{fs_cm.shape[1]} / {tsl_cm.shape[0]}x{tsl_cm.shape[1]}</span>
      </div>
    </section>

    <section class='grid kpi'>
      <article class='card'><div class='t'>Fingerspelling Accuracy</div><div class='v'>{fs.test_accuracy * 100:.2f}%</div></article>
      <article class='card'><div class='t'>TSL-51 Accuracy</div><div class='v'>{tsl.test_accuracy * 100:.2f}%</div></article>
      <article class='card'><div class='t'>Fingerspelling Throughput (B16)</div><div class='v'>{fs.throughput_sps_batch16:.2f}</div></article>
      <article class='card'><div class='t'>TSL-51 Throughput (B16)</div><div class='v'>{tsl.throughput_sps_batch16:.2f}</div></article>
    </section>

    <section class='grid split-2'>
      <article class='card'>
        <div class='head'><h2>Model A: Fingerspelling</h2><span class='badge good'>15 classes</span></div>
        <table>
          <tr><td>Run ID</td><td class='mono'>{FS_RUN.name}</td></tr>
          <tr><td>Test Accuracy</td><td><strong>{fs.test_accuracy:.6f}</strong></td></tr>
          <tr><td>Macro F1</td><td>{fs.f1_macro:.6f}</td></tr>
          <tr><td>Weighted F1</td><td>{fs.f1_weighted:.6f}</td></tr>
          <tr><td>ECE</td><td>{fs.ece:.6f}</td></tr>
          <tr><td>Latency p50/p95/p99 (ms)</td><td>{fs.latency_p50_ms:.3f} / {fs.latency_p95_ms:.3f} / {fs.latency_p99_ms:.3f}</td></tr>
          <tr><td>Throughput batch1/batch16</td><td>{fs.throughput_sps_batch1:.2f} / {fs.throughput_sps_batch16:.2f}</td></tr>
          <tr><td>Model size/load time</td><td>{fs.model_size_mb:.2f} MB / {fs.load_time_ms:.2f} ms</td></tr>
          <tr><td>Model file</td><td class='mono'>{FS_RUN / "artifacts" / "fingerspelling" / "model.keras"}</td></tr>
        </table>
      </article>
      <article class='card'>
        <div class='head'><h2>Model B: TSL-51</h2><span class='badge warn'>{tsl.num_classes} classes benchmarked</span></div>
        <table>
          <tr><td>Run ID</td><td class='mono'>{TSL_RUN.name}</td></tr>
          <tr><td>Test Accuracy</td><td><strong>{tsl.test_accuracy:.6f}</strong></td></tr>
          <tr><td>Macro F1</td><td>{tsl.f1_macro:.6f}</td></tr>
          <tr><td>Weighted F1</td><td>{tsl.f1_weighted:.6f}</td></tr>
          <tr><td>ECE</td><td>{tsl.ece:.6f}</td></tr>
          <tr><td>Latency p50/p95/p99 (ms)</td><td>{tsl.latency_p50_ms:.3f} / {tsl.latency_p95_ms:.3f} / {tsl.latency_p99_ms:.3f}</td></tr>
          <tr><td>Throughput batch1/batch16</td><td>{tsl.throughput_sps_batch1:.2f} / {tsl.throughput_sps_batch16:.2f}</td></tr>
          <tr><td>Model size/load time</td><td>{tsl.model_size_mb:.2f} MB / {tsl.load_time_ms:.2f} ms</td></tr>
          <tr><td>Model file</td><td class='mono'>{TSL_RUN / "artifacts" / "tsl51" / "tsl51_model.keras"}</td></tr>
        </table>
      </article>
    </section>

    <section class='grid split-2'>
      <article class='card'>
        <div class='head'><h2>Benchmark Comparison Matrix</h2><span class='badge good'>Standard Suite</span></div>
        <table>
          <tr><th>Metric</th><th>Fingerspelling</th><th>TSL-51</th><th>Winner</th></tr>
          {''.join(compare_rows)}
        </table>
      </article>
      <article class='card'>
        <div class='head'><h2>Visualization</h2><span class='badge warn'>Relative Bars</span></div>
        <div class='bars'>{''.join(compare_bars)}</div>
      </article>
    </section>

    <section class='grid split-2'>
      <article class='card'>
        <div class='head'><h2>Robustness: Fingerspelling</h2><span class='badge good'>Perturbation</span></div>
        <table>{robust_rows(fs)}</table>
      </article>
      <article class='card'>
        <div class='head'><h2>Robustness: TSL-51</h2><span class='badge warn'>Perturbation</span></div>
        <table>{robust_rows(tsl)}</table>
      </article>
    </section>

    <p class='foot'>Outputs: {out_model_html.name} | {out_html.name} | {out_csv.name} | {out_json.name}</p>
  </main>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")
    out_model_html.write_text(html, encoding="utf-8")


def main() -> int:
    fs, fs_cm = benchmark_fingerspelling()
    tsl, tsl_cm = benchmark_tsl51()
    write_outputs(fs, tsl, fs_cm, tsl_cm)
    print("OK reports generated:")
    print(REPORT_DIR / "benchmark-standard-report.html")
    print(REPORT_DIR / "benchmark-standard-summary.csv")
    print(REPORT_DIR / "benchmark-standard-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
