# Thai Fingerspelling Recognition
**จำแนกตัวอักษรภาษาไทย (ฟิงเกอร์สเปลลิง) ด้วย AI**

Recognise 15 one-stage Thai consonant hand-shapes in real time from your webcam.

| | |
|---|---|
| **Dataset** | One-Stage-TFS ([Mendeley CC BY 4.0](https://data.mendeley.com/datasets/rknd3wbz42/1)) |
| **Classes** | 15 Thai consonant finger-spellings |
| **Input** | 63 MediaPipe Hand keypoints (wrist-relative, scale-normalised) |
| **Model** | Dense DNN — trains in ~5 minutes on Colab GPU |

---

## Rust workspace (primary)

The Rust migration lives at the repo root (`Cargo.toml`, `crates/*`). CI runs
`cargo test --workspace` via [`.github/workflows/rust.yml`](.github/workflows/rust.yml).

```bash
# Install Rust: https://rustup.rs/
cargo test --workspace
cargo build -p tsl-cli --release
cargo run -p tsl-cli --bin tsl-train -- --help
```

| Crate | Role |
|-------|------|
| `tsl-core` | 63-D / 162×60 feature contract, scaler, labels (golden tests) |
| `tsl-ingest` | One-Stage-TFS zip and TSL-51 metadata ingest |
| `tsl-ml` | Candle model skeletons + training stub |
| `tsl-cli` | `tsl-demo-fs`, `tsl-demo-51`, `tsl-train`, `tsl-ingest` |
| `tsl-infer` | Tract ONNX load, EMA, artifact validation |
| `tsl-vision` | `LandmarkBackend` + stub/ONNX skeleton |

### Interim Rust state

- **Ready:** `tsl-core` golden parity (`tests/golden/`, tolerance `1e-4`).
- **Demos:** `tsl-demo-fs --dry-run` / `tsl-demo-51 --dry-run` smoke-test features;
  full webcam + ONNX models still require artifact export (see `scripts/convert_tflite_to_onnx.md`).
- **Training:** `tsl-train --feature-cache …` runs a CPU Candle smoke loop from Python NPZ caches; use **legacy Python** (`train_local_all.py`) for full GPU training.
- **CI:** GitHub Actions runs `python scripts/export_golden.py` then `cargo test --workspace` on push/PR.

### Verification (repo root)

Run from the repository root (`D:\TSL` or `/mnt/d/TSL`). Matches
[`.github/workflows/rust.yml`](.github/workflows/rust.yml) for Rust; Python legacy
tests are local-only unless you add them to CI.

| Tier | When | Commands |
|------|------|----------|
| **Preflight** | After clone / before a long build | See below |
| **Smoke** | Quick CLI + contract checks | See below |
| **Full** | Pre-merge / release parity | See below |

**Preflight** (no `cargo` link step; catches Python contract drift early):

```bash
python scripts/export_golden.py
cd python-legacy && python -m pytest -q
```

Optional GPU training env check (WSL or Linux with TensorFlow installed):

```bash
cd python-legacy
python scripts/train_local_all.py --preflight
```

**Smoke** (Rust binaries; requires a working `cargo build`):

```bash
cargo run -p tsl-cli --bin tsl-demo-fs -- --dry-run
cargo run -p tsl-cli --bin tsl-demo-51 -- --dry-run
cargo run -p tsl-cli --bin tsl-train -- --help
```

**Full** (same as CI Rust job + legacy unit tests):

```bash
python scripts/export_golden.py
cargo test --workspace
cargo build --release -p tsl-cli
cd python-legacy && python -m pytest -q
```

**Windows GNU toolchain:** `rust-toolchain.toml` pins `stable-x86_64-pc-windows-gnu`.
If `cargo test` reports `dlltool.exe` not found, prepend rustup’s bundled MinGW
`self-contained` directory to `PATH` (PowerShell example):

```powershell
$gnu = "$env:USERPROFILE\.rustup\toolchains\stable-x86_64-pc-windows-gnu\lib\rustlib\x86_64-pc-windows-gnu\bin\self-contained"
$env:PATH = "$gnu;$env:PATH"
cargo test --workspace
```

If `dlltool` still fails with `CreateProcess`, the profile path likely contains
spaces (known `rustc` + GNU issue). Use GitHub Actions, install MSVC Build Tools
and `rustup default stable-x86_64-pc-windows-msvc`, or set `RUSTUP_HOME` under a
short path without spaces.

**WSL GPU smoke** (after `config.local.json` or CLI flags are set):

```bash
cd /mnt/d/TSL/python-legacy && source ~/venvs/tsl/bin/activate
python scripts/train_local_all.py --preflight
python scripts/train_local_all.py --tracks both --max-tsl51-samples 512 --tsl51-epochs 2
```

### Training TUI (Textual)

Interactive menu for preflight, smoke/full train, NPZ cache-only builds, ingest, and
`config.local.json` editing. Training still runs in a **subprocess** (`train_local_all.py`)
so the TUI stays fast and does not import TensorFlow.

```bash
cd python-legacy
pip install -r requirements-tui.txt   # once
# WSL (recommended for GPU):
source ~/venvs/tsl/bin/activate
export PYTHONPATH=src
python -m tsl_tui
# or:
python scripts/tsl_tui.py
```

From **Windows PowerShell** (native, no GPU): the home screen shows a WSL command hint.
Use WSL for real training.

| Key | Action |
|-----|--------|
| `p` | Preflight (`--preflight`) |
| `s` | Smoke train (512 TSL-51 samples, 2 epochs) |
| `f` | Full train (`--tracks both`, confirm first) |
| `c` | Cache only → FS (`--fs-only-cache`) or TSL-51 (`--tsl51-only-cache`) |
| `i` | Ingest form → `ingest_local.py` |
| `,` | Edit `config.local.json` |
| `q` | Quit |

### Web Translate (sign → Thai text, recommended)

Real-time **ภาษามือ → ข้อความไทย** with a Next.js UI and FastAPI backend. Shared
inference core lives in `python-legacy/src/tsl_translate/` (also used by Streamlit).

**Tracks:** Fingerspelling (63-D) and TSL-51 (162×60 + motion gate).

**Features:** auto-discover artifacts, live webcam from the browser, transcript
panel (debounced commit), confidence chart, Thai UI.

**One-shot dev (Windows PowerShell, two terminals):**

```powershell
# Terminal 1 — API (from repo root)
cd python-legacy
pip install -r requirements-translate.txt
python ..\scripts\tsl_translate_api.py

# Terminal 2 — frontend
cd web\translate
copy .env.local.example .env.local
npm install
npm run dev
```

Or: `powershell -File scripts\tsl_translate_dev.ps1` (opens API in a new window, runs UI in the current one).

Open http://localhost:3000 — API health: http://127.0.0.1:8000/health

Checklist:
1. Train or copy a complete artifact set under `artifacts/fingerspelling/` or `artifacts/tsl51/`.
2. Allow **browser** camera permission (not only Python).
3. Do not mix tracks (`fingerspelling` vs `tsl51` artifacts).
4. Set `NEXT_PUBLIC_API_URL` if the API is not on `http://127.0.0.1:8000`.

### Web Inference App (Streamlit, fallback)

Legacy Streamlit console — same models, server-side OpenCV webcam:

```bash
cd python-legacy
pip install -r requirements.txt
python -m streamlit run web_infer_app.py
```

Or from repo root: `python scripts/tsl_web_app.py`

---

## Canonical layout (datasets, artifacts, golden tests)

Use these paths in docs, ingest, and local training (avoid `_ready`, `_local`, and
other one-off folder names):

```text
data/
  fingerspelling/          # One-Stage-TFS after ingest (class folders under Training set)
  tsl51/metadata/          # expert_metadata.csv, user_sign_metadata.csv
runtime/                   # optional WSL scratch (NPZ caches via train_local_all --work-root)
artifacts/
  fingerspelling/          # model.keras, labels.json, scaler.pkl, …
  tsl51/                   # tsl51_model.keras, tsl51_labels.json, …
tests/golden/              # JSON only — Rust parity fixtures (not pytest)
python-legacy/tests/       # pytest for legacy Python
scripts/export_golden.py   # sole writer for tests/golden/ (CI canonical)
```

Regenerate golden vectors from the repo root: `python scripts/export_golden.py`.
`python-legacy/scripts/export_golden.py` redirects to that script.

---

## Legacy Python (`python-legacy/`)

Notebooks, scripts, webcam demos, and pytest live under `python-legacy/` with the
same layout as before (`src/`, `scripts/`, `notebooks/`, `tests/`, `webcam_*.py`).
Run commands from that directory or prefix paths with `python-legacy/`.

Optional local paths: copy `python-legacy/config.local.json.example` to
`python-legacy/config.local.json` (gitignored) for WSL training defaults.

---

## Quick-start (4 steps, legacy Python)

### Step 1 — Download the dataset
1. Open https://data.mendeley.com/datasets/rknd3wbz42/1 in your browser.
2. Click **Download All** → saves `One-Stage-TFS Thai One-Stage Fingerspelling Dataset.zip` (~650 MB).
3. Upload the ZIP to your Google Drive (e.g. `MyDrive/datasets/One-Stage-TFS.zip`).

### Step 2 — Train in Google Colab
1. Upload `python-legacy/notebooks/train_all_datasets.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. In the config section, set `REPO_URL` and `FS_ZIP_DRIVE_PATH`.
4. **Runtime → Run all**.
5. The notebook trains both tracks in sequence and saves artifacts to Google Drive.
6. At the end, download fingerspelling artifacts:
   - `model.keras`
   - `model.tflite`
   - `labels.json`
   - `scaler.pkl`
 - `model_manifest.json` (optional metadata)
7. Download TSL-51 artifacts:
 - `tsl51_model.keras`
 - `tsl51_model.tflite` (when conversion succeeds)
 - `tsl51_labels.json`
 - `tsl51_scaler.pkl`
 - `tsl51_model_manifest.json` (optional metadata)

### Step 2 (local) — Prepare dataset on this machine first
If you want to stay local (no Colab), run the local ingest script:

```bash
cd python-legacy
python -m pip install -r requirements.txt
pip install pandas huggingface_hub tqdm  # only for local TSL-51 metadata ingest

# 1) One-Stage-TFS zip (fingerspelling track)
python scripts/ingest_local.py fingerspelling --zip "D:\\datasets\\One-Stage-TFS.zip" --output "D:\\TSL\\data\\fingerspelling"

# If you already extracted the zip
python scripts/ingest_local.py fingerspelling --dataset-root "D:\\datasets\\One-Stage-TFS" --output "D:\\TSL\\data\\fingerspelling"

# 2) TSL-51 metadata (downloads only metadata from HF by default)
python scripts/ingest_local.py tsl51 --output "D:\\TSL\\data\\tsl51"
```

For TSL-51, if you already downloaded metadata files to a local folder, use:

```bash
python scripts/ingest_local.py tsl51 --source "D:\\tsl51_raw_dump" --output "D:\\TSL\\data\\tsl51"
```

After ingest, point training at `data/fingerspelling` and `data/tsl51/metadata/`.
Older folder names (`one_stage_tfs_ready`, `tsl51_raw_local`, …) still work if you pass them
explicitly via CLI or `config.local.json`.

### Step 2 (fast local GPU) — Train both datasets on this PC
For the RTX 4060 path, use WSL2 Ubuntu. Native Windows TensorFlow GPU is not the
recommended route for this repo.

Canonical local layout (bulk under `data/` is gitignored):

```text
D:\TSL\data\fingerspelling\          # One-Stage-TFS (Training set / Test set)
D:\TSL\data\tsl51\metadata\          # expert_metadata.csv, user_sign_metadata.csv
D:\TSL\artifacts\fingerspelling\     # model.keras, labels.json, scaler.pkl, …
D:\TSL\artifacts\tsl51\              # tsl51_model.keras, tsl51_labels.json, …
```

Install WSL2 once from PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Restart if Windows asks, then open Ubuntu and run:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git
cd /mnt/d/TSL/python-legacy

python3.11 -m venv ~/venvs/tsl
source ~/venvs/tsl/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-local-gpu.txt
python scripts/train_local_all.py --preflight
```

**Training TUI** (Textual menu; does not import TensorFlow in the UI process):

```bash
cd python-legacy
pip install -r requirements-tui.txt
# PYTHONPATH must include src/
export PYTHONPATH=src   # Linux/WSL
python -m tsl_tui
```

From repo root you can also run `python scripts/tsl_tui.py`. On Windows native, the home screen shows a WSL command hint; use WSL for GPU training.

Quick smoke test:

```bash
python scripts/train_local_all.py --tracks tsl51 --max-tsl51-samples 512 --tsl51-epochs 2
```

Full local GPU training for both datasets (from repo root orchestrator or directly):

```bash
# From Windows PowerShell (prints the WSL command):
powershell -File scripts/train_local_wsl.ps1

# Or inside WSL:
bash /mnt/d/TSL/scripts/train_local_wsl.sh

# Or directly in python-legacy (copy config.local.json.example → config.local.json):
python scripts/train_local_all.py \
  --tracks both \
  --work-root ~/tsl_training \
  --artifact-dir /mnt/d/TSL/artifacts \
  --fs-dataset-root /mnt/d/TSL/data/fingerspelling \
  --tsl51-metadata-dir /mnt/d/TSL/data/tsl51/metadata
```

The script copies heavy inputs/caches into `~/tsl_training` inside WSL for faster I/O,
then writes final model files to:

```text
D:\TSL\artifacts\fingerspelling\
D:\TSL\artifacts\tsl51\
```

Use `python-legacy/scripts/train_local_all.py` as the single local trainer. Use
`python-legacy/notebooks/train_all_datasets.ipynb` as the single Colab trainer.

### Step 3 — Download the Thai font
The webcam overlay needs a Thai-capable font.

1. Go to https://fonts.google.com/noto/specimen/Noto+Sans+Thai
2. Click **Download family** → extract → find `NotoSansThai-Regular.ttf`.
3. Copy it to `assets/NotoSansThai-Regular.ttf` (repo root `assets/`; legacy demo: `python-legacy/webcam_demo.py`).

> **Alternative:** Sarabun — https://fonts.google.com/specimen/Sarabun  
> Save as `assets/Sarabun-Regular.ttf`.

### Step 4 — Run the live demo
```
cd python-legacy
pip install -r requirements.txt

# Place model.keras, model.tflite, labels.json, scaler.pkl in repo root or python-legacy, then:
python webcam_demo.py
```
Hold one hand up in front of the camera, signing any of the 15 consonants.
Press **`q`** to quit.

**Optional (lighter, inference-only):** install `tflite-runtime` instead of full TensorFlow.
`python-legacy/webcam_demo.py` auto-detects it and uses `model.tflite` when present.
```
pip install tflite-runtime
```

#### CLI flags
| Flag | Default | What it does |
|------|---------|--------------|
| `--model PATH` | `model.keras` | Path to Keras model |
| `--labels PATH` | `labels.json` | Class-index → Thai letter map |
| `--scaler PATH` | `scaler.pkl` | Fitted `StandardScaler` from training |
| `--cam N` | `0` | Webcam index |
| `--use-tflite` | auto | Force TFLite backend (otherwise auto-detected from `model.tflite`) |
| `--threshold F` | `0.7` | Minimum smoothed top-1 probability to show a letter (else shows `?`) |
| `--smoothing-alpha F` | `0.4` | EMA weight for new frames (1.0 = no smoothing) |
| `--frame-skip N` | `1` | Run model inference every N frames (landmarks still drawn each frame) |
| `--top-k N` | `2` | How many top candidates to display under the big letter |
| `--save-log PATH` | off | Append `timestamp_iso,label,confidence,top2_label,top2_confidence` per prediction |

---

## Project structure

```
D:\TSL\
├── Cargo.toml                     ← Rust workspace
├── crates/                        ← tsl-core, tsl-ingest, tsl-ml, tsl-cli, …
├── data/                          ← datasets (gitignored bulk): fingerspelling, tsl51/
├── artifacts/                     ← trained outputs (gitignored)
├── scripts/                       ← export_golden.py, train_local_wsl.* (orchestration)
├── tests/golden/                  ← JSON golden vectors only (not pytest)
├── .github/workflows/rust.yml     ← CI: cargo test --workspace
├── python-legacy/                 ← legacy Python (unchanged layout inside)
│   ├── src/
│   │   ├── keypoints.py           ← fingerspelling keypoints (63-D)
│   │   └── sequence_keypoints.py  ← TSL-51 Holistic sequences (162-D × 60)
│   ├── scripts/
│   ├── notebooks/
│   │   └── train_all_datasets.ipynb
│   ├── tests/                     ← pytest only (not repo-root tests/)
│   ├── webcam_demo.py             ← fingerspelling live demo
│   ├── webcam_word_demo.py        ← TSL-51 word sign live demo
│   ├── web_infer_app.py           ← Streamlit web inference app
│   ├── requirements.txt
│   └── requirements-dev.txt
├── assets/
│   └── NotoSansThai-Regular.ttf   ← Thai font (download separately, see Step 3)
├── DATASETS.md
└── README.md
```

Trained files go under `artifacts/fingerspelling/` and `artifacts/tsl51/` (see canonical layout above). Legacy demos may also read models from repo root or `python-legacy/` if you copy them there.

See `DATASETS.md` for the exact dataset → notebook → webcam demo mapping.
The two tracks are separate; do not reuse a scaler, labels file, or model from
one track in the other demo.

Use `python-legacy/notebooks/train_all_datasets.ipynb` as the primary Colab trainer for both tracks.

---

## Testing

**Rust parity:** from repo root, `python scripts/export_golden.py` then `cargo test --workspace` (fixtures in `tests/golden/` only — not pytest).

**Legacy Python:** install dev deps and run pytest under `python-legacy/tests/`:

```
cd python-legacy
pip install -r requirements-dev.txt
pytest -q
```

The tests verify the keypoint normalisation contract (wrist at origin, hand span = 1.0,
`None` on degenerate input, correct hand selection when two hands are visible) — this is
the shared logic that BOTH training and inference depend on.

---

## TSL-51 word-level track (Thai word signs)

Recognise **51 Thai Sign Language word gestures** from a 60-frame Holistic landmark sequence — a separate track from fingerspelling above.

| | |
|---|---|
| **Dataset** | [Namonpas/thai-sign-language-tsl51](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51) |
| **License** | **CC BY-NC-SA 4.0** — non-commercial use only |
| **Classes** | 51 word signs (v1 excludes `null_act`) |
| **Input** | 162-D Holistic landmarks × 60 frames (shoulder-anchored) |
| **Model** | Transformer encoder — trains in Colab |

> **Non-commercial disclaimer:** The TSL-51 dataset and models trained from it may **not** be used for commercial purposes without permission from the dataset maintainers. See the notebook and `python-legacy/webcam_word_demo.py --help` for license details.

### Quick-start — word track (4 steps)

#### Step 1 — Install optional HF deps
```
pip install datasets huggingface_hub
```
(or use the commented block at the bottom of `python-legacy/requirements.txt`)

#### Step 2 — Train in Google Colab
Use `python-legacy/notebooks/train_all_datasets.ipynb`, keep `RUN_TSL51=True`, and run all cells.
TSL-51 artifacts will be written to Google Drive:
   - `tsl51_model.keras`
   - `tsl51_model.tflite`
   - `tsl51_labels.json`
   - `tsl51_scaler.pkl`
   - `tsl51_model_manifest.json` (optional metadata)

#### Step 3 — Thai font
Same as fingerspelling Step 3 — place `NotoSansThai-Regular.ttf` in `assets/`.

#### Step 4 — Run the word-sign demo
```
cd python-legacy
python webcam_word_demo.py
```
Perform a full word sign in front of the camera; the demo buffers 60 frames, then predicts when hand motion exceeds the threshold. Press **`q`** to quit.

#### CLI flags (word demo)
| Flag | Default | What it does |
|------|---------|--------------|
| `--model PATH` | `tsl51_model.keras` | Path to Keras model |
| `--labels PATH` | `tsl51_labels.json` | Class-index → Thai word map |
| `--scaler PATH` | `tsl51_scaler.pkl` | Fitted `StandardScaler` (per-frame 162-D) |
| `--cam N` | `0` | Webcam index |
| `--seq-len N` | `60` | Sequence length (must match training) |
| `--use-tflite` | auto | Force TFLite backend |
| `--threshold F` | `0.55` | Minimum smoothed confidence to show a word |
| `--smoothing-alpha F` | `0.4` | EMA weight for probability smoothing |
| `--motion-min F` | `0.008` | Min mean hand displacement to trigger prediction |
| `--save-log PATH` | off | CSV log: `timestamp_iso,label,confidence` |

---

## How it works

```
Image/frame
  └─ MediaPipe Hands
       └─ 21 landmarks × (x,y,z) = 63 floats
            └─ wrist-relative + scale-normalise  (python-legacy/src/keypoints.py)
                 └─ StandardScaler (scaler.pkl)
                      └─ Dense DNN (model.keras)
                           └─ softmax → Thai letter (labels.json)
```

**Why Dense, not LSTM?**  
Fingerspelling is a *static* hand shape — there is no motion to model.
LSTM is for time-series. Using a Dense classifier on a single frame is
both simpler and more accurate here.

---

## Webcam tips

- Plain background + good lighting → best results.
- Hold one hand clearly in frame.
- Default confidence threshold is 70% (raise/lower with `--threshold`).
- If predictions feel jumpy, try `--smoothing-alpha 0.25`; if they feel laggy, try `0.6`.
- `model.tflite` is preferred automatically over `model.keras` when both are present
  (forces with `--use-tflite`). On CPU it is typically 2–5× faster.

---

## Expanding this project (future phases)
- **More consonants / vowels:** collect more One-Stage-TFS classes or record your own images.
- **Motion words (true LSTM):** needs a video dataset of Thai word signs — the architecture changes significantly (MediaPipe Holistic → sequence of frames → LSTM/Transformer).
- **MediaPipe Tasks API migration:** `mp.solutions.hands` is deprecated; migrate to `mediapipe.tasks` when ready.

---

## Dataset citation
> Kaewwit, Chatkamol; Boonnak, Banjerd; Leelasantitham, Adisorn (2022),  
> "One-Stage-TFS Thai One-Stage Fingerspelling Dataset",  
> Mendeley Data, V1, https://doi.org/10.17632/rknd3wbz42.1  
> Licence: CC BY 4.0
