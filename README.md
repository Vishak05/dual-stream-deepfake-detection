# dsfnet — Dual-Stream Spatial-Frequency Deepfake Detector

A dual-stream deepfake detection model that fuses a ResNet-18 spatial stream
with a frequency-domain (FFT/DCT) stream, aimed at reducing the false-positive
rate increase reported when training data is augmented with StyleGAN3-generated
synthetic faces (Figures 1-9, Algorithms 1-4 — see `docs/` or the accompanying
report for the full architecture writeup).

- **Spatial stream** — ResNet-18, ImageNet-pretrained, unmodified from the base
  detection paper.
- **Frequency stream** — a new patch-wise 2D FFT/DCT log-power spectrum feature
  extractor + small CNN, designed to catch the periodic spectral artifacts GAN
  upsampling leaves behind, even when spatial artifacts are visually suppressed.
- **Fusion classifier** — concatenates both embeddings (768-d) and classifies
  real vs. fake.

The package (`src/dsfnet/`) is fully unit-tested on CPU with synthetic data —
no GPU or dataset is required to install it or run the test suite. Producing
real results (training on actual face datasets) requires a GPU and a real
dataset, via either of the two methods below.

## Installation

```bash
git clone https://github.com/<your-username>/dual-stream-deepfake-detection.git
cd dual-stream-deepfake-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run the test suite to confirm the install worked:

```bash
pytest -v
```

All tests run on CPU in under a minute using tiny synthetic fixtures.

---

## Method 1 — Run locally

Requires a CUDA-capable GPU for practical training times (CPU works but is
slow — expect several hours per full training run instead of minutes), a
Kaggle account, and several GB of free disk space for the dataset.

### 1. Install with Colab/local extras

```bash
pip install -e .[colab]   # adds the `kaggle` package for dataset download
```

### 2. Get a Kaggle API token

Kaggle's current UI shows your API key as a copyable string rather than
downloading a `kaggle.json` file directly (Kaggle → Settings → API →
"Create New Token" — copy the key shown, and note your username, e.g. from
`kaggle.com/<username>`). Build the file yourself instead:

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username": "your-kaggle-username", "key": "your-copied-api-key"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

If Kaggle's UI does offer a direct `kaggle.json` download for you (this has
varied over time), that works too — just place it at the same path.

### 3. Download the real/baseline-fake datasets

```bash
python scripts/download_datasets.py \
  --kaggle-dataset xhlulu/140k-real-and-fake-faces \
  --out-dir data/
```

Repeat `--kaggle-dataset` for additional sources (FFHQ, CelebA, or individual
GAN-fake datasets) as needed.

### 4. Get StyleGAN3-generated images

Either generate your own (needs a GPU and the NVlabs/stylegan3 repo):

```bash
git clone https://github.com/NVlabs/stylegan3.git
python scripts/generate_stylegan3.py \
  --stylegan3-repo stylegan3 \
  --checkpoint /path/to/stylegan3-r-ffhq-1024x1024.pkl \
  --out-dir data/stylegan3 \
  --num-images 5000
```

or source a pre-generated StyleGAN3 face dataset instead of running generation
yourself, if you want to skip this step.

### 5. Build the manifest CSVs

See [Building manifests](#building-manifests) below — produces
`original_manifest.csv`, `augmented_manifest.csv`, and `test_manifest.csv`.

### 6. Train both models

```bash
python scripts/train_model.py --manifest data/original_manifest.csv  --model spatial_only --out-dir runs/baseline
python scripts/train_model.py --manifest data/augmented_manifest.csv --model dual_stream  --out-dir runs/dual_stream
```

**Output:** each run writes `runs/<name>/<model>.pth` (the final trained
checkpoint) and `runs/<name>/history.json` (per-epoch train/val loss and
accuracy).

**Resuming an interrupted run:** `train_model.py` also writes
`runs/<name>/checkpoint.pt` after *every* epoch (model + optimizer state +
history so far). If training stops partway through (e.g. a disconnected
Colab session), just rerun the exact same command with the same `--out-dir`
— it automatically detects the checkpoint and resumes from the next epoch,
printing `Resuming from checkpoint: epoch N/total`. At most one epoch's
work is ever lost. Use a fresh `--out-dir` (or delete `checkpoint.pt`) if
you want to intentionally restart from scratch instead.

**Data loading speed:** `--num-workers` (default 4) controls how many
parallel processes load/decode images. Match it to your actual CPU core
count — passing more workers than available cores can make loading *slower*
(PyTorch will warn you if this happens); check with `nproc` (Linux/Colab) or
`os.cpu_count()`. On GPU, training also automatically uses mixed-precision
(AMP) for a substantial speedup — no flag needed, it's disabled automatically
on CPU.

### 7. Evaluate and compare

```bash
python scripts/evaluate_models.py \
  --test-manifest data/test_manifest.csv \
  --baseline-checkpoint runs/baseline/spatial_only.pth   --baseline-model spatial_only \
  --candidate-checkpoint runs/dual_stream/dual_stream.pth --candidate-model dual_stream \
  --out-report runs/fpr_comparison_report.json
```

**Output:** `runs/fpr_comparison_report.json`, containing full metrics
(accuracy, AUC, precision, recall, F1, FPR, and the raw confusion-matrix
counts) for both models side by side, plus a `"verdict"` field —
`"Improved"` if the candidate's false-positive rate is lower than the
baseline's, `"Flagged"` otherwise. This file is also printed to stdout.

### 8. (Optional) Generate explanation snapshots

```bash
python scripts/generate_explanations.py \
  --checkpoint runs/dual_stream/dual_stream.pth --model dual_stream \
  --manifest data/test_manifest.csv \
  --out-dir runs/explanations \
  --num-samples 10
```

**Output:** `runs/explanations/gradcam_*.png` (spatial-stream Grad-CAM
heatmaps) and `spectral_*.png` (frequency-stream spectral maps, dual-stream
model only) for the first N test images — useful for report/presentation
figures.

---

## Method 2 — Run in Google Colab

Useful when a local GPU or sufficient local disk space isn't available.

1. Push this repo to GitHub (if you haven't already).
2. Open `notebooks/colab_train.ipynb` in Google Colab.
3. Runtime → Change runtime type → GPU.
4. Run the cells in order:
   - Clone the repo and `pip install -e .[colab]`.
   - Mount Google Drive (for persisting datasets/checkpoints across sessions).
   - Create `kaggle.json` from your Kaggle username + copied API key (see
     [Get a Kaggle API token](#2-get-a-kaggle-api-token) above — Kaggle's UI
     shows a copyable key rather than a file to upload).
   - Download datasets via `scripts/download_datasets.py`.
   - Generate StyleGAN3 images via `scripts/generate_stylegan3.py`.
   - Build manifests (see [Building manifests](#building-manifests)).
   - Train both models via `scripts/train_model.py`.
   - Evaluate and compare via `scripts/evaluate_models.py`.

Outputs are identical to Method 1 — checkpoints, `history.json`, and the FPR
comparison report — just written to Colab's filesystem (point `--out-dir`
paths at a Google Drive folder if you want them to persist after the session
ends).

### StyleGAN3 checkpoints

Download `stylegan3-r-ffhq-1024x1024.pkl` or `stylegan3-t-ffhq-1024x1024.pkl`
from the NVlabs/stylegan3 releases and store them in Google Drive so they
persist across Colab sessions.

---

## Building manifests

Both methods need three CSV manifests before training: `original_manifest.csv`
(baseline dataset), `augmented_manifest.csv` (baseline + StyleGAN3 images),
and `test_manifest.csv` (held-out evaluation set). Build them with the
functions in `dsfnet.data.manifest`:

```python
from pathlib import Path
from dsfnet.data.manifest import (
    build_original_manifest, build_stylegan3_augmented_manifest, write_manifest_csv,
)

original = build_original_manifest(
    real_sources={"FFHQ": Path("data/ffhq"), "CelebA": Path("data/celeba")},
    fake_sources={
        "StarGAN": Path("data/fake/stargan"),
        "GDWCT": Path("data/fake/gdwct"),
        "AttGAN": Path("data/fake/attgan"),
        "StyleGAN": Path("data/fake/stylegan"),
        "StyleGAN2": Path("data/fake/stylegan2"),
    },
)
write_manifest_csv(original, Path("data/original_manifest.csv"))

augmented = build_stylegan3_augmented_manifest(original, Path("data/stylegan3"))
write_manifest_csv(augmented, Path("data/augmented_manifest.csv"))

# test_manifest.csv: build similarly from a held-out test image folder,
# scanned per source with dsfnet.data.manifest.scan_source_folder
```

---

## Repository layout

```
src/dsfnet/                       importable, unit-tested package
  data/                           transforms, dataset manifest, PyTorch Dataset
  models/                         spatial stream, frequency stream, dual-stream fusion, baseline
  train.py                        training/eval loop (Algorithm 3)
  evaluate.py                     FPR-focused metrics + comparison (Algorithm 4)
  explain.py                      Grad-CAM + spectral visualization
scripts/
  download_datasets.py            Kaggle dataset download
  generate_stylegan3.py           StyleGAN3 synthetic image generation
  train_model.py                  train baseline or dual-stream model
  evaluate_models.py              compare baseline vs. candidate, FPR verdict
  generate_explanations.py        Grad-CAM / spectral-map snapshots
notebooks/
  colab_train.ipynb               end-to-end Colab driver notebook
tests/                            pytest suite (CPU-only, no real data)
```

## Contributing

Run `pytest` locally before pushing — CI (`.github/workflows/tests.yml`)
re-runs the full suite on every push/PR. Real training runs (Method 1 or
Method 2) don't need to be reproduced for every code change; only the CPU
test suite needs to stay green.
