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

Models (`--model`): `spatial_only` (the base paper's ResNet-18),
`dual_stream` (v1: one mean log-power value per 32×32 patch) and
`dual_stream_v2` (DSF-Net v2: the 32×32 log-power spectrum averaged over
patches, which keeps periodic GAN upsampling peaks).

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

Two flags matter for the 140k real-and-fake-faces dataset:

- `--jpeg-quality 95` saves JPEGs at the same quality as that dataset. Without
  it the images are PNGs, and a model can learn "PNG = fake" from compression
  traces instead of GAN artifacts.
- `--no-custom-ops` uses StyleGAN3's pure-PyTorch ops instead of compiling its
  CUDA plugins. Needed when the CUDA toolkit (`nvcc`) isn't installed, e.g. a
  pip-only PyTorch install under WSL; about 2 images/s on an RTX 4050 at 256 px.

Use a separate `--seed-start` range (e.g. `--seed-start 100000 --num-images
1000`) into another folder for held-out StyleGAN3 test images.

Or source a pre-generated StyleGAN3 face dataset instead of running generation
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

To compare more than two runs at once, use `evaluate_runs.py`. It also
reports, per source, the fraction of images predicted fake — the FPR for real
sources and the detection rate for each fake source (e.g. held-out StyleGAN3):

```bash
python scripts/evaluate_runs.py --test-manifest data/test_manifest.csv \
  --run baseline_original=spatial_only:runs/baseline_original/spatial_only.pth \
  --run baseline_augmented=spatial_only:runs/baseline_augmented/spatial_only.pth \
  --run dual_augmented=dual_stream:runs/dual_augmented/dual_stream.pth \
  --reference baseline_augmented --out-report runs/evaluation_report.json
```

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

## Going from a smoke test to a real run

A reasonable first pass (either method) is to skip StyleGAN3 entirely —
create an empty `data/stylegan3/` (or `$DATA_DIR/stylegan3` in Colab) so
`augmented_manifest.csv` builds successfully as a copy of
`original_manifest.csv` — and run a couple of epochs just to confirm the
whole pipeline works. Once you're ready to add real StyleGAN3 images and do
the actual full run, three things need to happen, **in this order**:

1. **Generate or source real StyleGAN3 images** into that same
   `data/stylegan3/` folder (see [step 4](#4-get-stylegan3-generated-images)
   above) — it's fine that the folder already exists from the smoke test,
   the new images just get added alongside whatever (nothing, if you used
   the empty-placeholder approach) is already there.

2. **Rebuild `augmented_manifest.csv`** by rerunning the manifest-building
   step — only this one needs rebuilding; `original_manifest.csv` and
   `test_manifest.csv` don't reference the StyleGAN3 folder, so they're
   unaffected and don't need to change.

3. **Use a fresh `--out-dir` for the dual-stream model** (e.g.
   `runs/dual_stream_full` instead of reusing `runs/dual_stream`), *or*
   delete `runs/dual_stream/checkpoint.pt` before retraining. This matters
   because of the checkpoint/resume behavior described above: if you reuse
   the same `--out-dir` as the smoke test, `train_model.py` will see an
   existing `checkpoint.pt` and "resume" from the epoch the smoke test left
   off at — except the model at that checkpoint was trained on the *old*
   (StyleGAN3-empty) manifest, not the real one. It'll silently continue
   training on the new data without any warning that the model's early
   epochs came from a different dataset, quietly invalidating the run. A
   fresh output directory avoids this entirely.

Also bump `--epochs` up from your smoke-test value (e.g. `2`) to the real
target — `20`, matching the base paper's setting, is the default if you omit
the flag.

---

## Building manifests

Both methods need three CSV manifests before training: `original_manifest.csv`
(baseline dataset), `augmented_manifest.csv` (baseline + StyleGAN3 images),
and `test_manifest.csv` (held-out evaluation set).

For the 140k real-and-fake-faces layout (`<root>/{train,test}/{real,fake}/`),
`scripts/build_manifests.py` builds all three. Labels come from the folder
names — the Kaggle CSVs use 1 = real, the opposite of dsfnet's 1 = fake:

```bash
python scripts/build_manifests.py \
  --dataset-root data/real_vs_fake/real-vs-fake \
  --stylegan3-dir data/stylegan3 \
  --stylegan3-test-dir data/stylegan3_test \
  --out-dir data
```

`--max-train-per-class N` subsamples the training set for quicker runs.

For other dataset layouts, use the functions in `dsfnet.data.manifest`
directly:

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

## Comparing against the base paper under its protocol

The base paper (Hafezi et al., IEEE Access 2026) tests on images that each get
one random degradation (rotation 45+45U°, ×0.5/×2 scaling, blur, mirroring,
JPEG 50–99, Sec. III-A3). Its "augmented" training set holds 4,000 clean +
1,000 degraded StyleGAN3 images (Fig. 3). To reproduce both and evaluate the
proposed fixes:

```bash
# Degraded StyleGAN3 copies (Sec. III-B) and the paper's training composition
python scripts/build_perturbed_set.py --mode train-degrade \
  --manifest data/sg3_to_degrade.csv --out-dir data/stylegan3_train_degraded --out-manifest data/sg3_degraded.csv
python scripts/build_manifests.py ... --paper-degraded-dir data/stylegan3_train_degraded --out-dir data

# Perturbed test and calibration sets (writes <manifest>_ops.csv too)
python scripts/build_perturbed_set.py --manifest data/test_full_manifest.csv \
  --out-dir data/test_perturbed --out-manifest data/test_perturbed_manifest.csv
python scripts/build_perturbed_set.py --manifest data/valid_manifest.csv --seed 2 \
  --out-dir data/valid_perturbed --out-manifest data/valid_perturbed_manifest.csv

# Base paper's method vs proposed (label-symmetric augmentation + DSF-Net v2)
python scripts/train_model.py --manifest data/paper_augmented_manifest.csv --model spatial_only --out-dir runs/paper_repro
python scripts/train_model.py --manifest data/augmented_manifest.csv --model dual_stream_v2 --robust-aug --out-dir runs/dsfnet_v2_robust

# Evaluate with each model's threshold calibrated to 5% FPR on perturbed real validation images
python scripts/evaluate_runs.py --test-manifest data/test_perturbed_manifest.csv \
  --run paper_repro=spatial_only:runs/paper_repro/spatial_only.pth \
  --run proposed=dual_stream_v2:runs/dsfnet_v2_robust/dual_stream_v2.pth \
  --reference paper_repro --calibration-manifest data/valid_perturbed_manifest.csv --target-fpr 0.05 \
  --ops-csv data/test_perturbed_manifest_ops.csv --save-scores runs/scores.npz --out-report runs/evaluation.json
```

`--robust-aug` applies the degradations to every training image, real or
fake, so degradation can't act as a "fake" signal. Scores are logit margins
(fake minus real logit); threshold 0 equals a 0.5 probability.

## Repository layout

```
src/dsfnet/                       importable, unit-tested package
  data/                           transforms, dataset manifest, PyTorch Dataset, base-paper perturbations
  models/                         spatial stream, frequency stream, dual-stream fusion, baseline
  train.py                        training/eval loop (Algorithm 3)
  evaluate.py                     FPR-focused metrics + comparison (Algorithm 4)
  explain.py                      Grad-CAM + spectral visualization
scripts/
  download_datasets.py            Kaggle dataset download
  generate_stylegan3.py           StyleGAN3 synthetic image generation
  train_model.py                  train baseline or dual-stream model
  build_manifests.py              manifests for the 140k real-and-fake-faces layout
  build_perturbed_set.py          base-paper test perturbations / training degradations
  evaluate_models.py              compare baseline vs. candidate, FPR verdict
  evaluate_runs.py                compare runs: per-source/per-op rates, FPR calibration
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
