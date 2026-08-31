# dsfnet — Dual-Stream Spatial-Frequency Deepfake Detector

A dual-stream deepfake detection model that fuses a ResNet-18 spatial stream
with a frequency-domain (FFT/DCT) stream, plus a controlled re-analysis of the
base paper's reported accuracy/false-positive trade-off (Figures 1-9,
Algorithms 1-5 — see the accompanying report for the full writeup).

Two contributions:

**1. Confound diagnosis.** The base paper reports that StyleGAN3 augmentation
raises accuracy (68.89% → 82.98%) and recall (58.80% → 81.12%) but also false
positives (117 → 247 of 2,000 reals, i.e. FPR 5.85% → 12.35%), and attributes
the cost to the realism of the synthetic images. That comparison is confounded:

- the training fake prior moved 0.333 → 0.500, which shifts the argmax boundary
  ~0.69 nats toward "fake" for Bayesian reasons alone;
- the test set is 71% fake, so the augmented model's prior is *closer* to the
  test prior and part of the accuracy gain is prior matching;
- perturbations were applied to added fakes but not to reals, while every test
  image is perturbed, making "degraded ⇒ fake" learnable;
- AUC rose 8 points (threshold-free) while FPR doubled (at argmax) — the
  signature of decision-boundary placement, not a representational trade-off.

`scripts/analyze_confounds.py` measures how much of the false-positive rise each
of these explains, mostly without retraining.

**2. Spectral representation redesign.**

- **Spatial stream** — ResNet-18, ImageNet-pretrained, unmodified from the base
  detection paper.
- **Frequency stream** — a patch-wise transform that mean-centres and
  Hann-windows each patch, takes its 2D FFT (or DCT), drops the DC term, and
  bins the log-power spectrum into `n_bins` *radial* channels. Preserving the
  spectrum's shape is the point: the earlier version averaged over every
  frequency bin, reducing each patch to a scalar texture-energy value that
  cannot represent the periodic upsampling peaks the hypothesis is about. That
  version is retained as `--freq-variant scalar` for the ablation.
- **Fusion classifier** — concatenates both embeddings (768-d) and classifies
  real vs. fake, with optional logit adjustment for the training class prior.

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

### 5. Standardize every source to one resolution (do this first)

```bash
python scripts/standardize_sources.py \
  --source FFHQ=data/raw/ffhq --source CelebA=data/raw/celeba \
  --source StyleGAN2=data/raw/stylegan2 --source StyleGAN3=data/stylegan3 \
  --out-dir data/standardized --size 256
```

**Do not skip this.** FFHQ is 1024², CelebA aligned crops are ≈178×218, and
StyleGAN3 is generated at 256²/1024². Resizing straight to the 224×224 network
input applies a *different resampling factor per source*, and resampling
rewrites high-frequency content — so resize factor would correlate with label
and the frequency stream could learn the resize instead of the generator. All
later steps read from `data/standardized/`. A `data_card.json` records the
original resolutions so the control is auditable.

### 6. Build the manifest CSVs and the fixed test set

See [Building manifests](#building-manifests) below. The test set is
materialized once with its perturbations baked in, so every model is scored on
byte-identical images:

```bash
python scripts/build_test_set.py \
  --real-source FFHQ=data/standardized/FFHQ \
  --fake-source StyleGAN2=data/standardized/StyleGAN2 \
  --per-source 1000 \
  --out-dir data/test_perturbed --out-manifest data/test_manifest.csv
```

### 7. Train the run matrix

| Run | Model | Training set | Isolates |
|---|---|---|---|
| `base` | `spatial_only` | original | paper baseline |
| `sg3aug` | `spatial_only` | augmented | reproduces the trade-off |
| `prior` | `spatial_only` | prior-balanced | realism, prior held fixed |
| `symm` | `spatial_only` | symmetric | the degraded⇒fake shortcut |
| `dual` | `dual_stream` | augmented | the proposed model |
| `scalarabl` | `dual_stream --freq-variant scalar` | augmented | was the redesign needed? |
| `freqonly` | `frequency_only` | augmented | does the spectrum alone beat chance? |

```bash
python scripts/train_model.py --manifest data/original_manifest.csv  --model spatial_only --out-dir runs/base
python scripts/train_model.py --manifest data/augmented_manifest.csv --model spatial_only --out-dir runs/sg3aug
python scripts/train_model.py --manifest data/prior_manifest.csv     --model spatial_only --out-dir runs/prior
python scripts/train_model.py --manifest data/symmetric_manifest.csv --model spatial_only --out-dir runs/symm
python scripts/train_model.py --manifest data/augmented_manifest.csv --model dual_stream  --out-dir runs/dual
python scripts/train_model.py --manifest data/augmented_manifest.csv --model frequency_only --out-dir runs/freqonly
```

`sg3aug` is the run the base paper omits — without it, comparing `base` against
`dual` confounds the architecture change with the dataset change.

Spectral options: `--freq-variant {radial,scalar}`, `--n-bins`,
`--freq-transform {fft,dct}`, `--no-window`, `--keep-dc`.
Prior control: `--logit-adjust` trains against `log(class prior)` so models
trained on differently balanced sets stay comparable at a fixed threshold.

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

### 8. Evaluate and compare

```bash
python scripts/evaluate_models.py \
  --test-manifest data/test_manifest.csv \
  --model base=spatial_only=runs/base/spatial_only.pth \
  --model sg3aug=spatial_only=runs/sg3aug/spatial_only.pth \
  --model prior=spatial_only=runs/prior/spatial_only.pth \
  --model symm=spatial_only=runs/symm/spatial_only.pth \
  --model dual=dual_stream=runs/dual/dual_stream.pth \
  --baseline sg3aug --candidate dual \
  --target-fpr 0.05 \
  --scores-dir runs/scores --out-report runs/comparison.json
```

Models are given as `NAME=TYPE=CHECKPOINT` (`=` rather than `:`, since Windows
paths contain a drive-letter colon), and any number can be compared.

**Output:** `runs/comparison.json` with, per model, metrics at argmax *and* at
a 5% false-positive budget, `recall_at_target_fpr`, calibration error, and a
per-generator breakdown. The `comparison` block reports the verdict at argmax
alongside **FPR at matched recall** — the criterion that threshold placement
alone cannot win. Per-model test scores land in `runs/scores/`.

### 9. Decompose the false-positive rise

```bash
python scripts/analyze_confounds.py \
  --scores base=runs/scores/base.npz --scores sg3aug=runs/scores/sg3aug.npz \
  --scores prior=runs/scores/prior.npz --scores symm=runs/scores/symm.npz \
  --out-report runs/confound_decomposition.json
```

Attributes the extra false positives to prior shift, threshold placement, the
augmentation shortcut, and residual realism. The prior and threshold components
need no retraining — they re-score the saved vectors.

### 10. Figures

```bash
pip install -e .[viz]
python scripts/plot_results.py --scores dual=runs/scores/dual.npz \
  --spectra-manifest data/test_manifest.csv --out-dir runs/figures
```

ROC overlay, FPR-vs-recall curve, reliability diagram, risk-coverage curve, and
the real-vs-fake radial spectrum. That last one is the falsification check: if
the two curves lie on top of each other, the frequency stream has nothing to
learn and no downstream result should be read as if it did.

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

Training needs four manifests -- the base paper's two plus the two controls --
and `test_manifest.csv` for evaluation. Build them from the **standardized**
folders (step 5) with the functions in `dsfnet.data.manifest`:

```python
from pathlib import Path
from dsfnet.data.manifest import (
    build_original_manifest, build_stylegan3_augmented_manifest,
    build_prior_balanced_manifest, build_symmetric_augmented_manifest,
    class_counts, write_manifest_csv,
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

# The base paper's set. Note this also moves the class prior, which is the confound.
augmented = build_stylegan3_augmented_manifest(original, Path("data/standardized/StyleGAN3"))
write_manifest_csv(augmented, Path("data/augmented_manifest.csv"))

# Control 1 -- replaces half the original fakes with StyleGAN3 instead of adding
# them, so realism changes while the real:fake ratio stays put. This is the
# controlled ablation the base paper claims but does not run.
prior_balanced = build_prior_balanced_manifest(original, Path("data/standardized/StyleGAN3"))
write_manifest_csv(prior_balanced, Path("data/prior_manifest.csv"))

# Control 2 -- adds perturbed reals to match the perturbed fakes, removing the
# "degraded => fake" shortcut. Generate them first:
#   python scripts/perturb_images.py --in-dir data/standardized/FFHQ \
#       --out-dir data/perturbed_real --limit 1000
symmetric = build_symmetric_augmented_manifest(
    original, Path("data/standardized/StyleGAN3"), Path("data/perturbed_real"),
)
write_manifest_csv(symmetric, Path("data/symmetric_manifest.csv"))

# Sanity-check what you built -- class_counts reports the fake prior directly.
for name, entries in [("original", original), ("augmented", augmented),
                      ("prior-balanced", prior_balanced), ("symmetric", symmetric)]:
    print(name, class_counts(entries))
```

`test_manifest.csv` comes from `scripts/build_test_set.py` (step 6), not from
this snippet — it has to be materialized with its perturbations baked in.

---

## Repository layout

```
src/dsfnet/                       importable, unit-tested package
  data/
    transforms.py                 augmentation (symmetric), normalization, denormalization
    manifest.py                   dataset manifests + the prior-balanced/symmetric controls
    perturbations.py              test-time perturbations (paper Sec. III-A-3)
    dataset.py                    PyTorch Dataset, optionally yielding source labels
  models/
    spectral.py                   Algorithm 1: scalar (v1) and radial (v2) spectra, FFT/DCT
    frequency_stream.py           Algorithm 2: spectral CNN embedding
    spatial_stream.py             ResNet-18 backbone (unmodified)
    dual_stream.py                fusion + classification
    frequency_only.py             frequency stream + linear head (hypothesis check)
    baseline.py                   spatial stream + linear head
  train.py                        training/eval loop (Algorithm 3) + logit adjustment
  evaluate.py                     metrics, operating points, prior correction, calibration
  inference.py                    scoring and per-sample score persistence
  explain.py                      Grad-CAM + spectral visualization
scripts/
  standardize_sources.py          Phase 0: one resolution for every source
  download_datasets.py            Kaggle dataset download
  generate_stylegan3.py           StyleGAN3 synthetic image generation
  perturb_images.py               perturbed copies (for the symmetric control)
  build_test_set.py               materialize the fixed perturbed test set
  train_model.py                  train any of the three model types
  evaluate_models.py              compare N models at a matched operating point (Algorithm 4)
  analyze_confounds.py            attribute the false-positive rise (Algorithm 5)
  plot_results.py                 ROC, trade-off, reliability, risk-coverage, spectra
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
