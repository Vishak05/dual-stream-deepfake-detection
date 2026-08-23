# dsfnet — Dual-Stream Spatial-Frequency Deepfake Detector

Implements the design in `../../phase2/Phase2_Report_v2.md` (Figures 1-9, Algorithms 1-4):
a ResNet-18 spatial stream (unmodified from the base paper) fused with a new
FFT/DCT-based frequency stream, aimed at reducing the false-positive rate
increase the base paper reports after StyleGAN3 data augmentation.

## Local development (no GPU, no dataset required)

```bash
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest -v
```

All tests run on CPU in under a minute using tiny synthetic fixtures — no
real dataset or GPU is needed to develop or verify this package.

## Real training (Google Colab)

This machine has no GPU and insufficient free disk space for the dataset, so
real dataset acquisition, StyleGAN3 generation, and model training all happen
in `notebooks/colab_train.ipynb`, never locally:

1. Push this repo to GitHub.
2. Open `notebooks/colab_train.ipynb` in Google Colab (Runtime > GPU).
3. Follow the cells: clone repo, `pip install -e .[colab]`, mount Drive,
   upload your `kaggle.json` API token, download datasets, generate StyleGAN3
   images, train both models, run the FPR comparison report.

### StyleGAN3 checkpoints

Download `stylegan3-r-ffhq-1024x1024.pkl` or `stylegan3-t-ffhq-1024x1024.pkl`
from the NVlabs/stylegan3 releases and store them in your Google Drive so
they persist across Colab sessions.

### Building manifests

Once real image folders exist (from `scripts/download_datasets.py` and
`scripts/generate_stylegan3.py`), build the three CSV manifests
`train_model.py`/`evaluate_models.py` expect, using the functions in
`dsfnet.data.manifest`:

```python
from pathlib import Path
from dsfnet.data.manifest import (
    build_original_manifest, build_stylegan3_augmented_manifest, write_manifest_csv,
)

original = build_original_manifest(
    real_sources={"FFHQ": Path("/content/data/ffhq"), "CelebA": Path("/content/data/celeba")},
    fake_sources={
        "StarGAN": Path("/content/data/fake/stargan"),
        "GDWCT": Path("/content/data/fake/gdwct"),
        "AttGAN": Path("/content/data/fake/attgan"),
        "StyleGAN": Path("/content/data/fake/stylegan"),
        "StyleGAN2": Path("/content/data/fake/stylegan2"),
    },
)
write_manifest_csv(original, Path("/content/data/original_manifest.csv"))

augmented = build_stylegan3_augmented_manifest(original, Path("/content/data/stylegan3"))
write_manifest_csv(augmented, Path("/content/data/augmented_manifest.csv"))

# test_manifest.csv: build similarly from a held-out test image folder,
# scanned per source with dsfnet.data.manifest.scan_source_folder
```

## Repository layout

```
src/dsfnet/          importable, unit-tested package
  data/              transforms, dataset manifest, PyTorch Dataset
  models/            spatial stream, frequency stream, dual-stream fusion, baseline
  train.py           training/eval loop (Algorithm 3)
  evaluate.py        FPR-focused metrics + comparison (Algorithm 4)
  explain.py         Grad-CAM + spectral visualization
scripts/             CLI entrypoints, meant to run in Colab
notebooks/           Colab driver notebook
tests/               pytest suite (CPU-only, no real data)
```

## Team workflow

Each teammate clones the repo, runs `pytest` locally before pushing, and CI
(`.github/workflows/tests.yml`) re-runs the suite on every push/PR. Real
training runs happen from anyone's Colab session pointed at a shared Google
Drive folder for datasets/checkpoints — nothing large needs to be committed.
