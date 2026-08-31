"""Figures for the report: ROC overlay, FPR-vs-recall trade-off, reliability
diagram, risk-coverage, and the real-vs-fake radial spectrum.

The spectrum plot is the visual evidence for the project's core hypothesis: if
GAN upsampling leaves periodic traces, the mean radial log-power profile of fake
images should separate from that of real ones at the higher-frequency bins. If
the two curves lie on top of each other, the frequency stream has nothing to
learn -- worth knowing before interpreting any downstream result.

Requires matplotlib:  pip install -e .[viz]
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from dsfnet.evaluate import expected_calibration_error
from dsfnet.inference import load_predictions


def parse_scores(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, _, path = value.partition("=")
    return name, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot comparison figures from saved scores.")
    parser.add_argument("--scores", type=parse_scores, action="append", default=[],
                        help="NAME=PATH.npz from scripts/evaluate_models.py. Repeatable.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--spectra-manifest", type=Path, default=None,
                        help="Manifest to draw the real-vs-fake radial spectrum from.")
    parser.add_argument("--spectra-samples", type=int, default=200)
    parser.add_argument("--n-bins", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install -e .[viz]") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded = {name: load_predictions(path)[0] for name, path in args.scores}

    if loaded:
        _plot_roc(plt, loaded, args.target_fpr, args.out_dir / "roc.png")
        _plot_tradeoff(plt, loaded, args.out_dir / "fpr_vs_recall.png")
        _plot_reliability(plt, loaded, args.out_dir / "reliability.png")
        _plot_risk_coverage(plt, loaded, args.out_dir / "risk_coverage.png")

    if args.spectra_manifest:
        _plot_spectra(plt, args, args.out_dir / "radial_spectrum.png")

    print(f"figures -> {args.out_dir}")


def _plot_roc(plt, loaded, target_fpr, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in loaded.items():
        fpr, tpr, _ = roc_curve(p.y_true, p.y_score)
        ax.plot(fpr, tpr, label=name)
    ax.axvline(target_fpr, color="grey", linestyle="--", linewidth=1,
               label=f"FPR budget = {target_fpr:g}")
    ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_tradeoff(plt, loaded, out_path):
    """The figure the base paper's argument needs: FPR against recall, so a
    'trade-off' can be read as movement along one curve versus a better curve."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in loaded.items():
        fpr, tpr, _ = roc_curve(p.y_true, p.y_score)
        ax.plot(tpr, fpr, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("False positive rate")
    ax.set_title("False positives as a function of recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_reliability(plt, loaded, out_path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for name, p in loaded.items():
        observed = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (p.y_score > lo) & (p.y_score <= hi)
            observed.append(p.y_true[mask].mean() if mask.any() else np.nan)
        ece = expected_calibration_error(p.y_true, p.y_score)
        ax.plot(centres, observed, marker="o", markersize=3, label=f"{name} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgrey", linewidth=1)
    ax.set_xlabel("Predicted P(fake)")
    ax.set_ylabel("Observed fraction fake")
    ax.set_title("Reliability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_risk_coverage(plt, loaded, out_path):
    """Selective prediction: error rate if the least-confident samples are
    deferred to a human reviewer."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, p in loaded.items():
        confidence = np.maximum(p.y_score, 1.0 - p.y_score)
        order = np.argsort(-confidence)
        errors = (p.y_pred() != p.y_true)[order]
        coverage = np.arange(1, len(errors) + 1) / len(errors)
        risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
        ax.plot(coverage, risk, label=name)
    ax.set_xlabel("Coverage (fraction auto-decided)")
    ax.set_ylabel("Error rate on covered samples")
    ax.set_title("Risk-coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_spectra(plt, args, out_path):
    import torch

    from dsfnet.data.manifest import read_manifest_csv
    from dsfnet.data.transforms import build_eval_transform
    from dsfnet.models.spectral import patch_radial_power_spectrum
    from PIL import Image

    entries = read_manifest_csv(args.spectra_manifest)
    transform = build_eval_transform()
    profiles: dict[int, list[np.ndarray]] = {0: [], 1: []}

    for label in (0, 1):
        subset = [e for e in entries if e.label == label][: args.spectra_samples]
        for entry in subset:
            with Image.open(entry.path) as image:
                x = transform(image.convert("RGB")).unsqueeze(0)
            spectral = patch_radial_power_spectrum(x, n_bins=args.n_bins)
            profiles[label].append(spectral.mean(dim=(0, 2, 3)).numpy())

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, name, colour in ((0, "real", "#1e8449"), (1, "fake", "#be185d")):
        if not profiles[label]:
            continue
        stacked = np.stack(profiles[label])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        bins = np.arange(len(mean))
        ax.plot(bins, mean, label=f"{name} (n={len(stacked)})", color=colour)
        ax.fill_between(bins, mean - std, mean + std, alpha=0.15, color=colour)
    ax.set_xlabel("Radial frequency bin (low to high)")
    ax.set_ylabel("Mean log-power")
    ax.set_title("Radial power spectrum, real vs fake")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
