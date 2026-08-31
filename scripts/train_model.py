"""Train a baseline, frequency-only, or dual-stream detector.

Run against a real manifest CSV built from scripts/standardize_sources.py +
scripts/download_datasets.py + scripts/generate_stylegan3.py output.

Alongside the checkpoint this writes `val_scores.npz` (per-sample validation
scores) and `run.json` (config plus the training-set class prior). Those two
files are what let scripts/analyze_confounds.py decompose the false-positive
difference without retraining anything.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import class_counts, read_manifest_csv
from dsfnet.data.transforms import build_train_transform, build_eval_transform
from dsfnet.inference import predict_scores, save_predictions
from dsfnet.models.baseline import SpatialOnlyClassifier
from dsfnet.models.dual_stream import DualStreamNet
from dsfnet.models.frequency_only import FrequencyOnlyClassifier
from dsfnet.train import train_one_epoch, evaluate_loss_accuracy, logit_adjustment_vector

_MODELS = {
    "spatial_only": SpatialOnlyClassifier,
    "dual_stream": DualStreamNet,
    "frequency_only": FrequencyOnlyClassifier,
}
# Only the frequency-bearing models accept the spectral configuration.
_FREQUENCY_MODELS = {"dual_stream", "frequency_only"}


def build_model(args) -> torch.nn.Module:
    if args.model in _FREQUENCY_MODELS:
        return _MODELS[args.model](
            variant=args.freq_variant,
            n_bins=args.n_bins,
            transform=args.freq_transform,
            window=not args.no_window,
            drop_dc=not args.keep_dc,
        )
    return _MODELS[args.model]()


def stratified_split(entries, val_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits by index, stratified on (label, source) where possible.

    The previous implementation called random_split twice on two Dataset objects
    and relied on identical seeds yielding identical permutations. That happened
    to work but was unstratified, so a rare generator could land entirely in one
    side of the split.
    """
    indices = np.arange(len(entries))
    labels = [e.label for e in entries]
    for strata in ([f"{e.label}:{e.source}" for e in entries], labels, None):
        try:
            return train_test_split(
                indices, test_size=val_split, random_state=seed, stratify=strata
            )
        except ValueError:
            continue  # a stratum with a single member; fall back to coarser strata
    raise RuntimeError("unreachable: stratify=None cannot fail")


def save_checkpoint(path: Path, model, optimizer: torch.optim.Optimizer, epoch: int, history: list) -> None:
    """Saves everything needed to resume training after epoch `epoch` completes."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, list]:
    """Restores model/optimizer state in place; returns (last_completed_epoch, history)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["history"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline, frequency-only or dual-stream detector.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV from dsfnet.data.manifest.write_manifest_csv")
    parser.add_argument("--model", choices=list(_MODELS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for parallel image loading.")

    spectral = parser.add_argument_group("frequency stream")
    spectral.add_argument("--freq-variant", choices=("radial", "scalar"), default="radial",
                          help="'radial' is the redesigned profile; 'scalar' is the v1 ablation arm.")
    spectral.add_argument("--n-bins", type=int, default=16, help="Radial frequency bins.")
    spectral.add_argument("--freq-transform", choices=("fft", "dct"), default="fft")
    spectral.add_argument("--no-window", action="store_true", help="Disable the Hann window (ablation).")
    spectral.add_argument("--keep-dc", action="store_true", help="Keep the DC term (ablation).")

    control = parser.add_argument_group("prior control")
    control.add_argument("--logit-adjust", action="store_true",
                         help="Add log(class prior) to logits during training, so models trained on "
                              "differently balanced sets stay comparable at a fixed threshold.")
    control.add_argument("--logit-adjust-tau", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    entries = read_manifest_csv(args.manifest)
    counts = class_counts(entries)
    print(f"manifest: real={counts['real']} fake={counts['fake']} fake_prior={counts['fake_prior']:.4f}", flush=True)

    full_train = DeepfakeImageDataset(entries, transform=build_train_transform())
    full_eval = DeepfakeImageDataset(entries, transform=build_eval_transform(), return_source=True)

    train_idx, val_idx = stratified_split(entries, args.val_split, args.seed)
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_eval, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    adjustment = None
    if args.logit_adjust:
        adjustment = logit_adjustment_vector(counts["fake_prior"], tau=args.logit_adjust_tau, device=device)
        print(f"logit adjustment: {adjustment.tolist()}", flush=True)

    checkpoint_path = args.out_dir / "checkpoint.pt"
    start_epoch = 1
    history: list = []
    if checkpoint_path.exists():
        last_completed_epoch, history = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch = last_completed_epoch + 1
        print(f"Resuming from checkpoint: epoch {start_epoch}/{args.epochs} (found {checkpoint_path})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device, logit_adjustment=adjustment)
        val_result = evaluate_loss_accuracy(model, _labels_only(val_loader), device)
        print(f"epoch {epoch}: train_loss={train_result.loss:.4f} val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_result.loss, "val_loss": val_result.loss, "val_accuracy": val_result.accuracy})
        save_checkpoint(checkpoint_path, model, optimizer, epoch, history)

    torch.save(model.state_dict(), args.out_dir / f"{args.model}.pth")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))

    val_predictions = predict_scores(model, val_loader, device)
    save_predictions(
        args.out_dir / "val_scores.npz",
        val_predictions,
        train_fake_prior=counts["fake_prior"],
        model=args.model,
    )

    (args.out_dir / "run.json").write_text(json.dumps({
        "model": args.model,
        "manifest": str(args.manifest),
        "class_counts": counts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_split": args.val_split,
        "logit_adjust": args.logit_adjust,
        "freq_variant": args.freq_variant if args.model in _FREQUENCY_MODELS else None,
        "n_bins": args.n_bins if args.model in _FREQUENCY_MODELS else None,
        "freq_transform": args.freq_transform if args.model in _FREQUENCY_MODELS else None,
    }, indent=2))
    print(f"wrote {args.out_dir/'val_scores.npz'} and {args.out_dir/'run.json'}")


def _labels_only(loader: DataLoader):
    """evaluate_loss_accuracy expects (x, y); the eval dataset also yields sources."""
    for batch in loader:
        yield (batch[0], batch[1])


if __name__ == "__main__":
    main()
