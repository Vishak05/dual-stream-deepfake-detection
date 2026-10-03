"""Run inside Colab against a real manifest CSV built from
scripts/download_datasets.py + scripts/generate_stylegan3.py output."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import read_manifest_csv
from dsfnet.data.transforms import build_train_transform, build_eval_transform, build_robust_train_transform
from dsfnet.models import MODELS as _MODELS
from dsfnet.train import train_one_epoch, evaluate_loss_accuracy



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
    parser = argparse.ArgumentParser(description="Train the baseline or dual-stream Deepfake detector.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV from dsfnet.data.manifest.write_manifest_csv")
    parser.add_argument("--model", choices=list(_MODELS), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robust-aug", action="store_true", help="Train with the label-symmetric robustness augmentation (blur, JPEG, scaling, rotation on every image).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for parallel image loading.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    entries = read_manifest_csv(args.manifest)
    train_transform = build_robust_train_transform() if args.robust_aug else build_train_transform()
    full_train = DeepfakeImageDataset(entries, transform=train_transform)
    full_eval = DeepfakeImageDataset(entries, transform=build_eval_transform())

    n_val = int(len(entries) * args.val_split)
    n_train = len(entries) - n_val
    train_subset, _ = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    _, val_subset = random_split(full_eval, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = _MODELS[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = args.out_dir / "checkpoint.pt"
    start_epoch = 1
    history: list = []
    if checkpoint_path.exists():
        last_completed_epoch, history = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch = last_completed_epoch + 1
        print(f"Resuming from checkpoint: epoch {start_epoch}/{args.epochs} (found {checkpoint_path})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, device)
        val_result = evaluate_loss_accuracy(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={train_result.loss:.4f} val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_result.loss, "val_loss": val_result.loss, "val_accuracy": val_result.accuracy})
        save_checkpoint(checkpoint_path, model, optimizer, epoch, history)

    torch.save(model.state_dict(), args.out_dir / f"{args.model}.pth")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
