"""Evaluates any number of trained runs on one test manifest and writes a single
JSON report: overall metrics per run, the fraction of each source predicted
fake (FPR for real sources, detection rate for fake sources), and an FPR
verdict for every run against a chosen reference run (Algorithm 4)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import read_manifest_csv
from dsfnet.data.transforms import build_eval_transform
from dsfnet.evaluate import compute_metrics, compare_fpr
from dsfnet.models import MODELS as _MODELS



def parse_run(spec: str) -> tuple[str, str, Path]:
    """Parses NAME=MODEL:CHECKPOINT, e.g. baseline_orig=spatial_only:runs/a/spatial_only.pth."""
    name, rest = spec.split("=", 1)
    model, checkpoint = rest.split(":", 1)
    if model not in _MODELS:
        raise argparse.ArgumentTypeError(f"unknown model {model!r}; choose from {list(_MODELS)}")
    return name, model, Path(checkpoint)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate several trained runs on one test manifest.")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--run", type=parse_run, action="append", required=True, help="NAME=MODEL:CHECKPOINT (repeatable).")
    parser.add_argument("--reference", required=True, help="Run NAME whose FPR the other runs are compared against.")
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def per_source_fake_rate(sources: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    return {
        s: {"n": int((sources == s).sum()), "predicted_fake_rate": float(y_pred[sources == s].mean())}
        for s in sorted(set(sources.tolist()))
    }


@torch.no_grad()
def predict(model, loader, device):
    y_pred, y_score = [], []
    for x, _ in loader:
        logits = model(x.to(device)).float()
        y_score.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        y_pred.append(logits.argmax(dim=1).cpu())
    return torch.cat(y_pred).numpy(), torch.cat(y_score).numpy()


def main() -> None:
    args = build_arg_parser().parse_args()
    names = [name for name, _, _ in args.run]
    if args.reference not in names:
        raise SystemExit(f"--reference {args.reference!r} is not one of the --run names {names}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = read_manifest_csv(args.test_manifest)
    y_true = np.array([e.label for e in entries])
    sources = np.array([e.source for e in entries])
    loader = DataLoader(
        DeepfakeImageDataset(entries, transform=build_eval_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    results, metrics = {}, {}
    for name, model_name, checkpoint in args.run:
        model = _MODELS[model_name]().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        y_pred, y_score = predict(model, loader, device)
        metrics[name] = compute_metrics(y_true, y_pred, y_score)
        results[name] = {
            "model": model_name,
            "checkpoint": str(checkpoint),
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in vars(metrics[name]).items()},
            "per_source": per_source_fake_rate(sources, y_pred),
        }
        print(f"{name}: acc={metrics[name].accuracy:.4f} auc={metrics[name].auc:.4f} fpr={metrics[name].fpr:.4f}", flush=True)

    for name in names:
        if name != args.reference:
            results[name]["fpr_verdict_vs_reference"] = compare_fpr(metrics[args.reference], metrics[name])

    report = {"test_manifest": str(args.test_manifest), "reference": args.reference, "runs": results}
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
