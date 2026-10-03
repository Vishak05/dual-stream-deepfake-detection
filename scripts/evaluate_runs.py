"""Evaluates any number of trained runs on one test manifest and writes a single
JSON report: overall metrics per run, the fraction of each source predicted
fake (FPR for real sources, detection rate for fake sources), and an FPR
verdict for every run against a chosen reference run (Algorithm 4).

With --calibration-manifest, each run's decision threshold is also set from its
scores on that manifest's real images so their FPR equals --target-fpr, and
every metric is reported at that threshold too. --ops-csv adds a breakdown by
perturbation (from build_perturbed_set.py); --save-scores keeps per-image scores."""
import csv
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
    parser.add_argument("--calibration-manifest", type=Path, default=None, help="Manifest whose real images set each run's threshold.")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--ops-csv", type=Path, default=None, help="path,op CSV from build_perturbed_set.py for a per-perturbation breakdown.")
    parser.add_argument("--save-scores", type=Path, default=None, help="Write per-image scores, labels and sources to this .npz.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def per_source_fake_rate(sources: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    return {
        s: {"n": int((sources == s).sum()), "predicted_fake_rate": float(y_pred[sources == s].mean())}
        for s in sorted(set(sources.tolist()))
    }


def per_group_rates(groups: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict]:
    """Accuracy, FPR (real images) and detection rate (fake images) per group."""
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        real, fake = m & (y_true == 0), m & (y_true == 1)
        out[g] = {
            "n": int(m.sum()),
            "accuracy": float((y_pred[m] == y_true[m]).mean()),
            "fpr": float(y_pred[real].mean()) if real.any() else None,
            "detection_rate": float(y_pred[fake].mean()) if fake.any() else None,
        }
    return out


def threshold_for_fpr(real_scores: np.ndarray, target_fpr: float) -> float:
    """Smallest score threshold whose FPR on these real images is <= target_fpr
    (an image is called fake when its score is >= the threshold)."""
    return float(np.quantile(real_scores, 1 - target_fpr, method="higher")) + 1e-9


def load_ops(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row["path"]: row["op"].split(":")[0] for row in csv.DictReader(f)}


def _jsonable(metrics) -> dict:
    return {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in vars(metrics).items()}


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Returns each image's softmax probability of being fake."""
    y_score = []
    for x, _ in loader:
        logits = model(x.to(device)).float()
        y_score.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(y_score).numpy()


def main() -> None:
    args = build_arg_parser().parse_args()
    names = [name for name, _, _ in args.run]
    if args.reference not in names:
        raise SystemExit(f"--reference {args.reference!r} is not one of the --run names {names}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def loader_for(entries):
        return DataLoader(
            DeepfakeImageDataset(entries, transform=build_eval_transform()),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    entries = read_manifest_csv(args.test_manifest)
    y_true = np.array([e.label for e in entries])
    sources = np.array([e.source for e in entries])
    loader = loader_for(entries)
    ops = None
    if args.ops_csv is not None:
        op_of = load_ops(args.ops_csv)
        ops = np.array([op_of[e.path] for e in entries])
    calibration_loader = None
    if args.calibration_manifest is not None:
        calibration_loader = loader_for([e for e in read_manifest_csv(args.calibration_manifest) if e.label == 0])

    def evaluate_at(y_score: np.ndarray, threshold: float) -> tuple:
        y_pred = (y_score >= threshold).astype(int)
        m = compute_metrics(y_true, y_pred, y_score)
        out = {"threshold": threshold, "metrics": _jsonable(m), "per_source": per_source_fake_rate(sources, y_pred)}
        if ops is not None:
            out["per_op"] = per_group_rates(ops, y_true, y_pred)
        return m, out

    results, metrics, scores = {}, {}, {}
    for name, model_name, checkpoint in args.run:
        model = _MODELS[model_name]().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        scores[name] = predict(model, loader, device)
        metrics[name], at_half = evaluate_at(scores[name], 0.5)
        results[name] = {"model": model_name, "checkpoint": str(checkpoint), **at_half}
        line = f"{name}: acc={metrics[name].accuracy:.4f} auc={metrics[name].auc:.4f} fpr={metrics[name].fpr:.4f}"
        if calibration_loader is not None:
            real_scores = predict(model, calibration_loader, device)
            threshold = threshold_for_fpr(real_scores, args.target_fpr)
            calibrated_metrics, calibrated = evaluate_at(scores[name], threshold)
            calibrated["target_fpr"] = args.target_fpr
            calibrated["calibration_fpr"] = float((real_scores >= threshold).mean())
            results[name]["calibrated"] = calibrated
            line += f" | calibrated: fpr={calibrated_metrics.fpr:.4f} recall={calibrated_metrics.recall:.4f}"
        print(line, flush=True)

    for name in names:
        if name != args.reference:
            results[name]["fpr_verdict_vs_reference"] = compare_fpr(metrics[args.reference], metrics[name])

    if args.save_scores is not None:
        extra = {"op": ops} if ops is not None else {}
        np.savez(args.save_scores, y=y_true, source=sources, **extra, **{f"score_{n}": s for n, s in scores.items()})

    report = {"test_manifest": str(args.test_manifest), "reference": args.reference, "runs": results}
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
