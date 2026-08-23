"""Run inside Colab: loads a baseline checkpoint and a candidate (dual-stream)
checkpoint, evaluates both on the fixed test manifest, and reports whether
the candidate's FPR improves on the baseline (Figure 8 / Algorithm 4)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import read_manifest_csv
from dsfnet.data.transforms import build_eval_transform
from dsfnet.models.baseline import SpatialOnlyClassifier
from dsfnet.models.dual_stream import DualStreamNet
from dsfnet.evaluate import compute_metrics, compare_fpr

_MODELS = {
    "spatial_only": SpatialOnlyClassifier,
    "dual_stream": DualStreamNet,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate model FPR on the fixed test set.")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-model", choices=list(_MODELS), required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-model", choices=list(_MODELS), required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


@torch.no_grad()
def _predict(model, loader, device):
    y_true, y_pred, y_score = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_score.extend(probs.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = read_manifest_csv(args.test_manifest)
    dataset = DeepfakeImageDataset(entries, transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    baseline = _MODELS[args.baseline_model]().to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
    baseline.eval()

    candidate = _MODELS[args.candidate_model]().to(device)
    candidate.load_state_dict(torch.load(args.candidate_checkpoint, map_location=device))
    candidate.eval()

    baseline_metrics = compute_metrics(*_predict(baseline, loader, device))
    candidate_metrics = compute_metrics(*_predict(candidate, loader, device))
    verdict = compare_fpr(baseline_metrics, candidate_metrics)

    report = {
        "baseline": vars(baseline_metrics),
        "candidate": vars(candidate_metrics),
        "verdict": verdict,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
