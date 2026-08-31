"""Score any number of models on the fixed test set and compare them at a
matched operating point (Algorithm 4).

The base paper compares at the argmax threshold only, which conflates
representational quality with decision-boundary placement. This reports the
argmax numbers for continuity, plus recall at a fixed false-positive budget and
an FPR-at-matched-recall verdict that threshold placement alone cannot win.

Per-model test scores are written to `--scores-dir` for scripts/analyze_confounds.py.

Example:
    python scripts/evaluate_models.py \
        --test-manifest data/test_manifest.csv \
        --model base=spatial_only=runs/base/spatial_only.pth \
        --model sg3aug=spatial_only=runs/sg3aug/spatial_only.pth \
        --model dual=dual_stream=runs/dual/dual_stream.pth \
        --baseline sg3aug --candidate dual \
        --scores-dir runs/scores --out-report runs/comparison.json
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dsfnet.data.dataset import DeepfakeImageDataset
from dsfnet.data.manifest import class_counts, read_manifest_csv
from dsfnet.data.transforms import build_eval_transform
from dsfnet.evaluate import (
    compare_at_matched_recall,
    compare_fpr,
    compute_metrics,
    compute_metrics_by_source,
    expected_calibration_error,
    metrics_at_threshold,
    recall_at_fpr,
    threshold_for_target_fpr,
)
from dsfnet.inference import predict_scores, save_predictions
from dsfnet.models.baseline import SpatialOnlyClassifier
from dsfnet.models.dual_stream import DualStreamNet
from dsfnet.models.frequency_only import FrequencyOnlyClassifier

_MODELS = {
    "spatial_only": SpatialOnlyClassifier,
    "dual_stream": DualStreamNet,
    "frequency_only": FrequencyOnlyClassifier,
}
_FREQUENCY_MODELS = {"dual_stream", "frequency_only"}


def parse_model(value: str) -> tuple[str, str, Path]:
    # '=' rather than ':' as the separator, since Windows checkpoint paths
    # contain a drive-letter colon.
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected NAME=TYPE=CHECKPOINT, got {value!r}")
    name, model_type, checkpoint = parts
    if model_type not in _MODELS:
        raise argparse.ArgumentTypeError(f"unknown model type {model_type!r}; choose from {sorted(_MODELS)}")
    return name, model_type, Path(checkpoint)


def read_run_config(checkpoint: Path) -> dict:
    """The run.json the training script writes next to each checkpoint."""
    run_json = checkpoint.parent / "run.json"
    return json.loads(run_json.read_text()) if run_json.exists() else {}


def train_fake_prior(run: dict) -> float | None:
    """The class prior the model was trained under.

    This has to travel with the test scores: prior correction in
    scripts/analyze_confounds.py cannot run without knowing which prior each
    model's posterior absorbed, and that is the single largest confound in the
    base paper's comparison.
    """
    counts = run.get("class_counts") or {}
    return counts.get("fake_prior")


def load_model(model_type: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Rebuilds the architecture, reading the spectral config from the run.json
    the training script wrote next to the checkpoint when one is present."""
    kwargs = {}
    if model_type in _FREQUENCY_MODELS:
        run = read_run_config(checkpoint)
        if run:
            kwargs = {
                "variant": run.get("freq_variant") or "radial",
                "n_bins": run.get("n_bins") or 16,
                "transform": run.get("freq_transform") or "fft",
            }
    model = _MODELS[model_type](**kwargs).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare models at a matched operating point.")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True,
                        help="NAME=TYPE=CHECKPOINT. Repeatable.")
    parser.add_argument("--baseline", type=str, default=None, help="Model NAME to treat as the baseline.")
    parser.add_argument("--candidate", type=str, default=None, help="Model NAME to treat as the candidate.")
    parser.add_argument("--target-fpr", type=float, default=0.05,
                        help="False-positive budget for the headline recall@FPR metric.")
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, default=None,
                        help="Directory to write per-model test scores (.npz) for the confound analysis.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = read_manifest_csv(args.test_manifest)
    counts = class_counts(entries)
    dataset = DeepfakeImageDataset(entries, transform=build_eval_transform(), return_source=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    report = {
        "test_manifest": str(args.test_manifest),
        "test_class_counts": counts,
        "target_fpr": args.target_fpr,
        "models": {},
    }
    scores: dict[str, object] = {}

    for name, model_type, checkpoint in args.model:
        model = load_model(model_type, checkpoint, device)
        prediction = predict_scores(model, loader, device)
        scores[name] = prediction
        prior = train_fake_prior(read_run_config(checkpoint))

        argmax_metrics = compute_metrics(prediction.y_true, prediction.y_pred(), prediction.y_score)
        budget_threshold = threshold_for_target_fpr(prediction.y_true, prediction.y_score, args.target_fpr)
        budget_metrics = metrics_at_threshold(prediction.y_true, prediction.y_score, budget_threshold)

        report["models"][name] = {
            "type": model_type,
            "checkpoint": str(checkpoint),
            "train_fake_prior": prior,
            "at_argmax": asdict(argmax_metrics),
            f"at_fpr_{args.target_fpr:g}": {
                "threshold": budget_threshold,
                **asdict(budget_metrics),
            },
            "recall_at_target_fpr": recall_at_fpr(prediction.y_true, prediction.y_score, args.target_fpr),
            "expected_calibration_error": expected_calibration_error(prediction.y_true, prediction.y_score),
            "by_source": [asdict(m) for m in compute_metrics_by_source(
                prediction.y_true, prediction.y_pred(), prediction.sources
            )] if prediction.sources is not None else None,
        }

        if args.scores_dir:
            metadata = {"model": model_type, "checkpoint": str(checkpoint)}
            if prior is not None:
                metadata["train_fake_prior"] = prior
            save_predictions(Path(args.scores_dir) / f"{name}.npz", prediction, **metadata)

        print(f"{name}: acc={argmax_metrics.accuracy:.4f} auc={argmax_metrics.auc:.4f} "
              f"fpr={argmax_metrics.fpr:.4f} recall={argmax_metrics.recall:.4f} "
              f"recall@fpr{args.target_fpr:g}={report['models'][name]['recall_at_target_fpr']:.4f}", flush=True)

    if args.baseline and args.candidate:
        base, cand = scores[args.baseline], scores[args.candidate]
        base_metrics = compute_metrics(base.y_true, base.y_pred(), base.y_score)
        cand_metrics = compute_metrics(cand.y_true, cand.y_pred(), cand.y_score)
        report["comparison"] = {
            "baseline": args.baseline,
            "candidate": args.candidate,
            "at_argmax": compare_fpr(base_metrics, cand_metrics),
            "at_matched_recall": compare_at_matched_recall(base.y_true, base.y_score, cand.y_score),
        }
        matched = report["comparison"]["at_matched_recall"]
        print(f"\n{args.candidate} vs {args.baseline} at matched recall "
              f"({matched['baseline_recall']:.4f}): "
              f"FPR {matched['baseline_fpr']:.4f} -> {matched['candidate_fpr']:.4f} "
              f"=> {matched['verdict']}")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {args.out_report}")


if __name__ == "__main__":
    main()
