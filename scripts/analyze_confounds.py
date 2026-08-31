"""Algorithm 5 -- decompose the false-positive rise into attributable causes.

The base paper reports that StyleGAN3 augmentation raised false positives from
117 to 247 (of 2,000 real test images) and attributes this to the realism of the
synthetic images. But the augmentation changed three things at once, and the
evaluation fixed a fourth:

  prior     the training fake prior moved 0.333 -> 0.500, shifting the argmax
            boundary ~0.69 nats toward "fake" for Bayesian reasons alone
  shortcut  perturbations were applied to added fakes but not to reals, while
            every test image is perturbed, so "degraded => fake" is learnable
  threshold both models were read at argmax despite an 8-point AUC gain, so the
            augmented model simply sits at a higher-recall operating point
  realism   whatever remains -- the effect the paper claims

Each component is measured against the same augmented model, so they are not
mutually exclusive and need not sum to the total; `residual` is reported as
total minus the sum, and a large negative residual means the components overlap.

Example:
    python scripts/analyze_confounds.py \
        --scores base=runs/scores/base.npz \
        --scores sg3aug=runs/scores/sg3aug.npz \
        --scores prior=runs/scores/prior.npz \
        --scores symm=runs/scores/symm.npz \
        --out-report runs/confound_decomposition.json
"""
import argparse
import json
from pathlib import Path

from dsfnet.evaluate import (
    compute_metrics,
    metrics_at_threshold,
    prior_correct_scores,
    threshold_for_target_recall,
)
from dsfnet.inference import load_predictions


def parse_scores(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    name, _, path = value.partition("=")
    return name, Path(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attribute the false-positive rise to its causes.")
    parser.add_argument("--scores", type=parse_scores, action="append", required=True,
                        help="NAME=PATH.npz written by scripts/evaluate_models.py. Repeatable.")
    parser.add_argument("--baseline-name", default="base", help="Model trained on the Original set.")
    parser.add_argument("--augmented-name", default="sg3aug", help="Model trained on the StyleGAN3-Augmented set.")
    parser.add_argument("--prior-name", default="prior", help="Model trained on the Prior-Balanced set.")
    parser.add_argument("--symmetric-name", default="symm", help="Model trained on the Symmetric-Augmented set.")
    parser.add_argument("--out-report", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    loaded = {name: load_predictions(path) for name, path in args.scores}

    def require(name: str):
        if name not in loaded:
            raise SystemExit(f"missing --scores {name}=...; got {sorted(loaded)}")
        return loaded[name]

    base, base_meta = require(args.baseline_name)
    aug, aug_meta = require(args.augmented_name)

    base_metrics = compute_metrics(base.y_true, base.y_pred(), base.y_score)
    aug_metrics = compute_metrics(aug.y_true, aug.y_pred(), aug.y_score)
    total = aug_metrics.fp - base_metrics.fp

    components: dict[str, dict] = {}

    # --- prior: re-score the augmented model under the baseline's training prior
    base_prior = base_meta.get("train_fake_prior")
    aug_prior = aug_meta.get("train_fake_prior")
    if base_prior is not None and aug_prior is not None:
        corrected = prior_correct_scores(aug.y_score, float(aug_prior), float(base_prior))
        corrected_metrics = metrics_at_threshold(aug.y_true, corrected, 0.5)
        components["prior"] = {
            "description": f"augmented model re-scored at the baseline's fake prior "
                           f"({aug_prior:.4f} -> {base_prior:.4f})",
            "fp_after": corrected_metrics.fp,
            "fp_explained": aug_metrics.fp - corrected_metrics.fp,
            "recall_after": corrected_metrics.recall,
        }

    # --- threshold: hold recall at the baseline's level instead of reading argmax
    matched_threshold = threshold_for_target_recall(aug.y_true, aug.y_score, base_metrics.recall)
    matched = metrics_at_threshold(aug.y_true, aug.y_score, matched_threshold)
    components["threshold"] = {
        "description": f"augmented model read at the baseline's recall ({base_metrics.recall:.4f}) "
                       f"instead of argmax",
        "threshold": matched_threshold,
        "fp_after": matched.fp,
        "fp_explained": aug_metrics.fp - matched.fp,
        "recall_after": matched.recall,
    }

    # --- data / shortcut: separately trained controls
    for key, name, description in (
        ("realism_prior_held", args.prior_name,
         "prior-balanced training set: realism changes, class prior does not"),
        ("shortcut", args.symmetric_name,
         "symmetric augmentation: perturbed reals added to match perturbed fakes"),
    ):
        if name not in loaded:
            components[key] = {"description": description, "status": f"missing --scores {name}=..."}
            continue
        control, _ = loaded[name]
        control_metrics = compute_metrics(control.y_true, control.y_pred(), control.y_score)
        components[key] = {
            "description": description,
            "fp_after": control_metrics.fp,
            "fp_explained": aug_metrics.fp - control_metrics.fp,
            "recall_after": control_metrics.recall,
        }

    explained = sum(c.get("fp_explained", 0) for c in components.values())
    report = {
        "baseline": {"name": args.baseline_name, "fp": base_metrics.fp, "fpr": base_metrics.fpr,
                     "recall": base_metrics.recall, "auc": base_metrics.auc,
                     "accuracy": base_metrics.accuracy},
        "augmented": {"name": args.augmented_name, "fp": aug_metrics.fp, "fpr": aug_metrics.fpr,
                      "recall": aug_metrics.recall, "auc": aug_metrics.auc,
                      "accuracy": aug_metrics.accuracy},
        "total_fp_increase": total,
        "components": components,
        "residual_after_components": total - explained,
        "note": "Components are each measured against the augmented model and overlap; "
                "they are not a partition and need not sum to the total.",
    }

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))

    print(f"false positives: {base_metrics.fp} -> {aug_metrics.fp}  (total rise {total})")
    for key, component in components.items():
        if "fp_explained" in component:
            print(f"  {key:20s} explains {component['fp_explained']:+5d} FP  "
                  f"(recall after {component['recall_after']:.4f})")
        else:
            print(f"  {key:20s} {component['status']}")
    print(f"\nreport -> {args.out_report}")


if __name__ == "__main__":
    main()
