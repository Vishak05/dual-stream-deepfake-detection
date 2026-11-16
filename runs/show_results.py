"""Prints the headline results from evaluation_perturbed.json as a terminal table."""
import json
from pathlib import Path

runs = json.loads((Path(__file__).resolve().parent / "evaluation_perturbed.json").read_text())["runs"]
rows = [("paper_repro", "Base paper's method"), ("baseline_robust", "+ symmetric augmentation"),
        ("dsfnet_v2_robust", "Proposed system (DSF-Net v2)")]
print("\nPerturbed test set (base paper's protocol), 23,000 images\n")
print(f"{'Model':<30}{'Accuracy':>10}{'AUC':>8}{'FPR':>8}{'Detection':>11}{'StyleGAN2':>11}")
print("-" * 78)
for key, name in rows:
    r = runs[key]; m = r["metrics"]; c = r["calibrated"]
    print(f"{name:<30}{m['accuracy']:>10.1%}{m['auc']:>8.3f}{m['fpr']:>8.1%}"
          f"{c['metrics']['recall']:>11.1%}{c['per_source']['StyleGAN2']['predicted_fake_rate']:>11.1%}")
print()
