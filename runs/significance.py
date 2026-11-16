"""Paired significance tests on the perturbed test set at each run's calibrated
(5%-FPR) threshold: McNemar on per-image correctness, bootstrap 95% CI on the
AUC difference, and Wilson CIs for per-source detection rates."""
import json, math, sys
import numpy as np
from sklearn.metrics import roc_auc_score

R = "/home/vishak/dsfnet-runs"
tag = sys.argv[1] if len(sys.argv) > 1 else "perturbed"
rep = json.load(open(f"{R}/evaluation_{tag}.json"))["runs"]
z = np.load(f"{R}/scores_{tag}.npz", allow_pickle=True)
y, src = z["y"], z["source"]
names = list(rep)
pred = {n: (z[f"score_{n}"] >= rep[n]["calibrated"]["threshold"]).astype(int) for n in names}

def wilson(k, n, zc=1.96):
    p = k / n; d = 1 + zc * zc / n; c = (p + zc * zc / (2 * n)) / d
    h = zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)

def mcnemar(a, b):
    ca, cb = pred[a] == y, pred[b] == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())   # a right/b wrong, a wrong/b right
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return n01, n10, math.erfc(math.sqrt(stat / 2))           # chi2(1) survival

def auc_diff_ci(a, b, reps=1000, seed=0):
    rng = np.random.default_rng(seed); sa, sb = z[f"score_{a}"], z[f"score_{b}"]; d = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return round(roc_auc_score(y, sa) - roc_auc_score(y, sb), 4), np.round(np.percentile(d, [2.5, 97.5]), 4).tolist()

out = {"per_source_detection_ci": {}, "pairs": {}}
for n in names:
    out["per_source_detection_ci"][n] = {
        s: {"rate": round(100 * pred[n][src == s].mean(), 1), "ci95": wilson(int(pred[n][src == s].sum()), int((src == s).sum()))}
        for s in sorted(set(src.tolist()))}
pairs = [("dsfnet_v2_robust", "paper_repro"), ("baseline_robust", "paper_repro"), ("dsfnet_v2_robust", "baseline_robust"),
         ("dsfnet_v2", "baseline_augmented"), ("paper_repro", "baseline_augmented")]
for a, b in pairs:
    n01, n10, p = mcnemar(a, b); d, ci = auc_diff_ci(a, b)
    out["pairs"][f"{a} vs {b}"] = {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "mcnemar_p": p, "auc_diff": d, "auc_diff_ci95": ci}
    print(f"{a} vs {b}: +{n01}/-{n10} images, McNemar p={p:.2e}, AUC diff {d} CI {ci}")
for n in ["paper_repro", "baseline_robust", "dsfnet_v2_robust"]:
    print(n, {s: v for s, v in out["per_source_detection_ci"][n].items()})
json.dump(out, open(f"{R}/significance_{tag}.json", "w"), indent=2)
