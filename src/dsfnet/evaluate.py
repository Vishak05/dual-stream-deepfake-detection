"""Metrics, operating-point selection and calibration.

The base paper reports confusion matrices at the argmax threshold and AUC
separately. That combination hides the finding this project is built around: its
AUC *rose* (76.47% -> 84.38%) while its FPR roughly doubled (5.85% -> 12.35%).
A threshold-free measure improving while a fixed-threshold measure degrades is
the signature of a decision-boundary shift, not a representational trade-off --
so everything here is built to compare models at matched operating points rather
than at argmax.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class SourceMetrics:
    """Per-generator breakdown.

    Deliberately not the full metric set: a single source is almost always all
    real or all fake, so AUC and precision are undefined for it. `error_rate` is
    the false-positive rate for a real source and the miss rate for a fake one --
    the quantity that actually matters per generator.
    """

    source: str
    label: int
    n: int
    correct: int
    accuracy: float
    error_rate: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> ClassificationMetrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        auc=roc_auc_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        fpr=fpr,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def compare_fpr(baseline: ClassificationMetrics, candidate: ClassificationMetrics) -> str:
    """Figure 8's 'Trade-off Decision Logic' box."""
    return "Improved" if candidate.fpr < baseline.fpr else "Flagged"


# --- operating points -------------------------------------------------------

def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> ClassificationMetrics:
    """Metrics with the decision boundary placed at `threshold` instead of argmax."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_score)


def threshold_for_target_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Highest-recall threshold whose FPR stays at or below `target_fpr`.

    Selected on validation and then applied to test, this is the deployment-correct
    way to set a forensic detector: fix the tolerable false-accusation rate first,
    then ask how many fakes are caught.
    """
    fpr, _, thresholds = roc_curve(y_true, y_score)
    allowed = np.flatnonzero(fpr <= target_fpr)
    return float(thresholds[allowed[-1]])


def threshold_for_target_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    """Lowest-FPR threshold that still reaches `target_recall`."""
    _, tpr, thresholds = roc_curve(y_true, y_score)
    reached = np.flatnonzero(tpr >= target_recall)
    if reached.size == 0:
        return float(thresholds[-1])
    return float(thresholds[reached[0]])


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """The headline metric: how many fakes are caught at a fixed false-positive budget."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    allowed = np.flatnonzero(fpr <= target_fpr)
    return float(tpr[allowed[-1]])


def compare_at_matched_recall(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    candidate_score: np.ndarray,
    baseline_threshold: float = 0.5,
) -> dict:
    """Acceptance criterion: is the candidate's FPR lower at the *same* recall?

    Comparing at argmax lets a model look better simply by sitting at a different
    point on its ROC curve. Matching recall first removes that, so a verdict of
    "Improved" cannot be earned by threshold placement alone.
    """
    baseline = metrics_at_threshold(y_true, baseline_score, baseline_threshold)
    matched_threshold = threshold_for_target_recall(y_true, candidate_score, baseline.recall)
    candidate = metrics_at_threshold(y_true, candidate_score, matched_threshold)

    return {
        "baseline_threshold": float(baseline_threshold),
        "baseline_recall": baseline.recall,
        "baseline_fpr": baseline.fpr,
        "candidate_threshold": matched_threshold,
        "candidate_recall": candidate.recall,
        "candidate_fpr": candidate.fpr,
        "verdict": "Improved" if candidate.fpr < baseline.fpr else "Flagged",
    }


# --- prior correction and calibration ---------------------------------------

def prior_correct_scores(
    y_score: np.ndarray,
    train_fake_prior: float,
    target_fake_prior: float,
    eps: float = 1e-7,
) -> np.ndarray:
    """Re-express scores as if the model had been trained under a different prior.

    A classifier's posterior absorbs its training prior. The base paper's
    augmentation moved that prior from 5,000/15,000 = 0.333 fake to
    10,000/20,000 = 0.500, which shifts the decision boundary by
    log(0.5/0.5) - log(0.333/0.667) ~= 0.69 nats toward "fake" on its own --
    raising true *and* false positives with no change in learned features.
    Undoing that shift isolates how much of the reported trade-off was Bayesian
    bookkeeping rather than realism.
    """
    for name, prior in (("train_fake_prior", train_fake_prior), ("target_fake_prior", target_fake_prior)):
        if not 0.0 < prior < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {prior}")

    p = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    shift = _logit(target_fake_prior) - _logit(train_fake_prior)
    return 1.0 / (1.0 + np.exp(-(logit + shift)))


def temperature_scale(y_score: np.ndarray, temperature: float, eps: float = 1e-7) -> np.ndarray:
    """Divides the logit by `temperature` (>1 softens, <1 sharpens)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(logit / temperature)))


def fit_temperature(
    y_true: np.ndarray,
    y_score: np.ndarray,
    lo: float = 0.05,
    hi: float = 10.0,
    iterations: int = 4,
    grid: int = 60,
) -> float:
    """Fits a temperature by minimizing NLL, via successive grid refinement.

    A grid search rather than an optimizer keeps this dependency-free and cannot
    fail to converge; the objective is one-dimensional and smooth.
    """
    y_true = np.asarray(y_true, dtype=float)
    best = 1.0
    for _ in range(iterations):
        candidates = np.linspace(lo, hi, grid)
        losses = [_nll(y_true, temperature_scale(y_score, float(t))) for t in candidates]
        best_idx = int(np.argmin(losses))
        best = float(candidates[best_idx])
        step = (hi - lo) / (grid - 1)
        lo, hi = max(1e-3, best - step), best + step
    return best


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """ECE over confidence of the predicted class."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    predicted = (y_score >= 0.5).astype(int)
    confidence = np.where(predicted == 1, y_score, 1.0 - y_score)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


# --- per-generator breakdown ------------------------------------------------

def compute_metrics_by_source(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: list[str],
) -> list[SourceMetrics]:
    """Accuracy and error rate per generator family, sorted by error rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sources = np.asarray(sources)

    results = []
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        n = int(mask.sum())
        correct = int((y_pred[mask] == y_true[mask]).sum())
        accuracy = correct / n if n else 0.0
        results.append(
            SourceMetrics(
                source=source,
                label=int(y_true[mask][0]) if n else -1,
                n=n,
                correct=correct,
                accuracy=accuracy,
                error_rate=1.0 - accuracy,
            )
        )
    return sorted(results, key=lambda m: m.error_rate, reverse=True)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _nll(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
