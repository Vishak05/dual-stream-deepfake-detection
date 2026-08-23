from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
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
