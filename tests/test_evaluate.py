import numpy as np
from dsfnet.evaluate import compute_metrics, compare_fpr


def test_compute_metrics_matches_hand_computed_confusion_matrix():
    # 2 true negatives, 1 false positive, 1 false negative, 2 true positives
    y_true  = np.array([0, 0, 0, 1, 1, 1])
    y_pred  = np.array([0, 0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.4, 0.7, 0.9])

    m = compute_metrics(y_true, y_pred, y_score)

    assert (m.tn, m.fp, m.fn, m.tp) == (2, 1, 1, 2)
    assert m.accuracy == 4 / 6
    assert m.fpr == 1 / 3          # FP / (FP + TN) = 1 / (1 + 2)
    assert m.recall == 2 / 3       # TP / (TP + FN) = 2 / (2 + 1)


def test_compare_fpr_improved_when_candidate_lower():
    baseline = compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.6, 0.4, 0.9])
    )
    candidate = compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.4, 0.9])
    )
    assert compare_fpr(baseline, candidate) == "Improved"


def test_compare_fpr_flagged_when_candidate_not_lower():
    same = compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.6, 0.4, 0.9])
    )
    assert compare_fpr(same, same) == "Flagged"
