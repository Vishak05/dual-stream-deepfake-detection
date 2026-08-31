import numpy as np
import pytest

from dsfnet.evaluate import (
    compare_at_matched_recall,
    compare_fpr,
    compute_metrics,
    compute_metrics_by_source,
    expected_calibration_error,
    fit_temperature,
    metrics_at_threshold,
    prior_correct_scores,
    recall_at_fpr,
    temperature_scale,
    threshold_for_target_fpr,
    threshold_for_target_recall,
)


# --- confusion-matrix metrics ----------------------------------------------

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


# --- operating points, prior correction and calibration ---------------------

def _separable_scores(seed: int = 0, overlap: float = 0.3):
    rng = np.random.RandomState(seed)
    y_true = np.array([0] * 200 + [1] * 200)
    negatives = rng.uniform(0.0, 0.5 + overlap, 200)
    positives = rng.uniform(0.5 - overlap, 1.0, 200)
    return y_true, np.clip(np.concatenate([negatives, positives]), 0, 1)


def test_threshold_for_target_fpr_respects_the_budget():
    y_true, y_score = _separable_scores()
    for target in (0.01, 0.05, 0.10):
        threshold = threshold_for_target_fpr(y_true, y_score, target)
        assert metrics_at_threshold(y_true, y_score, threshold).fpr <= target + 1e-9


def test_recall_at_fpr_matches_metrics_at_that_threshold():
    y_true, y_score = _separable_scores()
    threshold = threshold_for_target_fpr(y_true, y_score, 0.05)
    assert recall_at_fpr(y_true, y_score, 0.05) == pytest.approx(
        metrics_at_threshold(y_true, y_score, threshold).recall
    )


def test_threshold_for_target_recall_reaches_it():
    y_true, y_score = _separable_scores()
    threshold = threshold_for_target_recall(y_true, y_score, 0.80)
    assert metrics_at_threshold(y_true, y_score, threshold).recall >= 0.80


def test_prior_correction_is_identity_when_priors_match():
    _, y_score = _separable_scores()
    corrected = prior_correct_scores(y_score, 0.5, 0.5)
    assert np.allclose(corrected, y_score, atol=1e-6)


def test_prior_correction_applies_the_expected_logit_shift():
    """The base paper's augmentation moved the fake prior 1/3 -> 1/2, which alone
    shifts the boundary by log(0.5/0.5) - log((1/3)/(2/3)) = +ln(2) ~= 0.693."""
    score = np.array([0.5])
    corrected = prior_correct_scores(score, 1 / 3, 1 / 2)
    expected_logit = 0.0 + np.log(2.0)
    assert np.log(corrected / (1 - corrected))[0] == pytest.approx(expected_logit, abs=1e-6)


def test_prior_correction_toward_fewer_fakes_lowers_scores():
    _, y_score = _separable_scores()
    corrected = prior_correct_scores(y_score, 0.5, 1 / 3)
    assert (corrected < y_score).all()


def test_prior_correction_rejects_degenerate_priors():
    with pytest.raises(ValueError, match="train_fake_prior"):
        prior_correct_scores(np.array([0.5]), 0.0, 0.5)
    with pytest.raises(ValueError, match="target_fake_prior"):
        prior_correct_scores(np.array([0.5]), 0.5, 1.0)


def test_temperature_scaling_recovers_a_known_temperature():
    rng = np.random.RandomState(0)
    logits = rng.normal(0, 2.0, 4000)
    y_true = (rng.uniform(size=4000) < 1 / (1 + np.exp(-logits))).astype(int)
    # deliberately overconfident scores: the true logits, doubled. Recovering the
    # calibrated model therefore means dividing by 2, i.e. a temperature of 2.
    overconfident = 1 / (1 + np.exp(-(logits * 2.0)))
    assert fit_temperature(y_true, overconfident) == pytest.approx(2.0, rel=0.15)


def test_temperature_scaling_preserves_ranking():
    _, y_score = _separable_scores()
    scaled = temperature_scale(y_score, 2.0)
    assert np.array_equal(np.argsort(scaled), np.argsort(y_score))


def test_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        temperature_scale(np.array([0.5]), 0.0)


def test_expected_calibration_error_is_zero_for_a_perfect_model():
    y_true = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y_true, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_expected_calibration_error_is_large_for_a_confidently_wrong_model():
    y_true = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y_true, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_matched_recall_comparison_ignores_pure_threshold_advantage():
    """A model whose scores are a monotone rescaling of another's has the same
    ROC, so at matched recall it cannot be an improvement -- even though at
    argmax it would look different."""
    y_true, y_score = _separable_scores()
    shifted = prior_correct_scores(y_score, 1 / 3, 1 / 2)  # same ranking, moved boundary

    result = compare_at_matched_recall(y_true, y_score, shifted)
    assert result["candidate_recall"] >= result["baseline_recall"]
    assert result["verdict"] == "Flagged"
    assert result["candidate_fpr"] == pytest.approx(result["baseline_fpr"], abs=1e-9)


def test_metrics_by_source_reports_error_rate_per_generator():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    sources = ["FFHQ", "FFHQ", "StyleGAN2", "StyleGAN2", "StarGAN", "StarGAN"]

    by_source = compute_metrics_by_source(y_true, y_pred, sources)
    lookup = {m.source: m for m in by_source}

    assert lookup["FFHQ"].error_rate == pytest.approx(0.5)      # 1 false positive of 2
    assert lookup["StyleGAN2"].error_rate == pytest.approx(0.0)  # both caught
    assert lookup["StarGAN"].error_rate == pytest.approx(1.0)    # both missed
    # sorted worst-first, so the generator that generalizes least is on top
    assert by_source[0].source == "StarGAN"
