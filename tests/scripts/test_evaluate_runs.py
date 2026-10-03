from pathlib import Path
import argparse
import importlib.util
import sys

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_runs.py"
_spec = importlib.util.spec_from_file_location("evaluate_runs", _SCRIPT_PATH)
evaluate_runs = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_runs"] = evaluate_runs
_spec.loader.exec_module(evaluate_runs)


def test_parse_run_splits_name_model_and_checkpoint():
    name, model, checkpoint = evaluate_runs.parse_run("dual_aug=dual_stream:runs/dual_aug/dual_stream.pth")
    assert (name, model, checkpoint) == ("dual_aug", "dual_stream", Path("runs/dual_aug/dual_stream.pth"))


def test_parse_run_rejects_unknown_model():
    with pytest.raises(argparse.ArgumentTypeError):
        evaluate_runs.parse_run("x=resnet50:x.pth")


def test_per_source_fake_rate():
    sources = np.array(["FFHQ", "FFHQ", "StyleGAN3", "StyleGAN3"])
    y_pred = np.array([0, 1, 1, 1])
    rates = evaluate_runs.per_source_fake_rate(sources, y_pred)
    assert rates == {
        "FFHQ": {"n": 2, "predicted_fake_rate": 0.5},
        "StyleGAN3": {"n": 2, "predicted_fake_rate": 1.0},
    }


def test_threshold_for_fpr_hits_target_on_calibration_reals():
    real_scores = np.linspace(0, 1, 1000)
    threshold = evaluate_runs.threshold_for_fpr(real_scores, 0.05)
    assert (real_scores >= threshold).mean() <= 0.05
    assert (real_scores >= threshold - 0.002).mean() > 0.04


def test_per_group_rates_splits_fpr_and_detection():
    groups = np.array(["jpeg", "jpeg", "jpeg", "blur"])
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0])
    rates = evaluate_runs.per_group_rates(groups, y_true, y_pred)
    assert rates["jpeg"] == {"n": 3, "accuracy": 2 / 3, "fpr": 0.5, "detection_rate": 1.0}
    assert rates["blur"] == {"n": 1, "accuracy": 0.0, "fpr": None, "detection_rate": 0.0}


def test_load_ops_strips_parameters(tmp_path):
    path = tmp_path / "ops.csv"
    path.write_text("path,op\na.jpg,rotation:135\nb.jpg,jpeg:57\n")
    assert evaluate_runs.load_ops(path) == {"a.jpg": "rotation", "b.jpg": "jpeg"}


def test_predict_returns_unsaturated_logit_margins():
    import torch

    class FixedLogits(torch.nn.Module):
        def forward(self, x):
            return x  # the "images" here are already (real, fake) logits

    logits = torch.tensor([[0.0, 2.0], [3.0, 1.0], [0.0, 120.0], [0.0, 130.0]])
    loader = [(logits, torch.zeros(4))]
    margins = evaluate_runs.predict(FixedLogits(), loader, torch.device("cpu"))
    assert margins.tolist() == [2.0, -2.0, 120.0, 130.0]
    # softmax would round both confident images to exactly 1.0; margins keep them apart
    assert margins[3] > margins[2]
