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
