from pathlib import Path
import importlib.util
import sys

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_models.py"
_spec = importlib.util.spec_from_file_location("evaluate_models", _SCRIPT_PATH)
evaluate_models = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_models"] = evaluate_models
_spec.loader.exec_module(evaluate_models)


def test_arg_parser_accepts_any_number_of_models(tmp_path):
    """Algorithm 4 compares three models; the previous CLI took exactly two."""
    parser = evaluate_models.build_arg_parser()
    args = parser.parse_args([
        "--test-manifest", str(tmp_path / "test.csv"),
        "--model", f"base=spatial_only={tmp_path / 'base.pth'}",
        "--model", f"sg3aug=spatial_only={tmp_path / 'sg3aug.pth'}",
        "--model", f"dual=dual_stream={tmp_path / 'dual.pth'}",
        "--baseline", "sg3aug",
        "--candidate", "dual",
        "--out-report", str(tmp_path / "report.json"),
    ])
    assert [name for name, _, _ in args.model] == ["base", "sg3aug", "dual"]
    assert args.baseline == "sg3aug"
    assert args.candidate == "dual"
    assert args.target_fpr == 0.05


def test_model_spec_survives_a_windows_drive_letter():
    """':' would split a path like C:\\runs\\x.pth, so '=' is the separator."""
    name, model_type, checkpoint = evaluate_models.parse_model(
        r"dual=dual_stream=C:\runs\dual\dual_stream.pth"
    )
    assert name == "dual"
    assert model_type == "dual_stream"
    assert str(checkpoint).endswith("dual_stream.pth")


def test_model_spec_rejects_a_missing_field():
    with pytest.raises(Exception, match="NAME=TYPE=CHECKPOINT"):
        evaluate_models.parse_model("dual=dual_stream")


def test_model_spec_rejects_an_unknown_type():
    with pytest.raises(Exception, match="unknown model type"):
        evaluate_models.parse_model("x=not_a_model=ckpt.pth")


def test_frequency_config_is_read_from_the_run_json(tmp_path):
    """A checkpoint trained with a non-default spectral config will not load into
    a default-shaped model, so the architecture is rebuilt from run.json."""
    import json
    import torch
    from dsfnet.models.dual_stream import DualStreamNet

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "model": "dual_stream", "freq_variant": "radial", "n_bins": 8, "freq_transform": "dct",
    }))
    checkpoint = run_dir / "dual_stream.pth"
    torch.save(DualStreamNet(pretrained_spatial=False, variant="radial", n_bins=8,
                             transform="dct").state_dict(), checkpoint)

    model = evaluate_models.load_model("dual_stream", checkpoint, torch.device("cpu"))
    assert model.frequency.n_bins == 8
    assert model.frequency.transform == "dct"
