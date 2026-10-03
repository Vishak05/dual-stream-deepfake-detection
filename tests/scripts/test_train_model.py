from pathlib import Path
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_model.py"
_spec = importlib.util.spec_from_file_location("train_model", _SCRIPT_PATH)
train_model = importlib.util.module_from_spec(_spec)
sys.modules["train_model"] = train_model
_spec.loader.exec_module(train_model)


def test_arg_parser_defaults_match_base_paper_hyperparameters(tmp_path):
    parser = train_model.build_arg_parser()
    args = parser.parse_args([
        "--manifest", str(tmp_path / "manifest.csv"),
        "--model", "dual_stream",
        "--out-dir", str(tmp_path / "runs" / "exp1"),
    ])
    assert args.epochs == 20
    assert args.batch_size == 64
    assert args.lr == 1e-4
    assert args.val_split == 0.25
    assert args.model == "dual_stream"


def test_arg_parser_rejects_unknown_model_name(tmp_path):
    parser = train_model.build_arg_parser()
    try:
        parser.parse_args([
            "--manifest", str(tmp_path / "manifest.csv"),
            "--model", "not_a_real_model",
            "--out-dir", str(tmp_path),
        ])
        assert False, "expected SystemExit from argparse choices validation"
    except SystemExit:
        pass


def test_arg_parser_accepts_v2_model_and_robust_aug(tmp_path):
    args = train_model.build_arg_parser().parse_args([
        "--manifest", str(tmp_path / "manifest.csv"),
        "--model", "dual_stream_v2",
        "--out-dir", str(tmp_path / "runs" / "v2"),
        "--robust-aug",
    ])
    assert args.model == "dual_stream_v2"
    assert args.robust_aug is True
