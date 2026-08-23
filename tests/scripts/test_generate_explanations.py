from pathlib import Path
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_explanations.py"
_spec = importlib.util.spec_from_file_location("generate_explanations", _SCRIPT_PATH)
generate_explanations = importlib.util.module_from_spec(_spec)
sys.modules["generate_explanations"] = generate_explanations
_spec.loader.exec_module(generate_explanations)


def test_arg_parser_defaults(tmp_path):
    parser = generate_explanations.build_arg_parser()
    args = parser.parse_args([
        "--checkpoint", str(tmp_path / "dual_stream.pth"),
        "--model", "dual_stream",
        "--manifest", str(tmp_path / "test_manifest.csv"),
        "--out-dir", str(tmp_path / "explanations"),
    ])
    assert args.num_samples == 10
    assert args.class_idx == 1
    assert args.model == "dual_stream"


def test_arg_parser_rejects_unknown_model_name(tmp_path):
    parser = generate_explanations.build_arg_parser()
    try:
        parser.parse_args([
            "--checkpoint", str(tmp_path / "ckpt.pth"),
            "--model", "not_a_real_model",
            "--manifest", str(tmp_path / "test_manifest.csv"),
            "--out-dir", str(tmp_path),
        ])
        assert False, "expected SystemExit from argparse choices validation"
    except SystemExit:
        pass
