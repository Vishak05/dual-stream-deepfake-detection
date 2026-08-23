from pathlib import Path
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_stylegan3.py"
_spec = importlib.util.spec_from_file_location("generate_stylegan3", _SCRIPT_PATH)
generate_stylegan3 = importlib.util.module_from_spec(_spec)
sys.modules["generate_stylegan3"] = generate_stylegan3
_spec.loader.exec_module(generate_stylegan3)


def test_arg_parser_defaults_and_required_fields(tmp_path):
    parser = generate_stylegan3.build_arg_parser()
    args = parser.parse_args([
        "--stylegan3-repo", str(tmp_path / "stylegan3"),
        "--checkpoint", str(tmp_path / "stylegan3-r-ffhq-1024x1024.pkl"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert args.num_images == 5000
    assert args.resolution == 256
    assert args.seed_start == 0


def test_arg_parser_rejects_invalid_resolution(tmp_path):
    parser = generate_stylegan3.build_arg_parser()
    try:
        parser.parse_args([
            "--stylegan3-repo", str(tmp_path),
            "--checkpoint", str(tmp_path / "ckpt.pkl"),
            "--out-dir", str(tmp_path / "out"),
            "--resolution", "512",
        ])
        assert False, "expected SystemExit from argparse choices validation"
    except SystemExit:
        pass
