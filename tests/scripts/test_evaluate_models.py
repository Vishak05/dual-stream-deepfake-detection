from pathlib import Path
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_models.py"
_spec = importlib.util.spec_from_file_location("evaluate_models", _SCRIPT_PATH)
evaluate_models = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_models"] = evaluate_models
_spec.loader.exec_module(evaluate_models)


def test_arg_parser_accepts_baseline_and_candidate_checkpoints(tmp_path):
    parser = evaluate_models.build_arg_parser()
    args = parser.parse_args([
        "--test-manifest", str(tmp_path / "test.csv"),
        "--baseline-checkpoint", str(tmp_path / "baseline.pth"),
        "--baseline-model", "spatial_only",
        "--candidate-checkpoint", str(tmp_path / "dual.pth"),
        "--candidate-model", "dual_stream",
        "--out-report", str(tmp_path / "report.json"),
    ])
    assert args.baseline_model == "spatial_only"
    assert args.candidate_model == "dual_stream"
