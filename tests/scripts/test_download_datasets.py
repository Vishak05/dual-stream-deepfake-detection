from pathlib import Path
import importlib.util
import sys

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "download_datasets.py"
_spec = importlib.util.spec_from_file_location("download_datasets", _SCRIPT_PATH)
download_datasets = importlib.util.module_from_spec(_spec)
sys.modules["download_datasets"] = download_datasets
_spec.loader.exec_module(download_datasets)


def test_arg_parser_accepts_repeated_dataset_flag_and_out_dir(tmp_path):
    parser = download_datasets.build_arg_parser()
    args = parser.parse_args([
        "--kaggle-dataset", "xhlulu/140k-real-and-fake-faces",
        "--kaggle-dataset", "ffhq-face-data-set",
        "--out-dir", str(tmp_path),
    ])
    assert args.kaggle_dataset == ["xhlulu/140k-real-and-fake-faces", "ffhq-face-data-set"]
    assert args.out_dir == tmp_path
