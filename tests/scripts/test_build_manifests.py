from pathlib import Path
import importlib.util
import sys

from PIL import Image

from dsfnet.data.manifest import read_manifest_csv

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_manifests.py"
_spec = importlib.util.spec_from_file_location("build_manifests", _SCRIPT_PATH)
build_manifests = importlib.util.module_from_spec(_spec)
sys.modules["build_manifests"] = build_manifests
_spec.loader.exec_module(build_manifests)


def _make_images(folder: Path, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (8, 8)).save(folder / f"{i}.jpg")


def test_builds_manifests_with_folder_labels_and_subsampling(tmp_path, monkeypatch):
    root = tmp_path / "real-vs-fake"
    _make_images(root / "train" / "real", 4)
    _make_images(root / "train" / "fake", 4)
    _make_images(root / "test" / "real", 2)
    _make_images(root / "test" / "fake", 2)
    _make_images(tmp_path / "sg3", 3)
    _make_images(tmp_path / "sg3_test", 1)
    out = tmp_path / "manifests"

    monkeypatch.setattr(sys, "argv", [
        "build_manifests.py",
        "--dataset-root", str(root),
        "--stylegan3-dir", str(tmp_path / "sg3"),
        "--stylegan3-test-dir", str(tmp_path / "sg3_test"),
        "--out-dir", str(out),
        "--max-train-per-class", "3",
    ])
    build_manifests.main()

    original = read_manifest_csv(out / "original_manifest.csv")
    augmented = read_manifest_csv(out / "augmented_manifest.csv")
    test = read_manifest_csv(out / "test_manifest.csv")

    assert sorted(e.label for e in original) == [0, 0, 0, 1, 1, 1]
    assert all(e.label == 0 for e in original if "/real/" in e.path)
    assert len(augmented) == 9 and sum(e.source == "StyleGAN3" for e in augmented) == 3
    assert len(test) == 5 and [e.source for e in test].count("StyleGAN3") == 1
