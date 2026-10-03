from pathlib import Path
import csv
import importlib.util
import sys

from PIL import Image

from dsfnet.data.manifest import ManifestEntry, read_manifest_csv, write_manifest_csv

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_perturbed_set.py"
_spec = importlib.util.spec_from_file_location("build_perturbed_set", _SCRIPT_PATH)
build_perturbed_set = importlib.util.module_from_spec(_spec)
sys.modules["build_perturbed_set"] = build_perturbed_set
_spec.loader.exec_module(build_perturbed_set)


def _manifest(tmp_path: Path, n: int) -> Path:
    entries = []
    for i in range(n):
        path = tmp_path / f"src_{i}.jpg"
        Image.new("RGB", (32, 32), (i * 20, 100, 50)).save(path)
        entries.append(ManifestEntry(path=str(path), label=i % 2, source="FFHQ" if i % 2 == 0 else "StyleGAN"))
    manifest = tmp_path / "in.csv"
    write_manifest_csv(entries, manifest)
    return manifest


def test_test_mode_keeps_labels_and_writes_ops(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, 6)
    monkeypatch.setattr(sys, "argv", [
        "build_perturbed_set.py", "--manifest", str(manifest), "--out-dir", str(tmp_path / "out"),
        "--out-manifest", str(tmp_path / "perturbed.csv"), "--workers", "1",
    ])
    build_perturbed_set.main()

    original = read_manifest_csv(manifest)
    perturbed = read_manifest_csv(tmp_path / "perturbed.csv")
    assert [(e.label, e.source) for e in perturbed] == [(e.label, e.source) for e in original]
    assert all(Path(e.path).exists() for e in perturbed)
    with open(tmp_path / "perturbed_ops.csv") as f:
        ops = list(csv.DictReader(f))
    assert [r["path"] for r in ops] == [e.path for e in perturbed]
    assert all(":" in r["op"] for r in ops)
