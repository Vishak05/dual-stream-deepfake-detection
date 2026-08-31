"""Arg-parser and behaviour checks for the Phase 0/2 data-pipeline scripts."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest
from PIL import Image


def _load(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


standardize_sources = _load("standardize_sources")
build_test_set = _load("build_test_set")
perturb_images = _load("perturb_images")
analyze_confounds = _load("analyze_confounds")


def _folder_of_images(root: Path, name: str, count: int, size: int = 48) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(abs(hash(name)) % 2 ** 31)
    for i in range(count):
        arr = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(folder / f"{name}_{i:03d}.png")
    return folder


# --- standardize_sources ---------------------------------------------------

def test_source_spec_parses_name_and_path():
    name, path = standardize_sources.parse_source("FFHQ=data/raw/ffhq")
    assert name == "FFHQ" and str(path).endswith("ffhq")


def test_source_spec_rejects_missing_separator():
    with pytest.raises(Exception, match="NAME=PATH"):
        standardize_sources.parse_source("FFHQ")


def test_standardization_gives_every_source_the_same_resolution(tmp_path, monkeypatch):
    """The whole point of Phase 0: differing source resolutions would otherwise
    make resize factor correlate with label."""
    ffhq = _folder_of_images(tmp_path, "FFHQ", 2, size=64)
    celeba = _folder_of_images(tmp_path, "CelebA", 2, size=32)
    out_dir = tmp_path / "standardized"

    monkeypatch.setattr(sys, "argv", [
        "standardize_sources.py",
        "--source", f"FFHQ={ffhq}",
        "--source", f"CelebA={celeba}",
        "--out-dir", str(out_dir),
        "--size", "40",
    ])
    standardize_sources.main()

    written = sorted(out_dir.rglob("*.png"))
    assert len(written) == 4
    assert {Image.open(p).size for p in written} == {(40, 40)}

    import json
    card = json.loads((out_dir / "data_card.json").read_text())
    assert card["size"] == 40
    assert card["sources"]["FFHQ"]["original_resolutions"] == {"64x64": 2}
    assert card["sources"]["CelebA"]["original_resolutions"] == {"32x32": 2}


# --- build_test_set --------------------------------------------------------

def test_test_set_is_byte_identical_across_runs(tmp_path, monkeypatch):
    """Every model must see the same test images, so two builds with the same
    seed have to produce identical bytes."""
    reals = _folder_of_images(tmp_path, "FFHQ", 3)
    fakes = _folder_of_images(tmp_path, "StyleGAN3", 3)

    def run(out_name):
        monkeypatch.setattr(sys, "argv", [
            "build_test_set.py",
            "--real-source", f"FFHQ={reals}",
            "--fake-source", f"StyleGAN3={fakes}",
            "--out-dir", str(tmp_path / out_name),
            "--out-manifest", str(tmp_path / f"{out_name}.csv"),
            "--per-source", "3", "--seed", "7",
        ])
        build_test_set.main()

    run("a")
    run("b")

    for left in sorted((tmp_path / "a").rglob("*.png")):
        right = tmp_path / "b" / left.relative_to(tmp_path / "a")
        assert left.read_bytes() == right.read_bytes()


def test_test_set_manifest_labels_reals_and_fakes(tmp_path, monkeypatch):
    from dsfnet.data.manifest import class_counts, read_manifest_csv

    reals = _folder_of_images(tmp_path, "FFHQ", 2)
    fakes = _folder_of_images(tmp_path, "StarGAN", 3)
    manifest = tmp_path / "test.csv"

    monkeypatch.setattr(sys, "argv", [
        "build_test_set.py",
        "--real-source", f"FFHQ={reals}",
        "--fake-source", f"StarGAN={fakes}",
        "--out-dir", str(tmp_path / "out"),
        "--out-manifest", str(manifest),
        "--per-source", "5",
    ])
    build_test_set.main()

    counts = class_counts(read_manifest_csv(manifest))
    assert counts["real"] == 2 and counts["fake"] == 3


def test_build_test_set_requires_at_least_one_source(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "build_test_set.py",
        "--out-dir", str(tmp_path / "out"),
        "--out-manifest", str(tmp_path / "t.csv"),
    ])
    with pytest.raises(SystemExit, match="at least one"):
        build_test_set.main()


# --- perturb_images --------------------------------------------------------

def test_perturb_images_honours_the_limit(tmp_path, monkeypatch):
    source = _folder_of_images(tmp_path, "FFHQ", 5)
    out_dir = tmp_path / "perturbed"

    monkeypatch.setattr(sys, "argv", [
        "perturb_images.py",
        "--in-dir", str(source), "--out-dir", str(out_dir), "--limit", "2",
    ])
    perturb_images.main()
    assert len(list(out_dir.glob("*.png"))) == 2


# --- analyze_confounds -----------------------------------------------------

def test_confound_decomposition_attributes_a_pure_prior_shift(tmp_path, monkeypatch):
    """If the 'augmented' model is the baseline with only its prior moved, the
    prior component must explain the entire false-positive rise."""
    from dsfnet.evaluate import prior_correct_scores
    from dsfnet.inference import PredictionResult, save_predictions

    rng = np.random.RandomState(0)
    y_true = np.array([0] * 200 + [1] * 200)
    scores = np.clip(np.concatenate([rng.uniform(0, 0.8, 200), rng.uniform(0.2, 1.0, 200)]), 0, 1)
    shifted = prior_correct_scores(scores, 1 / 3, 1 / 2)

    save_predictions(tmp_path / "base.npz", PredictionResult(y_true, scores), train_fake_prior=1 / 3)
    save_predictions(tmp_path / "aug.npz", PredictionResult(y_true, shifted), train_fake_prior=1 / 2)

    report_path = tmp_path / "decomposition.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_confounds.py",
        "--scores", f"base={tmp_path / 'base.npz'}",
        "--scores", f"aug={tmp_path / 'aug.npz'}",
        "--baseline-name", "base", "--augmented-name", "aug",
        "--out-report", str(report_path),
    ])
    analyze_confounds.main()

    import json
    report = json.loads(report_path.read_text())
    assert report["total_fp_increase"] > 0
    # undoing the prior shift recovers the baseline's false-positive count exactly
    assert report["components"]["prior"]["fp_after"] == report["baseline"]["fp"]
    assert report["components"]["prior"]["fp_explained"] == report["total_fp_increase"]
    # the two models share a ROC, so the controls are reported as missing, not faked
    assert "status" in report["components"]["realism_prior_held"]
