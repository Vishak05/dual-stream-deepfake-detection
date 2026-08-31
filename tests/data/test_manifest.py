from pathlib import Path

import pytest

from dsfnet.data.manifest import (
    ManifestEntry,
    scan_source_folder,
    build_original_manifest,
    build_prior_balanced_manifest,
    build_stylegan3_augmented_manifest,
    build_symmetric_augmented_manifest,
    class_counts,
    split_leave_one_generator_out,
    write_manifest_csv,
    read_manifest_csv,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n")  # minimal placeholder bytes; content unused by manifest scan


def _folder(root: Path, name: str, count: int) -> Path:
    folder = root / name
    for i in range(count):
        _touch(folder / f"{name}_{i:03d}.png")
    return folder


def test_scan_source_folder_finds_images_by_extension(tmp_path):
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    _touch(tmp_path / "ignore.txt")
    entries = scan_source_folder(tmp_path, label=0, source="FFHQ")
    assert len(entries) == 2
    assert all(e.label == 0 and e.source == "FFHQ" for e in entries)


def test_build_original_manifest_merges_real_and_fake_with_correct_labels(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    _touch(real_dir / "r1.png")
    _touch(fake_dir / "f1.png")
    _touch(fake_dir / "f2.png")

    entries = build_original_manifest(
        real_sources={"FFHQ": real_dir},
        fake_sources={"StarGAN": fake_dir},
    )
    labels = sorted(e.label for e in entries)
    assert labels == [0, 1, 1]


def test_build_stylegan3_augmented_manifest_appends_to_original(tmp_path):
    fake_dir = tmp_path / "fake"
    sg3_dir = tmp_path / "stylegan3"
    _touch(fake_dir / "f1.png")
    _touch(sg3_dir / "s1.png")
    _touch(sg3_dir / "s2.png")

    original = build_original_manifest(real_sources={}, fake_sources={"StarGAN": fake_dir})
    augmented = build_stylegan3_augmented_manifest(original, sg3_dir)

    assert len(augmented) == len(original) + 2
    assert sum(1 for e in augmented if e.source == "StyleGAN3") == 2


def test_manifest_csv_roundtrip(tmp_path):
    entries = [ManifestEntry(path="a.png", label=0, source="FFHQ"),
               ManifestEntry(path="b.png", label=1, source="StarGAN")]
    csv_path = tmp_path / "manifest.csv"
    write_manifest_csv(entries, csv_path)
    loaded = read_manifest_csv(csv_path)
    assert loaded == entries


# --- controlled variants ---------------------------------------------------

def test_class_counts_reports_the_fake_prior(tmp_path):
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 20)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 10)},
    )
    assert class_counts(original) == {"real": 20, "fake": 10, "total": 30, "fake_prior": pytest.approx(1 / 3)}


def test_prior_balanced_manifest_holds_the_class_prior_fixed(tmp_path):
    """The control the base paper claims but does not run: realism changes while
    the real:fake ratio stays exactly as it was."""
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 20)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 10)},
    )
    sg3 = _folder(tmp_path, "StyleGAN3", 10)

    balanced = build_prior_balanced_manifest(original, sg3, replace_fraction=0.5)
    augmented = build_stylegan3_augmented_manifest(original, sg3)

    assert class_counts(balanced)["fake_prior"] == pytest.approx(class_counts(original)["fake_prior"])
    # the paper's own construction moves the prior, which is the confound
    assert class_counts(augmented)["fake_prior"] == pytest.approx(0.5)

    sources = [e.source for e in balanced]
    assert sources.count("StyleGAN3") == 5
    assert sources.count("StyleGAN2") == 5


def test_prior_balanced_manifest_is_deterministic(tmp_path):
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 10)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 10)},
    )
    sg3 = _folder(tmp_path, "StyleGAN3", 10)
    assert build_prior_balanced_manifest(original, sg3, seed=7) == \
           build_prior_balanced_manifest(original, sg3, seed=7)


def test_prior_balanced_manifest_reports_insufficient_stylegan3(tmp_path):
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 4)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 10)},
    )
    with pytest.raises(ValueError, match="only found 2"):
        build_prior_balanced_manifest(original, _folder(tmp_path, "StyleGAN3", 2), replace_fraction=1.0)


def test_symmetric_manifest_adds_perturbed_reals(tmp_path):
    """Removes the 'degraded => fake' shortcut by giving the real class the same
    perturbation treatment the fake class receives."""
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 10)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 5)},
    )
    sg3 = _folder(tmp_path, "StyleGAN3", 5)
    perturbed = _folder(tmp_path, "RealPerturbed", 5)

    asymmetric = build_stylegan3_augmented_manifest(original, sg3)
    symmetric = build_symmetric_augmented_manifest(original, sg3, perturbed)

    assert class_counts(asymmetric)["real"] == 10
    assert class_counts(symmetric)["real"] == 15
    assert class_counts(symmetric)["fake"] == class_counts(asymmetric)["fake"]
    assert all(e.label == 0 for e in symmetric if e.source == "Real-Perturbed")


def test_symmetric_manifest_can_cap_the_number_of_perturbed_reals(tmp_path):
    original = build_original_manifest(
        {"FFHQ": _folder(tmp_path, "FFHQ", 4)},
        {"StyleGAN2": _folder(tmp_path, "StyleGAN2", 4)},
    )
    symmetric = build_symmetric_augmented_manifest(
        original, _folder(tmp_path, "StyleGAN3", 4), _folder(tmp_path, "RealPerturbed", 10),
        n_perturbed_real=3,
    )
    assert sum(1 for e in symmetric if e.source == "Real-Perturbed") == 3


def test_leave_one_generator_out_removes_it_from_training():
    entries = [
        ManifestEntry("a.png", 1, "StarGAN"),
        ManifestEntry("b.png", 1, "AttGAN"),
        ManifestEntry("c.png", 0, "FFHQ"),
    ]
    kept, held_out = split_leave_one_generator_out(entries, "StarGAN")
    assert [e.source for e in held_out] == ["StarGAN"]
    assert "StarGAN" not in {e.source for e in kept}


def test_leave_one_generator_out_reports_available_sources():
    entries = [ManifestEntry("a.png", 1, "StarGAN")]
    with pytest.raises(ValueError, match=r"available: \['StarGAN'\]"):
        split_leave_one_generator_out(entries, "GDWCT")
