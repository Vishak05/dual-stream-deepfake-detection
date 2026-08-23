from pathlib import Path
from dsfnet.data.manifest import (
    ManifestEntry,
    scan_source_folder,
    build_original_manifest,
    build_stylegan3_augmented_manifest,
    write_manifest_csv,
    read_manifest_csv,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n")  # minimal placeholder bytes; content unused by manifest scan


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
