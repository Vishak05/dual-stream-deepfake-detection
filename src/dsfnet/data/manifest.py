import csv
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    label: int          # 0 = real, 1 = fake
    source: str          # e.g. "FFHQ", "StarGAN", "StyleGAN3"


def scan_source_folder(folder: Path, label: int, source: str) -> list[ManifestEntry]:
    folder = Path(folder)
    entries = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in _IMAGE_EXTENSIONS:
            entries.append(ManifestEntry(path=str(p), label=label, source=source))
    return entries


def build_original_manifest(
    real_sources: dict[str, Path],
    fake_sources: dict[str, Path],
) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for name, folder in real_sources.items():
        entries.extend(scan_source_folder(folder, label=0, source=name))
    for name, folder in fake_sources.items():
        entries.extend(scan_source_folder(folder, label=1, source=name))
    return entries


def build_stylegan3_augmented_manifest(
    original: list[ManifestEntry],
    stylegan3_folder: Path,
) -> list[ManifestEntry]:
    augmented = scan_source_folder(stylegan3_folder, label=1, source="StyleGAN3")
    return list(original) + augmented


def write_manifest_csv(entries: list[ManifestEntry], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "source"])
        for e in entries:
            writer.writerow([e.path, e.label, e.source])


def read_manifest_csv(path: Path) -> list[ManifestEntry]:
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entries.append(ManifestEntry(path=row["path"], label=int(row["label"]), source=row["source"]))
    return entries
