import csv
import random
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
    """The base paper's augmented set: StyleGAN3 fakes are *added* to the original.

    Note that this also changes the class prior -- 10k real / 5k fake becomes
    10k real / 10k fake -- so a model trained on this is not comparable to one
    trained on `original` at a fixed decision threshold. Use
    `build_prior_balanced_manifest` for the comparison that holds the prior fixed.
    """
    augmented = scan_source_folder(stylegan3_folder, label=1, source="StyleGAN3")
    return list(original) + augmented


def build_prior_balanced_manifest(
    original: list[ManifestEntry],
    stylegan3_folder: Path,
    replace_fraction: float = 0.5,
    seed: int = 42,
) -> list[ManifestEntry]:
    """Replaces a fraction of the original fakes with StyleGAN3 fakes, keeping the
    real:fake ratio identical to `original`.

    This is the controlled ablation the base paper claims to run ("the only
    difference between the two training settings is the inclusion of
    StyleGAN3-generated images") but does not: adding images changes training-set
    realism *and* the class prior at once. Here only realism changes, so any
    remaining difference is attributable to the synthetic images themselves.
    """
    if not 0.0 <= replace_fraction <= 1.0:
        raise ValueError(f"replace_fraction must be in [0, 1], got {replace_fraction}")

    reals = [e for e in original if e.label == 0]
    fakes = [e for e in original if e.label == 1]
    n_replace = int(round(len(fakes) * replace_fraction))

    stylegan3 = scan_source_folder(stylegan3_folder, label=1, source="StyleGAN3")
    if len(stylegan3) < n_replace:
        raise ValueError(
            f"need {n_replace} StyleGAN3 images to replace {replace_fraction:.0%} of "
            f"{len(fakes)} fakes, but only found {len(stylegan3)} in {stylegan3_folder}"
        )

    rng = random.Random(seed)
    kept_fakes = list(fakes)
    rng.shuffle(kept_fakes)
    kept_fakes = kept_fakes[: len(fakes) - n_replace]

    chosen_sg3 = list(stylegan3)
    rng.shuffle(chosen_sg3)
    chosen_sg3 = chosen_sg3[:n_replace]

    return reals + kept_fakes + chosen_sg3


def build_symmetric_augmented_manifest(
    original: list[ManifestEntry],
    stylegan3_folder: Path,
    perturbed_real_folder: Path,
    n_perturbed_real: int | None = None,
    seed: int = 42,
) -> list[ManifestEntry]:
    """StyleGAN3-augmented, plus perturbed *real* images to match.

    The base paper perturbs 1,000 of the added StyleGAN3 fakes offline while
    leaving the real subset untouched (Sec. III-B), then perturbs every test
    image (Sec. III-A-3). That asymmetry lets a detector learn "degraded => fake",
    which would inflate false positives on a perturbed test set regardless of how
    realistic the fakes are. Adding matched perturbed reals removes the shortcut.

    Generate `perturbed_real_folder` with scripts/perturb_images.py.
    """
    augmented = build_stylegan3_augmented_manifest(original, stylegan3_folder)
    perturbed = scan_source_folder(perturbed_real_folder, label=0, source="Real-Perturbed")
    if n_perturbed_real is not None:
        rng = random.Random(seed)
        rng.shuffle(perturbed)
        perturbed = perturbed[:n_perturbed_real]
    return augmented + perturbed


def split_leave_one_generator_out(
    entries: list[ManifestEntry],
    held_out_source: str,
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    """Splits off one generator family so it is never seen during training.

    The base paper evaluates on generators it also trains on, which measures
    in-distribution accuracy rather than generalization to an unseen generator.
    """
    held_out = [e for e in entries if e.source == held_out_source]
    if not held_out:
        available = sorted({e.source for e in entries})
        raise ValueError(f"no entries with source {held_out_source!r}; available: {available}")
    kept = [e for e in entries if e.source != held_out_source]
    return kept, held_out


def class_counts(entries: list[ManifestEntry]) -> dict[str, int]:
    """Real/fake counts and the fake prior -- the quantity the base paper changed
    without controlling for it."""
    n_real = sum(1 for e in entries if e.label == 0)
    n_fake = sum(1 for e in entries if e.label == 1)
    total = n_real + n_fake
    return {
        "real": n_real,
        "fake": n_fake,
        "total": total,
        "fake_prior": (n_fake / total) if total else 0.0,
    }


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
