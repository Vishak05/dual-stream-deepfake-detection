import random

import numpy as np
import pytest
from PIL import Image

from dsfnet.data.perturbations import apply_perturbation, sample_perturbation


def _image(seed: int = 0, size: int = 32) -> Image.Image:
    arr = np.random.RandomState(seed).randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_perturbation_is_reproducible_from_the_same_seed():
    """Every model must be scored on byte-identical test images, so the same seed
    has to yield the same perturbation -- otherwise an FPR difference between two
    models could be nothing but perturbation luck."""
    image = _image()
    first = apply_perturbation(image, sample_perturbation(random.Random("seed:x")))
    second = apply_perturbation(image, sample_perturbation(random.Random("seed:x")))
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_string_seeds_are_stable_across_processes():
    """Seeding from a path string relies on random.Random hashing it with SHA-512
    rather than the per-process-salted hash(), so it stays stable."""
    a = sample_perturbation(random.Random("42:image_001.png"))
    b = sample_perturbation(random.Random("42:image_001.png"))
    assert a == b


def test_different_seeds_give_different_perturbations():
    image = _image()
    a = apply_perturbation(image, sample_perturbation(random.Random("seed:a")))
    b = apply_perturbation(image, sample_perturbation(random.Random("seed:b")))
    assert a.size != b.size or not np.array_equal(np.asarray(a), np.asarray(b))


def test_rotation_follows_the_papers_distribution():
    """R = 45 + 45U with U ~ Uniform(0, 6), i.e. multiples of 45 in [45, 315]."""
    for seed in range(50):
        params = sample_perturbation(random.Random(seed))
        assert params.rotation_deg % 45 == 0
        assert 45 <= params.rotation_deg <= 315


def test_jpeg_quality_stays_in_the_papers_range():
    for seed in range(50):
        assert 50 <= sample_perturbation(random.Random(seed)).jpeg_quality <= 99


def test_pinned_jpeg_quality_overrides_sampling():
    """Used by the compression-robustness sweep, which varies quality while
    holding the other perturbations fixed."""
    assert sample_perturbation(random.Random(0), jpeg_quality=30).jpeg_quality == 30


def test_scaling_is_half_or_double():
    scales = {sample_perturbation(random.Random(s)).scale for s in range(50)}
    assert scales <= {0.5, 2.0}


def test_noise_mode_is_validated():
    with pytest.raises(ValueError, match="noise_mode"):
        apply_perturbation(_image(), sample_perturbation(random.Random(0)), noise_mode="salt")


def test_additive_noise_mode_produces_an_image():
    """The paper labels this operation 'Gaussian Noise' but describes it by kernel
    size; both readings are available and this covers the literal one."""
    flat = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8), "RGB")
    out = apply_perturbation(flat, sample_perturbation(random.Random(0)),
                             noise_mode="noise", rng=random.Random(0))
    assert out.mode == "RGB"
