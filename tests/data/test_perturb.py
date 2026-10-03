import random

import numpy as np
from PIL import Image

from dsfnet.data.perturb import TEST_OPS, base_paper_perturbation, training_degradation


def _image(seed: int = 0, size: int = 64) -> Image.Image:
    arr = np.random.RandomState(seed).randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def test_base_paper_perturbation_is_deterministic_per_seed():
    a, op_a, q_a = base_paper_perturbation(_image(), random.Random(7))
    b, op_b, q_b = base_paper_perturbation(_image(), random.Random(7))
    assert op_a == op_b and q_a == q_b
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_base_paper_perturbation_reaches_every_op_with_paper_parameters():
    seen = {}
    for seed in range(300):
        _, op, quality = base_paper_perturbation(_image(), random.Random(seed))
        name, param = op.split(":")
        seen.setdefault(name, set()).add(param)
        assert (quality is not None) == (name == "jpeg")
    assert set(seen) == set(TEST_OPS)
    assert {int(a) for a in seen["rotation"]} <= {45, 90, 135, 180, 225, 270, 315}
    assert seen["scaling"] == {"0.5", "2.0"}
    assert all(50 <= int(q) <= 99 for q in seen["jpeg"])


def test_scaling_changes_size_and_other_ops_keep_it():
    for seed in range(100):
        out, op, _ = base_paper_perturbation(_image(size=64), random.Random(seed))
        expected = {"scaling:0.5": (32, 32), "scaling:2.0": (128, 128)}.get(op, (64, 64))
        assert out.size == expected


def test_training_degradation_keeps_size_and_changes_pixels():
    img = _image(size=64)
    out = training_degradation(img, random.Random(3))
    assert out.size == (64, 64)
    assert not np.array_equal(np.asarray(out), np.asarray(img))
