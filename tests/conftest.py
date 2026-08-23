import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def make_dummy_image(tmp_path):
    def _make(name: str = "img.png", size: tuple[int, int] = (16, 16), seed: int = 0):
        rng = np.random.RandomState(seed)
        arr = rng.randint(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        path = tmp_path / name
        Image.fromarray(arr, "RGB").save(path)
        return path
    return _make
