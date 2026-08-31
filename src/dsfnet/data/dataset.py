from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from .manifest import ManifestEntry


class DeepfakeImageDataset(Dataset):
    def __init__(
        self,
        entries: list[ManifestEntry],
        transform: Callable,
        return_source: bool = False,
    ):
        """`return_source` yields (tensor, label, source) instead of (tensor, label).

        The manifest already carries `source`, but dropping it here made
        per-generator metrics impossible -- which is exactly the breakdown needed
        to show whether a gain is concentrated on unseen generators.
        """
        self.entries = entries
        self.transform = transform
        self.return_source = return_source

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB")
        tensor = self.transform(image)
        if self.return_source:
            return tensor, entry.label, entry.source
        return tensor, entry.label
