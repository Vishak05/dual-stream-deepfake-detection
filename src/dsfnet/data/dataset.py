from typing import Callable
from PIL import Image
from torch.utils.data import Dataset
from .manifest import ManifestEntry


class DeepfakeImageDataset(Dataset):
    def __init__(self, entries: list[ManifestEntry], transform: Callable):
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB")
        tensor = self.transform(image)
        return tensor, entry.label
