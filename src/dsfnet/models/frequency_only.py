import torch
import torch.nn as nn

from .frequency_stream import FrequencyStream


class FrequencyOnlyClassifier(nn.Module):
    """Frequency stream + linear head, mirroring SpatialOnlyClassifier.

    This is the honesty check for the project's core hypothesis: if the spectral
    branch alone cannot separate real from fake above chance, then the dual-stream
    model's fusion is decorative and the gain is coming from the spatial stream.
    That needs to be known before any result is written up, not after.
    """

    def __init__(self, num_classes: int = 2, **frequency_kwargs):
        super().__init__()
        self.frequency = FrequencyStream(**frequency_kwargs)
        self.classifier = nn.Linear(FrequencyStream.EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.frequency(x))
