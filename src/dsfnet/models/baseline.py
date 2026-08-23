import torch
import torch.nn as nn
from .spatial_stream import SpatialStream


class SpatialOnlyClassifier(nn.Module):
    """Reproduces the base paper's ResNet-18 baseline (spatial stream + FC head)."""

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.spatial = SpatialStream(pretrained=pretrained)
        self.classifier = nn.Linear(SpatialStream.EMBED_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.spatial(x))
