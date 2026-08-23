import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SpatialStream(nn.Module):
    """Existing module, unmodified from the base paper (green in Figure 1)."""

    EMBED_DIM = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop final FC

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)          # (B, 512, 1, 1)
        return torch.flatten(feats, 1)     # (B, 512)
