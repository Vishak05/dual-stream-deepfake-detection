import torch
import torch.nn as nn

from .spatial_stream import SpatialStream
from .frequency_stream import FrequencyStream


class DualStreamNet(nn.Module):
    """Hybrid Fusion + Classification (blue/modified in Figure 1, Figure 6).
    Existing fusion+classification pattern, modified to accept two input
    streams instead of one."""

    def __init__(
        self,
        pretrained_spatial: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        **frequency_kwargs,
    ):
        super().__init__()
        self.spatial = SpatialStream(pretrained=pretrained_spatial)
        self.frequency = FrequencyStream(**frequency_kwargs)
        fused_dim = SpatialStream.EMBED_DIM + FrequencyStream.EMBED_DIM  # 768
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_s = self.spatial(x)
        f_f = self.frequency(x)
        z = torch.cat([f_s, f_f], dim=1)
        return self.classifier(z)
