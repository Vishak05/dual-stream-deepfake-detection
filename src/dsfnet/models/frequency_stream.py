import torch
import torch.nn as nn
from .spectral import patch_log_power_spectrum


class FrequencyStream(nn.Module):
    """New contribution (pink in Figure 1). Implements Algorithm 2 / Figure 5's
    'Spectral CNN Embedding' box: Conv-ReLU-Pool x2, Conv-ReLU, GlobalAvgPool."""

    EMBED_DIM = 256

    def __init__(self, patch_size: int = 32, stride: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, self.EMBED_DIM, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectral_map = patch_log_power_spectrum(x, self.patch_size, self.stride)  # (B,1,h,w)
        h = self.pool(self.relu(self.conv1(spectral_map)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.relu(self.conv3(h))
        h = self.gap(h)                     # (B, EMBED_DIM, 1, 1)
        return torch.flatten(h, 1)           # (B, EMBED_DIM)
