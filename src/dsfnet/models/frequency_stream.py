import torch
import torch.nn as nn
from .spectral import patch_log_power_spectrum, patch_mean_spectrum


class FrequencyStream(nn.Module):
    """New contribution (pink in Figure 1). Implements Algorithm 2 / Figure 5's
    'Spectral CNN Embedding' box: Conv-ReLU-Pool x2, Conv-ReLU, GlobalAvgPool.

    mode="patch_energy" (v1): input is the (H/32 x W/32) map of per-patch mean
    log-power. mode="mean_spectrum" (DSF-Net v2): input is the 32x32 log-power
    spectrum averaged over patches, which keeps the spectral shape."""

    EMBED_DIM = 256
    MODES = ("patch_energy", "mean_spectrum")

    def __init__(self, patch_size: int = 32, stride: int = 32, mode: str = "patch_energy"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        # Log-power values are large (~10-20); normalize them before the convs.
        self.input_norm = nn.BatchNorm2d(1) if mode == "mean_spectrum" else nn.Identity()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, self.EMBED_DIM, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def spectral_map(self, x: torch.Tensor) -> torch.Tensor:
        """The (B,1,h,w) spectral input this stream's CNN sees."""
        if self.mode == "mean_spectrum":
            return patch_mean_spectrum(x, self.patch_size)
        return patch_log_power_spectrum(x, self.patch_size, self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectral_map = self.input_norm(self.spectral_map(x))
        h = self.pool(self.relu(self.conv1(spectral_map)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.relu(self.conv3(h))
        h = self.gap(h)                     # (B, EMBED_DIM, 1, 1)
        return torch.flatten(h, 1)           # (B, EMBED_DIM)
