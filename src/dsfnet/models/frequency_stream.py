import torch
import torch.nn as nn

from .spectral import patch_log_power_spectrum, patch_radial_power_spectrum

VARIANTS = ("radial", "scalar")


class FrequencyStream(nn.Module):
    """New contribution (pink in Figure 1). Implements Algorithm 2 / Figure 5's
    'Spectral CNN Embedding': BatchNorm, then Conv-BN-ReLU-Pool x2, Conv-BN-ReLU,
    GlobalAvgPool.

    ``variant`` selects the spectral representation from Algorithm 1:
      "radial" (default) -- the n_bins-channel radial profile
      "scalar"           -- the v1 one-value-per-patch map, kept for the ablation

    The leading BatchNorm is Algorithm 2's "standardize S across the batch". It
    matters more than it looks: log-power values are large and far from zero-mean,
    so feeding them straight into a convolution starts training badly conditioned.
    """

    EMBED_DIM = 256

    def __init__(
        self,
        patch_size: int = 32,
        stride: int = 32,
        variant: str = "radial",
        n_bins: int = 16,
        transform: str = "fft",
        window: bool = True,
        drop_dc: bool = True,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

        self.patch_size = patch_size
        self.stride = stride
        self.variant = variant
        self.n_bins = n_bins
        self.transform = transform
        self.window = window
        self.drop_dc = drop_dc

        in_channels = 1 if variant == "scalar" else n_bins

        self.input_norm = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, self.EMBED_DIM, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(self.EMBED_DIM)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def spectral_map(self, x: torch.Tensor) -> torch.Tensor:
        """The Algorithm 1 output for this stream's configuration.

        Exposed separately so explanations and figures can visualize exactly what
        the CNN consumes, rather than re-deriving it.
        """
        if self.variant == "scalar":
            return patch_log_power_spectrum(x, self.patch_size, self.stride)
        return patch_radial_power_spectrum(
            x,
            patch_size=self.patch_size,
            stride=self.stride,
            n_bins=self.n_bins,
            window=self.window,
            drop_dc=self.drop_dc,
            transform=self.transform,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(self.spectral_map(x))
        h = self.pool(self.relu(self.bn1(self.conv1(h))))
        h = self.pool(self.relu(self.bn2(self.conv2(h))))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.gap(h)                     # (B, EMBED_DIM, 1, 1)
        return torch.flatten(h, 1)           # (B, EMBED_DIM)
