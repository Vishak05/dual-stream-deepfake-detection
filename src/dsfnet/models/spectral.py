"""Patch-wise spectral decomposition (Algorithm 1).

Two representations live here:

``patch_log_power_spectrum``  -- v1, one scalar per patch. Kept as the ablation
    arm. It averages the log-power over *every* frequency bin, which reduces a
    patch to a texture-energy measure: two patches with the same total energy but
    completely different spectral shape map to the same number. GAN upsampling
    artifacts are peaks at particular frequencies plus an elevated high-frequency
    tail, so this representation cannot express the hypothesis it was built for.

``patch_radial_power_spectrum`` -- v2, an ``n_bins``-channel radial profile that
    preserves the *shape* of the spectrum. This is what the project actually
    tests. It additionally windows each patch (spectral leakage from a
    non-periodic patch otherwise puts a cross-shaped artifact along both axes,
    masking genuine periodicity) and discards the DC term (mean patch brightness,
    which is not a forgery cue and dominates an unweighted average).
"""

from functools import lru_cache

import torch

from ..data.transforms import denormalize_imagenet

# ITU-R BT.601 luma coefficients.
_BT601 = (0.299, 0.587, 0.114)


def patch_log_power_spectrum(
    image: torch.Tensor,
    patch_size: int = 32,
    stride: int = 32,
) -> torch.Tensor:
    """Algorithm 1, v1 (scalar): patch-wise 2D FFT + mean log-power spectrum.

    image: (B, C, H, W)
    returns: (B, 1, H//stride, W//stride) -- one scalar per patch, the mean
             log(1 + |FFT2(patch)|^2) over that patch's frequency bins.

    Retained unchanged as the ablation baseline for the radial representation.
    """
    if image.dim() != 4:
        raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(image.shape)}")

    gray = image.mean(dim=1, keepdim=True)                      # (B, 1, H, W)
    patches = gray.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    b, c, nh, nw, ph, pw = patches.shape
    patches = patches.reshape(b * nh * nw, ph, pw)

    spectrum = torch.fft.fft2(patches)
    power = spectrum.abs() ** 2
    log_power = torch.log1p(power)
    patch_energy = log_power.mean(dim=(1, 2))                   # (B*nh*nw,)

    return patch_energy.reshape(b, 1, nh, nw)


def patch_radial_power_spectrum(
    image: torch.Tensor,
    patch_size: int = 32,
    stride: int = 32,
    n_bins: int = 16,
    window: bool = True,
    drop_dc: bool = True,
    transform: str = "fft",
) -> torch.Tensor:
    """Algorithm 1, v2 (radial): windowed patch transform -> log-power with the DC
    term discarded -> radial frequency binning.

    image: (B, C, H, W), assumed ImageNet-normalized
    returns: (B, n_bins, H//stride, W//stride)

    Bin 0 is the lowest non-DC radial frequency and bin ``n_bins - 1`` the
    highest, so upsampling peaks and the high-frequency tail stay separable.
    """
    if image.dim() != 4:
        raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(image.shape)}")
    transform = transform.lower()
    if transform not in ("fft", "dct"):
        raise ValueError(f"transform must be 'fft' or 'dct', got {transform!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    gray = _luminance(image)                                     # (B, 1, H, W)
    patches = gray.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    b, _, nh, nw, ph, pw = patches.shape
    patches = patches.reshape(b * nh * nw, ph, pw)

    # Remove each patch's mean before windowing. Without this the brightness
    # offset is multiplied by the window shape and smears across many low
    # frequency bins, so dropping the single DC term would not remove it -- mean
    # brightness would leak into the profile and act as a spurious cue.
    patches = patches - patches.mean(dim=(-2, -1), keepdim=True)

    if window:
        patches = patches * _hann2d(patch_size, patches.dtype, patches.device)

    if transform == "fft":
        # fftshift puts the DC term at (p//2, p//2), so radii are measured from
        # the patch centre.
        spectrum = torch.fft.fftshift(torch.fft.fft2(patches), dim=(-2, -1))
        power = spectrum.abs() ** 2
        centered = True
    else:
        # DCT-II keeps the DC term at (0, 0), so radii are measured from there.
        power = _dct2d(patches) ** 2
        centered = False

    log_power = torch.log1p(power).reshape(patches.shape[0], -1)

    keep_idx, bin_idx, counts = _radial_layout(patch_size, n_bins, centered, drop_dc)
    keep_idx = keep_idx.to(log_power.device)
    bin_idx = bin_idx.to(log_power.device)
    counts = counts.to(log_power.device, log_power.dtype)

    kept = log_power.index_select(1, keep_idx)
    binned = torch.zeros(kept.shape[0], n_bins, dtype=kept.dtype, device=kept.device)
    binned.index_add_(1, bin_idx, kept)
    binned = binned / counts.clamp(min=1.0)

    return binned.reshape(b, nh, nw, n_bins).permute(0, 3, 1, 2).contiguous()


def _luminance(image: torch.Tensor) -> torch.Tensor:
    """BT.601 luminance of an ImageNet-normalized batch, as (B, 1, H, W)."""
    rgb = denormalize_imagenet(image)
    weights = torch.tensor(_BT601, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


@lru_cache(maxsize=None)
def _hann2d_cpu(patch_size: int) -> torch.Tensor:
    w = torch.hann_window(patch_size, periodic=False, dtype=torch.float64)
    return torch.outer(w, w)


def _hann2d(patch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return _hann2d_cpu(patch_size).to(device=device, dtype=dtype)


@lru_cache(maxsize=None)
def _radial_layout(
    patch_size: int,
    n_bins: int,
    centered: bool,
    drop_dc: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Maps each frequency bin of a patch to a radial band.

    Returns (keep_idx, bin_idx, counts): the flattened positions to keep, the
    radial band each kept position belongs to, and how many positions fall in
    each band (for averaging).
    """
    p = patch_size
    origin = float(p // 2) if centered else 0.0
    coords = torch.arange(p, dtype=torch.float32)
    dy = (coords - origin).view(-1, 1)
    dx = (coords - origin).view(1, -1)
    radius = torch.sqrt(dy ** 2 + dx ** 2).flatten()

    keep = torch.ones_like(radius, dtype=torch.bool)
    if drop_dc:
        keep[int(origin) * p + int(origin)] = False

    keep_idx = keep.nonzero(as_tuple=False).squeeze(1)
    kept_radius = radius[keep_idx]
    r_max = float(kept_radius.max())
    bin_idx = torch.clamp((kept_radius / r_max * n_bins).long(), max=n_bins - 1)
    counts = torch.bincount(bin_idx, minlength=n_bins).to(torch.float32)
    return keep_idx, bin_idx, counts


def _dct1d(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized DCT-II along the last dimension, built on torch.fft.

    torch has no DCT, so this uses the standard even-odd reordering identity:
    reorder the samples, take an FFT, and phase-shift the result.
    """
    n = x.shape[-1]
    reordered = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    spectrum = torch.fft.fft(reordered, dim=-1)
    k = torch.arange(n, dtype=x.dtype, device=x.device)
    phase = torch.polar(torch.ones_like(k), -torch.pi * k / (2 * n))
    return 2.0 * (spectrum * phase).real


def _dct2d(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized 2D DCT-II over the last two dimensions (separable)."""
    return _dct1d(_dct1d(x).transpose(-1, -2)).transpose(-1, -2)
