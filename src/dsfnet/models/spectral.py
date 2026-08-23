import torch


def patch_log_power_spectrum(
    image: torch.Tensor,
    patch_size: int = 32,
    stride: int = 32,
) -> torch.Tensor:
    """Algorithm 1 (Figure 2): patch-wise 2D FFT decomposition + log-power spectrum.

    image: (B, C, H, W)
    returns: (B, 1, H//stride, W//stride) — one scalar per patch, the mean
             log(1 + |FFT2(patch)|^2) over that patch's frequency bins.
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
