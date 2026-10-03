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


def patch_mean_spectrum(image: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """DSF-Net v2 spectral feature: the same patch-wise 2D FFT + log-power as
    Algorithm 1, but averaged over patch *positions* instead of over frequency
    bins. GAN upsampling leaves the same periodic peaks in every patch, so the
    average keeps them while position-specific content averages out.

    image: (B, C, H, W)
    returns: (B, 1, patch_size, patch_size) — the mean log(1 + |FFT2(patch)|^2),
             fftshift-ed so the zero frequency sits at the centre.
    """
    if image.dim() != 4:
        raise ValueError(f"expected a (B, C, H, W) tensor, got shape {tuple(image.shape)}")

    gray = image.mean(dim=1, keepdim=True)                       # (B, 1, H, W)
    patches = gray.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    b = patches.shape[0]
    patches = patches.reshape(b, -1, patch_size, patch_size)      # (B, P, ph, pw)

    log_power = torch.log1p(torch.fft.fft2(patches).abs() ** 2)
    log_power = torch.fft.fftshift(log_power, dim=(-2, -1))
    return log_power.mean(dim=1, keepdim=True)                   # (B, 1, ph, pw)
