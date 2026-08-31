import numpy as np
import pytest
import torch

from dsfnet.models.spectral import (
    _dct2d,
    patch_log_power_spectrum,
    patch_radial_power_spectrum,
)


# --- v1 scalar representation (kept as the ablation arm) --------------------

def test_patch_log_power_spectrum_output_shape():
    x = torch.rand(2, 3, 224, 224)
    out = patch_log_power_spectrum(x, patch_size=32, stride=32)
    assert out.shape == (2, 1, 7, 7)  # 224 // 32 == 7


def test_patch_log_power_spectrum_rejects_wrong_ndim():
    x = torch.rand(3, 224, 224)  # missing batch dim
    try:
        patch_log_power_spectrum(x)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_patch_log_power_spectrum_distinguishes_texture_from_flat():
    torch.manual_seed(0)
    constant = torch.full((1, 3, 32, 32), 0.5)
    noisy = torch.rand(1, 3, 32, 32)

    e_const = patch_log_power_spectrum(constant, patch_size=32, stride=32).item()
    e_noisy = patch_log_power_spectrum(noisy, patch_size=32, stride=32).item()

    assert e_noisy > e_const


# --- v2 radial representation ----------------------------------------------

def _dct2_reference(x: np.ndarray) -> np.ndarray:
    """Direct O(N^2) DCT-II, from the definition, as an independent check."""
    n = x.shape[-1]
    samples = np.arange(n)
    basis = 2.0 * np.cos(np.pi * (2 * samples[None, :] + 1) * samples[:, None] / (2 * n))
    return basis @ x @ basis.T


def test_radial_profile_has_one_channel_per_bin():
    out = patch_radial_power_spectrum(torch.randn(2, 3, 224, 224), n_bins=16)
    assert out.shape == (2, 16, 7, 7)


def test_dct_matches_direct_definition():
    x = torch.randn(3, 8, 8, dtype=torch.float64)
    mine = _dct2d(x).numpy()
    reference = np.stack([_dct2_reference(p) for p in x.numpy()])
    assert np.abs(mine - reference).max() < 1e-9


def test_injected_periodic_tone_lands_in_a_high_radial_bin():
    """The point of the redesign: a periodic pattern -- the signature GAN
    upsampling leaves -- must show up as a peak at a specific frequency. The v1
    scalar representation averages over every bin and cannot express this."""
    size, patch = 64, 32
    ramp = torch.arange(size, dtype=torch.float32)
    # high-frequency vertical stripes, near Nyquist
    stripes = torch.sin(2 * torch.pi * ramp * 0.4).view(1, 1, 1, size).expand(1, 3, size, size)
    flat = torch.zeros(1, 3, size, size)

    tone = patch_radial_power_spectrum(stripes, patch_size=patch, stride=patch, n_bins=8)
    plain = patch_radial_power_spectrum(flat, patch_size=patch, stride=patch, n_bins=8)

    tone_profile = tone.mean(dim=(0, 2, 3))
    plain_profile = plain.mean(dim=(0, 2, 3))
    peak_bin = int(torch.argmax(tone_profile - plain_profile))

    assert peak_bin >= 4, f"expected the tone in an upper bin, got bin {peak_bin}"
    assert tone_profile[peak_bin] > plain_profile[peak_bin]


@pytest.mark.parametrize("window", [True, False])
def test_profile_is_invariant_to_brightness(window):
    """Mean brightness is not a forgery cue, so a constant offset must not move
    the profile. Patch mean-centring is what guarantees this: windowing a patch
    with a non-zero mean multiplies the offset by the window shape, spreading it
    across low-frequency bins where dropping the DC term cannot reach it."""
    x = torch.randn(1, 3, 64, 64)
    base = patch_radial_power_spectrum(x, patch_size=32, stride=32, window=window)
    shifted = patch_radial_power_spectrum(x + 2.0, patch_size=32, stride=32, window=window)
    assert torch.allclose(base, shifted, atol=1e-4)


def test_hann_window_reduces_axis_leakage():
    """An unwindowed FFT of a non-periodic patch leaks energy along both axes,
    which can masquerade as genuine periodicity."""
    size = 64
    ramp = torch.linspace(0, 1, size)
    # a linear gradient is maximally non-periodic across the patch boundary
    gradient = ramp.view(1, 1, 1, size).expand(1, 3, size, size).contiguous()

    windowed = patch_radial_power_spectrum(gradient, patch_size=32, stride=32, window=True)
    unwindowed = patch_radial_power_spectrum(gradient, patch_size=32, stride=32, window=False)

    high_bins = slice(8, None)
    assert windowed[:, high_bins].mean() < unwindowed[:, high_bins].mean()


def test_radial_spectrum_rejects_bad_arguments():
    x = torch.randn(1, 3, 64, 64)
    with pytest.raises(ValueError, match="B, C, H, W"):
        patch_radial_power_spectrum(torch.randn(3, 64, 64))
    with pytest.raises(ValueError, match="fft.*dct"):
        patch_radial_power_spectrum(x, transform="wavelet")
    with pytest.raises(ValueError, match="n_bins"):
        patch_radial_power_spectrum(x, n_bins=0)
