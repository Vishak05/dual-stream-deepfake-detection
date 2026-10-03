import torch
from dsfnet.models.spectral import patch_log_power_spectrum, patch_mean_spectrum


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


def test_patch_mean_spectrum_output_shape():
    x = torch.rand(2, 3, 224, 224)
    out = patch_mean_spectrum(x, patch_size=32)
    assert out.shape == (2, 1, 32, 32)


def test_patch_mean_spectrum_keeps_periodic_peak():
    # A vertical stripe pattern with period 4 px puts energy at horizontal
    # frequency 32/4 = 8 bins from the (shifted) centre, in every patch.
    cols = torch.arange(64)
    stripes = (cols % 4 < 2).float().expand(1, 3, 64, 64)
    spectrum = patch_mean_spectrum(stripes, patch_size=32)[0, 0]
    centre = 16
    assert spectrum[centre, centre + 8] > spectrum[centre + 8, centre]
    assert spectrum[centre, centre + 8] > spectrum[centre, centre + 5]
