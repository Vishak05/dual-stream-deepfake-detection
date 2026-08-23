import numpy as np
import torch
from dsfnet.explain import GradCAM, spectral_map_to_image
from dsfnet.models.baseline import SpatialOnlyClassifier
from dsfnet.models.spectral import patch_log_power_spectrum


def test_gradcam_output_shape_and_range():
    model = SpatialOnlyClassifier(pretrained=False)
    target_layer = model.spatial.features[-2]  # layer4, before avgpool
    cam = GradCAM(model, target_layer)

    x = torch.rand(1, 3, 224, 224)
    heatmap = cam.generate(x, class_idx=1)

    assert heatmap.shape == (1, 7, 7)  # resnet18 layer4 spatial size for 224 input
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_spectral_map_to_image_normalizes_to_unit_range():
    spectral_map = patch_log_power_spectrum(torch.rand(1, 3, 224, 224))
    img = spectral_map_to_image(spectral_map)

    assert isinstance(img, np.ndarray)
    assert img.shape == (7, 7)
    assert img.min() >= 0.0
    assert img.max() <= 1.0 + 1e-6
