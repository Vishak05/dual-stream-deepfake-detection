import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM over the spatial stream's last conv block (layer4)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def spectral_map_to_image(spectral_map: torch.Tensor) -> np.ndarray:
    """Normalizes a (B,1,h,w) spectral map to a plottable (h,w) array in [0,1]."""
    arr = spectral_map.detach().cpu().numpy().squeeze()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr
