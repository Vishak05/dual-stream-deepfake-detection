"""Run against a real trained checkpoint (local or Colab) to produce Grad-CAM
heatmaps (spatial stream) and spectral-map visualizations (frequency stream,
dual_stream model only) for a handful of test images — used for report/PPT
snapshots (Figure 4-style visualizations)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dsfnet.data.manifest import read_manifest_csv
from dsfnet.data.transforms import build_eval_transform
from dsfnet.models.baseline import SpatialOnlyClassifier
from dsfnet.models.dual_stream import DualStreamNet
from dsfnet.models.spectral import patch_log_power_spectrum
from dsfnet.explain import GradCAM, spectral_map_to_image

_MODELS = {
    "spatial_only": SpatialOnlyClassifier,
    "dual_stream": DualStreamNet,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM and spectral-map visualizations from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=list(_MODELS), required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV from dsfnet.data.manifest.write_manifest_csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10, help="How many images (from the start of the manifest) to explain.")
    parser.add_argument("--class-idx", type=int, default=1, help="Class to explain Grad-CAM for (1 = fake).")
    return parser


def _array_to_png(arr: np.ndarray, out_path: Path, size: tuple[int, int] = (224, 224)) -> None:
    """Upscales a normalized [0,1] 2D array into a viewable grayscale PNG."""
    img_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(img_u8, mode="L").resize(size, resample=Image.NEAREST)
    image.save(out_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _MODELS[args.model]().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    target_layer = model.spatial.features[-2]  # layer4, before avgpool
    cam = GradCAM(model, target_layer)

    entries = read_manifest_csv(args.manifest)[: args.num_samples]
    transform = build_eval_transform()

    for i, entry in enumerate(entries):
        image = Image.open(entry.path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)

        heatmap = cam.generate(x, class_idx=args.class_idx)[0]  # (7, 7)
        _array_to_png(heatmap, args.out_dir / f"gradcam_{i:03d}_{entry.source}_label{entry.label}.png")

        if args.model == "dual_stream":
            spectral_map = patch_log_power_spectrum(x)
            spectral_img = spectral_map_to_image(spectral_map)
            _array_to_png(spectral_img, args.out_dir / f"spectral_{i:03d}_{entry.source}_label{entry.label}.png")

        print(f"[{i + 1}/{len(entries)}] wrote explanation(s) for {entry.path}")


if __name__ == "__main__":
    main()
