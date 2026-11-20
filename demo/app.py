"""Interactive demo: the base paper's method vs. DSF-Net v2 (proposed) on one face.

Pick or upload a face, apply one of the base paper's test-set perturbations
(Sec. III-A3), and compare both detectors' verdicts, Grad-CAM heatmaps and the
DSF-Net v2 frequency input.

    python demo/app.py --runs-dir ~/dsfnet-runs --examples-dir demo/examples
"""
import argparse
import json
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

from dsfnet.data.perturb import jpeg_roundtrip
from dsfnet.data.transforms import build_eval_transform
from dsfnet.explain import GradCAM, spectral_map_to_image
from dsfnet.models import MODELS

# (run name in evaluation_perturbed.json, label shown in the UI)
BASE_RUN = ("paper_repro", "Base paper's method (ResNet-18, StyleGAN3 augmentation)")
OURS_RUN = ("dsfnet_v2_robust", "Proposed: DSF-Net v2 + symmetric augmentation + calibration")

PERTURBATIONS = {
    "None (clean image)": lambda im: im,
    "Rotation 90°": lambda im: im.rotate(90),
    "Rotation 135°": lambda im: im.rotate(135),
    "Rotation 180°": lambda im: im.rotate(180),
    "Mirror (horizontal)": ImageOps.mirror,
    "Mirror (vertical)": ImageOps.flip,
    "Gaussian blur (kernel 9)": lambda im: F.gaussian_blur(im, kernel_size=9),
    "Gaussian blur (kernel 13)": lambda im: F.gaussian_blur(im, kernel_size=13),
    "Scale ×0.5": lambda im: im.resize((im.width // 2, im.height // 2), Image.BILINEAR),
    "Scale ×2": lambda im: im.resize((im.width * 2, im.height * 2), Image.BILINEAR),
    "JPEG quality 50": lambda im: jpeg_roundtrip(im, 50),
    "JPEG quality 75": lambda im: jpeg_roundtrip(im, 75),
}


class Detector:
    """One trained run: its model, Grad-CAM hook and decision threshold (on the
    logit margin fake - real)."""

    def __init__(self, run: dict, threshold: float, device: torch.device):
        self.model = MODELS[run["model"]]()
        self.model.load_state_dict(torch.load(run["checkpoint"], map_location=device))
        self.model.to(device).eval()
        self.cam = GradCAM(self.model, self.model.spatial.features[-2])
        self.threshold = threshold
        self.device = device

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        x = x.to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0]
        margin = float(logits[1] - logits[0])
        heatmap = self.cam.generate(x, class_idx=1)[0]
        return margin, heatmap


def overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = image.convert("RGB").resize((224, 224))
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = 0.55 * np.asarray(base) / 255.0 + 0.45 * colored
    return Image.fromarray((blended * 255).astype(np.uint8))


def verdict_markdown(name: str, margin: float, threshold: float) -> str:
    is_fake = margin > threshold
    prob = 1 / (1 + np.exp(-margin))
    color, word = ("#dc2626", "FAKE") if is_fake else ("#16a34a", "REAL")
    label = f"<span style='font-size:2.4em;font-weight:800;color:{color}'>● {word}</span>"
    return (
        f"### {name}\n"
        f"{label}\n\n"
        f"Score (logit margin fake − real): **{margin:+.2f}**  \n"
        f"Decision threshold: {threshold:+.2f}  \n"
        f"Softmax P(fake): {prob:.4f}"
    )


def build_app(runs_dir: Path, examples_dir: Path | None) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation = json.loads((runs_dir / "evaluation_perturbed.json").read_text())["runs"]
    # The base paper uses the default 0.5 probability threshold (margin 0); the
    # proposed method uses its threshold calibrated to 5% FPR on perturbed real
    # validation images.
    base = Detector(evaluation[BASE_RUN[0]], 0.0, device)
    ours = Detector(evaluation[OURS_RUN[0]], evaluation[OURS_RUN[0]]["calibrated"]["threshold"], device)
    transform = build_eval_transform()

    def run(image: Image.Image, perturbation: str):
        if image is None:
            raise gr.Error("Pick an example or upload a face image first.")
        image = image.convert("RGB")
        perturbed = PERTURBATIONS[perturbation](image)
        x = transform(perturbed).unsqueeze(0)

        base_margin, base_cam = base(x)
        ours_margin, ours_cam = ours(x)
        with torch.no_grad():
            spectrum = spectral_map_to_image(ours.model.frequency.spectral_map(x.to(device)))
        spectrum_img = Image.fromarray(
            (matplotlib.colormaps["magma"](spectrum)[..., :3] * 255).astype(np.uint8)
        ).resize((224, 224), Image.NEAREST)

        return (
            perturbed,
            verdict_markdown(BASE_RUN[1], base_margin, base.threshold),
            overlay(perturbed, base_cam),
            verdict_markdown(OURS_RUN[1], ours_margin, ours.threshold),
            overlay(perturbed, ours_cam),
            spectrum_img,
        )

    with gr.Blocks(title="DSF-Net Deepfake Detection Demo") as app:
        gr.Markdown(
            "# DSF-Net: Robust Deepfake Face Detection\n"
            "Comparing the base paper's method (Hafezi et al., IEEE Access 2026) with our proposed "
            "DSF-Net v2 under the base paper's own test perturbations."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Input face", height=280)
                perturbation = gr.Dropdown(list(PERTURBATIONS), value="None (clean image)",
                                           label="Perturbation (base paper's test protocol)")
                button = gr.Button("Detect", variant="primary")
            with gr.Column(scale=1):
                perturbed_out = gr.Image(label="Image the models see", height=280)
                spectrum_out = gr.Image(label="DSF-Net v2 frequency input (mean 32×32 log-power spectrum)",
                                        height=224)
        with gr.Row():
            with gr.Column():
                base_md = gr.Markdown()
                base_cam = gr.Image(label="Grad-CAM: where the base model looks", height=224)
            with gr.Column():
                ours_md = gr.Markdown()
                ours_cam = gr.Image(label="Grad-CAM: where DSF-Net v2 looks", height=224)

        outputs = [perturbed_out, base_md, base_cam, ours_md, ours_cam, spectrum_out]
        button.click(run, [image_in, perturbation], outputs)

        if examples_dir and (examples_dir / "examples.json").exists():
            spec = json.loads((examples_dir / "examples.json").read_text())
            gr.Examples(
                examples=[[str(examples_dir / e["file"]), e["perturbation"]] for e in spec],
                example_labels=[e["label"] for e in spec],
                inputs=[image_in, perturbation],
                outputs=outputs,
                fn=run,
                run_on_click=True,
                label="Examples from the held-out test set",
            )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the DSF-Net comparison demo.")
    parser.add_argument("--runs-dir", type=Path, default=Path.home() / "dsfnet-runs")
    parser.add_argument("--examples-dir", type=Path, default=Path(__file__).parent / "examples")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # Gradio's default font ships only weights 400/600, so the 800-weight verdicts
    # fell back to another font for some letters; Inter has all three.
    theme = gr.themes.Default(font=[gr.themes.GoogleFont("Inter", weights=(400, 600, 800)), "sans-serif"])
    build_app(args.runs_dir, args.examples_dir).launch(server_port=args.port, theme=theme)


if __name__ == "__main__":
    main()
