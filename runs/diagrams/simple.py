"""Simplified architecture and module diagrams (the detailed ones are in detailed/)."""
from PIL import Image
import diagram_lib
from diagram_lib import Diagram, STYLE
_box = Diagram.box
Diagram.box = lambda self, *a, **k: _box(self, *a, **{"ts": 25, "ds": 20, **k})

S = "snapshots/"
NAMES = {"base": "Base paper", "ours": "Our work", "mod": "Base-paper step, changed by us", "offline": "Offline stage"}


def legend_row(d, x, y, entries):
    for e in entries:
        if e == "offline":
            d.el.append(f'<rect x="{x}" y="{y}" width="52" height="30" rx="7" fill="#fff" stroke="#222" stroke-width="2" stroke-dasharray="7,5"/>')
        else:
            s = STYLE[e]
            dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
            d.el.append(f'<rect x="{x}" y="{y}" width="52" height="30" rx="7" fill="{s["fill"]}" stroke="{s["stroke"]}" stroke-width="{s["sw"]}"{dash}/>')
        d.text(x + 66, y + 22, NAMES[e], size=20, anchor="start")
        x += 66 + 11 * len(NAMES[e]) + 60


def overall():
    d = Diagram(2400, 800, "ARCHITECTURE DIAGRAM")
    W, H, cy = 320, 210, 400
    m1 = d.box(80, cy - H / 2, W, H, "1. Data\nAcquisition", "Kaggle 140k (FFHQ, StyleGAN)\n+ StyleGAN3 / StyleGAN2", "mod")
    m2 = d.box(460, cy - H / 2, W, H, "2. Preprocessing", "label-symmetric augmentation\n224×224, ImageNet norm", "mod")
    m3 = d.box(840, cy - 125 - H / 2, 345, H, "3. Spatial Stream", "ResNet-18 (ImageNet)\n→ 512-d embedding", "base")
    m4 = d.box(840, cy + 125 - H / 2, 345, H, "4. Frequency Stream", "patch-wise 2D FFT,\nmean log-power spectrum\n→ CNN → 256-d", "ours")
    m5 = d.box(1265, cy - H / 2, W, H, "5. Fusion", "concat 768-d → FC head\nscore s = z_fake − z_real", "ours")
    m6 = d.box(1655, cy - H / 2, W, H, "6. Decision", "fake if s > θ", "ours")
    out = d.pill(2050, cy - 40, 290, 80, "Real / Fake")
    cal = d.module(1635, 600, 360, 150, "Threshold calibration", None, dashed=True)
    d.text(1815, 680, "θ = 95th percentile of real\nvalidation scores (5% FPR)", size=19, italic=True)
    d.text(80, 160, "Input: real and fake face images", size=21, anchor="start")
    d.arrow((m1.cx, 172), m1.t)
    d.arrow(m1.r, m2.l)
    d.arrow(m2.r, (810, cy), (810, m3.cy), m3.l)
    d.arrow((810, m4.cy), m4.l)
    d.dot(810, cy)
    d.arrow(m3.r, (1225, m3.cy), (1225, cy), m5.l)
    d.arrow(m4.r, (1225, m4.cy), (1225, cy), head=False)
    d.arrow(m5.r, m6.l, label="s", at=(1620, cy - 12))
    d.arrow((m6.cx, 600), m6.b, label="θ", at=(m6.cx + 12, 570), anchor="start")
    d.arrow(m6.r, out.l)
    d.save("architecture")


def module(stem, num, name, sub, inp, steps, output, snap, caption, legend):
    n = len(steps)
    gap = 110
    bw = min(460, (2160 - gap * (n - 1)) // n)
    x0 = 60 + (2280 - (bw * n + gap * (n - 1))) / 2
    H = 490
    d = Diagram(2400, H, f"MODULE {num}: {name.upper()}")
    d.module(60, 110, 2280, 300, f"{num}. {name}", sub)
    boxes = [d.box(x0 + i * (bw + gap), 210, bw, 150, t, det, k) for i, (t, det, k) in enumerate(steps)]
    for a, b in zip(boxes, boxes[1:]):
        d.arrow(a.r, b.l)
    d.text(x0 - 20, 450, f"in: {inp}", size=20, italic=True, anchor="start")
    d.text(boxes[-1].x + bw + 20, 450, f"out: {output}", size=20, italic=True, anchor="end")
    d.save(stem)


overall()
module("module1_data", 1, "Data Acquisition", "real faces, training fakes, and unseen fakes for testing", "Kaggle + NVIDIA generators",
       [("Kaggle 140k", "FFHQ real, StyleGAN fake", "base"),
        ("StyleGAN3 generation", "5,000 StyleGAN3-R, ψ = 0.7", "base"),
        ("Unseen test fakes", "StyleGAN3-T and StyleGAN2", "ours"),
        ("Manifests", "train / test (23k) / validation (10k)", "mod")],
       "datasets → Modules 2 and 6", S + "m1_faces.png", "One face per source; StyleGAN2 and StyleGAN3-T never appear in training.",
       ("base", "ours", "mod"))
module("module2_preprocessing", 2, "Preprocessing and Augmentation", "label-symmetric: real and fake images get the same degradations",
       "image + label", [("Load image", "256×256 JPEG", "base"),
        ("Label-symmetric\naugmentation", "same for real and fake", "mod"),
        ("Random degradations", "rotation, crop, flip,\nblur, JPEG 50–99", "ours"),
        ("Normalise", "224×224, ImageNet mean/std", "base")],
       "tensor x → Modules 3 and 4", S + "m2_perturb.png", "Examples of the degradations: rotation, scaling, blur, mirroring, JPEG.",
       ("base", "ours", "mod"))
module("module3_spatial", 3, "Spatial Stream", "ResNet-18, as in the base paper", "tensor x",
       [("ResNet-18 backbone", "ImageNet-pretrained CNN", "base"),
        ("Residual blocks", "skip connections,\n64 → 512 channels", "base"),
        ("Global average pooling", "7×7×512 → 512", "base")],
       "512-d embedding → Module 5", S + "m3_gradcam.png", "Grad-CAM: the face regions that raise each model's fake score.",
       ("base",))
module("module4_frequency", 4, "Frequency Stream (DSF-Net v2)", "finds the periodic traces left by GAN upsampling", "tensor x",
       [("Patch split", "grayscale, 49 patches of 32×32", "ours"),
        ("2D FFT", "log-power spectrum", "ours"),
        ("Spectral averaging", "mean over patches → 32×32", "ours"),
        ("Spectral CNN", "3 conv layers + pooling", "ours")],
       "256-d embedding → Module 5", S + "m4_spectra.png", "Mean spectra of a real face and two fakes (input to the spectral CNN).",
       ("ours",))
module("module5_fusion", 5, "Fusion and Classification", "trained end to end with cross-entropy loss and Adam", "512-d + 256-d",
       [("Concatenation", "768-d feature vector", "ours"),
        ("MLP head", "FC 128, ReLU, Dropout 0.3", "ours"),
        ("Logits", "FC 2: z_real, z_fake", "ours"),
        ("Logit margin", "s = z_fake − z_real", "ours")],
       "score s → Module 6", S + "m5_scores.png", "Score distribution on the perturbed test set, with the threshold θ.",
       ("ours",))
module("module6_decision", 6, "FPR-Targeted Decision and Evaluation", "threshold set so only 5% of real faces are flagged", "score s",
       [("Test-time degradation", "base paper's protocol\n(Sec. III-A3)", "base"),
        ("Threshold calibration", "θ at 5% FPR on\nreal validation faces", "ours"),
        ("Decision", "fake if s > θ", "ours"),
        ("Evaluation", "detection at 5% FPR, AUC", "ours")],
       "Real / Fake", S + "m6_results.png", "Fakes caught at 5% FPR, per generator: base paper's method vs proposed system.",
       ("base", "ours"))
