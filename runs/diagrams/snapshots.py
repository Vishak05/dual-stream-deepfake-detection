"""Snapshot images embedded in the module diagrams."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = "runs/diagrams/snapshots/"
BASE, OURS = "#5b84b1", "#e07b39"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#888", "axes.labelcolor": "#222",
                     "xtick.color": "#444", "ytick.color": "#444"})

# Module 1: one face per source
faces = [("demo/examples/1_FFHQ_00001.jpg", "Real (FFHQ)"), ("demo/examples/4_StyleGAN_00MZYXAT77.jpg", "StyleGAN"),
         ("demo/examples/5_StyleGAN2_stylegan3_200005.jpg", "StyleGAN2 (test only)"),
         ("demo/examples/6_StyleGAN3-t_stylegan3_200082.jpg", "StyleGAN3-T (test only)")]
fig, ax = plt.subplots(1, 4, figsize=(10, 2.9))
for a, (p, t) in zip(ax, faces):
    a.imshow(Image.open(p).convert("RGB").resize((256, 256))); a.set_title(t, fontsize=12); a.axis("off")
plt.tight_layout(); plt.savefig(OUT + "m1_faces.png", dpi=130); plt.close()

# Module 3: Grad-CAM crop (rows 1-2, first three columns of the explanation grid)
Image.open("runs/explanations_v2_grid.png").crop((0, 18, 893, 632)).save(OUT + "m3_gradcam.png")

# Module 5: score distribution of the proposed system on the perturbed test set
z = np.load("runs/scores_perturbed.npz", allow_pickle=True)
s, y = z["score_dsfnet_v2_robust"], z["y"]
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.linspace(-20, 25, 70)
ax.hist(s[y == 0], bins=bins, color=BASE, alpha=0.75, label="Real faces (10,000)", edgecolor="white", linewidth=0.5)
ax.hist(s[y == 1], bins=bins, color=OURS, alpha=0.75, label="Fake faces (13,000)", edgecolor="white", linewidth=0.5)
ax.axvline(0.767, color="#222", lw=2)
ax.text(1.3, ax.get_ylim()[1] * 0.92, "θ = +0.767", fontsize=11, color="#222")
ax.set_xlabel("Score s = z_fake − z_real"); ax.set_ylabel("Images")
ax.legend(frameon=False, loc="upper right", fontsize=10.5)
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT + "m5_scores.png", dpi=130); plt.close()

# Module 6: detection at 5% FPR per generator
gens = ["StyleGAN\n(in training)", "StyleGAN3-R\n(held out)", "StyleGAN3-T\n(unseen)", "StyleGAN2\n(unseen)", "All fakes"]
base = [52.4, 84.7, 83.4, 20.2, 54.8]
ours = [98.6, 95.2, 85.7, 83.0, 96.2]
x = np.arange(len(gens)); w = 0.38
fig, ax = plt.subplots(figsize=(7.4, 3.6))
b1 = ax.bar(x - w / 2 - 0.01, base, w, color=BASE, label="Base paper's method")
b2 = ax.bar(x + w / 2 + 0.01, ours, w, color=OURS, label="Proposed system")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.1f}", ha="center", fontsize=9.5, color="#222")
ax.set_xticks(x, gens, fontsize=10); ax.set_ylim(0, 112); ax.set_ylabel("Fakes caught at 5% FPR (%)")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=10.5)
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT + "m6_results.png", dpi=130); plt.close()
