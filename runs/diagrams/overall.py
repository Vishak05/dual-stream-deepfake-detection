from diagram_lib import Diagram

d = Diagram(2400, 1650, "ARCHITECTURE DIAGRAM")

# Inputs
d.text(490, 112, "Kaggle 140k Real and Fake Faces", size=21)
d.text(30, 88, "Pretrained GAN\ncheckpoints", size=21, anchor="start")

# 1. Data acquisition
d.module(60, 170, 500, 790, "1. Data Acquisition", "Kaggle faces and generated fakes")
X, W, H = 95, 430, 104
b11 = d.box(X, 265, W, H, "Load the Kaggle faces", "FFHQ real, StyleGAN fake, 256×256", "base")
b12 = d.box(X, 400, W, H, "Generate StyleGAN3-R faces", "5,000 for training, ψ = 0.7", "base")
b13 = d.box(X, 535, W, H, "Generate test-only fakes", "1,000 each: StyleGAN3-R (held out),\nStyleGAN3-T, StyleGAN2", "ours")
b14 = d.box(X, 670, W, H, "Save all as JPEG quality 95", "file format gives no clue", "ours")
b15 = d.box(X, 805, W, H, "Build the manifests", "train; test (23,000);\nvalidation (10,000 real)", "mod")
d.arrow((490, 125), (490, 265))
d.arrow((45, 140), (45, b13.cy), b13.l)
d.arrow((45, b12.cy), b12.l)
d.dot(45, b12.cy)
for a, b in ((b11, b12), (b12, b13), (b13, b14), (b14, b15)):
    d.arrow(a.b, b.t)

# 2. Preprocessing and augmentation
d.module(640, 170, 520, 790, "2. Preprocessing and Augmentation", "same degradations for real and fake")
X = 685
b21 = d.box(X, 265, W, H, "Read image and label", "from the training manifest", "base")
b22 = d.box(X, 400, W, H, "Degrade every training image", "base paper: 1,000 fakes only;\nours: real and fake alike", "mod")
b23 = d.box(X, 535, W, H, "Geometric changes", "rotation ±180°, crop 50–100%,\nhorizontal and vertical flips", "ours")
b24 = d.box(X, 670, W, H, "Quality changes", "blur (kernel 13), ×0.5 down-up,\nJPEG quality 50–99", "ours")
b25 = d.box(X, 805, W, H, "Resize and normalise", "224×224, ImageNet mean and std;\ntest images get this step only", "base")
for a, b in ((b21, b22), (b22, b23), (b23, b24), (b24, b25)):
    d.arrow(a.b, b.t)
d.arrow(b15.r, (600, b15.cy), (600, b21.cy), b21.l)

# 3. Spatial stream
d.module(1240, 170, 1130, 230, "3. Spatial Stream", "the base paper's model, unchanged")
xs = [1275, 1555, 1835, 2115]
BW, Y3 = 225, 268
b31 = d.box(xs[0], Y3, BW, H, "ResNet-18 stem", "ImageNet-pretrained", "base")
b32 = d.box(xs[1], Y3, BW, H, "Residual stages 1–4", "64 → 512 channels", "base")
b33 = d.box(xs[2], Y3, BW, H, "Global average\npooling", "classifier removed", "base")
b34 = d.box(xs[3], Y3, BW, H, "Spatial\nembedding", "512-d", "base")
for a, b in ((b31, b32), (b32, b33), (b33, b34)):
    d.arrow(a.r, b.l)

# 4. Frequency stream
d.module(1240, 430, 1130, 370, "4. Frequency Stream (DSF-Net v2)", "keeps the periodic traces GAN upsampling leaves")
Y4a, Y4b = 528, 668
b41 = d.box(xs[0], Y4a, BW, H, "Grayscale and\nsplit", "49 patches of 32×32", "ours")
b42 = d.box(xs[1], Y4a, BW, H, "2D FFT per patch", "log(1 + |F|²), fftshift", "ours")
b43 = d.box(xs[2], Y4a, BW, H, "Average the\n49 spectra", "one 32×32 spectrum", "ours")
b44 = d.box(xs[3], Y4a, BW, H, "BatchNorm", "log-power is ≈ 10–20", "ours")
b45 = d.box(xs[3], Y4b, BW, H, "Spectral CNN", "Conv 32, 64, 256;\nReLU, max-pool", "ours")
b46 = d.box(xs[2], Y4b, BW, H, "Global average\npooling", None, "ours")
b47 = d.box(xs[0], Y4b, BW + 280, H, "Spectral embedding", "256-d", "ours")
for a, b in ((b41, b42), (b42, b43), (b43, b44)):
    d.arrow(a.r, b.l)
d.arrow(b44.b, b45.t)
d.arrow(b45.l, b46.r)
d.arrow(b46.l, b47.r)

# Module 2 -> streams
d.arrow(b25.r, (1200, b25.cy), (1200, b31.cy), b31.l)
d.arrow((1200, b41.cy), b41.l)
d.dot(1200, b41.cy)

# 5. Fusion
d.module(1240, 840, 1130, 230, "5. Fusion and Classification", "trained end to end: Adam 1e-4, batch 64, cross-entropy, 10 epochs")
Y5 = 940
b51 = d.box(xs[0], Y5, BW, H, "Concatenate", "512 + 256 = 768-d", "ours")
b52 = d.box(xs[1], Y5, BW, H, "FC 768 → 128", "ReLU, Dropout 0.3", "ours")
b53 = d.box(xs[2], Y5, BW, H, "FC 128 → 2", "logits z_real, z_fake", "ours")
b54 = d.box(xs[3], Y5, BW, H, "Score s", "s = z_fake − z_real", "ours")
for a, b in ((b51, b52), (b52, b53), (b53, b54)):
    d.arrow(a.r, b.l)
d.arrow(b34.r, (2400, b34.cy), (2400, 820), (1455, 820), (1455, 840 + 1), label="512-d", at=(2340, 812))
d.arrow((1455, 841), (1455, Y5))
d.arrow((1340, b47.y + H), (1340, Y5), label="256-d", at=(1300, 828))

# 6. Decision and evaluation
d.module(60, 1110, 1480, 510, "6. FPR-Targeted Decision and Evaluation", "threshold θ set so only 5% of real faces are flagged")
d.module(95, 1200, 905, 190, "Threshold calibration (offline)", None, dashed=True, align="left")
d.module(95, 1410, 905, 190, "Evaluation (offline)", None, dashed=True, align="left")
CW = 255
c1 = d.box(120, 1265, CW, H, "Perturb real\nvalidation faces", "one random degradation", "mod")
c2 = d.box(415, 1265, CW, H, "Score them", "with the trained model", "ours")
c3 = d.box(710, 1265, CW, H, "95th percentile", "θ = +0.767", "ours")
e1 = d.box(120, 1475, CW, H, "Perturb the\ntest set", "Sec. III-A3 protocol", "base")
e2 = d.box(415, 1475, CW, H, "Decide at θ", "23,000 images", "ours")
e3 = d.box(710, 1475, CW, H, "Report metrics", "detection at 5% FPR,\nAUC, McNemar", "ours")
f1 = d.box(1170, 1265, 330, H, "Compare with θ", "fake if s > θ", "ours")
for a, b in ((c1, c2), (c2, c3), (e1, e2), (e2, e3)):
    d.arrow(a.r, b.l)
d.arrow(c3.r, f1.l, label="θ", at=(1085, f1.cy - 10))
d.arrow((1040, f1.cy), (1040, 1400), (e2.cx + 40, 1400), (e2.cx + 40, 1475))
d.dot(1040, f1.cy)
d.arrow((310, 960 - 51), (310, 1080), (75, 1080), (75, c1.cy), c1.l, label="test (23,000) and validation (10,000 real) sets", at=(325, 1072), anchor="start")
d.arrow((75, 1080), (40, 1080), (40, e1.cy), e1.l)
d.dot(75, 1080)
d.arrow(b54.b, (b54.cx, 1090), (f1.cx, 1090), f1.t, label="score s", at=(b54.cx + 12, 1110), anchor="start")
out = d.pill(1600, 1430, 230, 64, "Real / Fake")
d.arrow(f1.b, (f1.cx, out.cy), out.l)

d.legend(1850, 1140, 520)
d.save("architecture", scale=1)
