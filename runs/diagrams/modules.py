from PIL import Image
from diagram_lib import Diagram

BW, BH, GAP_X, ROW_GAP, GROUP_PAD = 305, 130, 50, 75, 58
X0, Y0, CW = 60, 170, 1450
COLS = [X0 + 40 + i * (BW + GAP_X) for i in range(4)]


def page(stem, num, name, sub, inputs, rows, output, snap, snap_caption, legend=("base", "ours", "mod", "offline"),
         extra=None, output_from=-1, output_side=False):
    """rows: list of dicts with items [(title, detail, kind)], and optional group (dashed title),
    start column, dir (+1/-1), link (index of the box to link from, or None for no link).
    By default rows snake and link from the previous row's last box."""
    layout, y, col, direction, n = [], Y0 + 110, 0, 1, 0
    for r, row in enumerate(rows):
        if row.get("group"):
            y += GROUP_PAD
        c0 = row.get("start", col)
        dr = row.get("dir", direction)
        cols = [c0 + dr * i for i in range(len(row["items"]))]
        link = row.get("link", n - 1 if r else None)
        layout.append((row, y, cols, link))
        n += len(cols)
        col, direction = cols[-1], -dr
        y += BH + ROW_GAP
    bottom = y - ROW_GAP + 45
    img = Image.open(snap)
    iw = 730
    ih = int(iw * img.height / img.width)
    panel_h = 70 + ih + 30 + 25 * snap_caption.count("\n") + 40
    legend_h = 80 + 46 * len(legend)
    H = max(bottom + 150, Y0 + panel_h + 40 + legend_h + 40)

    d = Diagram(2400, H, f"MODULE {num}: {name.upper()}")
    d.module(X0, Y0, CW, bottom - Y0, f"{num}. {name}", sub)
    boxes = []
    for row, yy, cols, link in layout:
        if row.get("group"):
            gx0 = COLS[min(cols)] - 20
            gx1 = COLS[max(cols)] + BW + 20
            d.module(gx0, yy - GROUP_PAD - 5, gx1 - gx0, BH + GROUP_PAD + 25, row["group"], None, dashed=True, align="left")
        bx = [d.box(COLS[c], yy, BW, BH, t, det, k) for c, (t, det, k) in zip(cols, row["items"])]
        for a, b in zip(bx, bx[1:]):
            d.arrow(a.r, b.l) if b.x > a.x else d.arrow(a.l, b.r)
        if link is not None:
            src = boxes[link]
            d.arrow(src.b, (src.cx, bx[0].y), label=row.get("link_label"), at=(src.cx + 12, bx[0].y - 22), anchor="start")
        boxes += bx

    first = boxes[0]
    if len(inputs) == 1:
        d.text(X0, 118, inputs[0], size=20, anchor="start")
        d.arrow((first.cx, 130), (first.cx, first.y))
    else:
        for x, label in zip((first.x + 70, first.x + BW - 40), inputs):
            d.text(x, 118, label, size=20)
            d.arrow((x, 130), (x, first.y))
    last = boxes[output_from]
    if output_side:
        d.arrow(last.r, (1535, last.cy), (1535, bottom + 80))
        d.text(1520, bottom + 70, output, size=20, anchor="end", italic=True)
    else:
        d.arrow(last.b, (last.cx, bottom + 80))
        d.text(last.cx + 16, bottom + 70, output, size=20, anchor="start", italic=True)

    px = 1590
    d.module(px, Y0, 770, panel_h, "Snapshot", None, fill="#fafafa")
    d.image(px + 20, Y0 + 70, iw, ih, snap)
    d.text(px + 385, Y0 + 70 + ih + 34, snap_caption, size=18, italic=True)
    d.legend(px, Y0 + panel_h + 40, 770, legend)
    if extra:
        extra(d, boxes)
    d.save(stem)


S = "snapshots/"

page("module1_data", 1, "Data Acquisition", "real faces, training fakes, and fakes kept for testing only",
     ["Kaggle 140k faces + NVIDIA pretrained generators (ψ = 0.7)"],
     [{"items": [("Load the Kaggle faces", "70k FFHQ real, 70k StyleGAN\nfake, 256×256", "base"),
        ("Generate StyleGAN3-R", "5,000 training faces,\nseeds 0–4999", "base"),
        ("Generate held-out\nStyleGAN3-R", "1,000 faces, new seeds", "ours"),
        ("Generate unseen fakes", "1,000 StyleGAN3-T,\n1,000 StyleGAN2", "ours")]},
      {"items": [("Save as JPEG quality 95", "same format for every source", "ours"),
        ("Original manifest", "Kaggle training split", "base"),
        ("Augmented manifest", "+ 5,000 StyleGAN3-R", "base"),
        ("Paper-composition\nmanifest", "4,000 clean + 1,000 degraded\nStyleGAN3 (Fig. 3)", "mod")]},
      {"items": [("Test and validation sets", "test: 10,000 real + 13,000 fake;\nvalidation: 10,000 real", "ours")]}],
     "manifests → Modules 2 and 6",
     S + "m1_faces.png", "One face per source. StyleGAN2 and StyleGAN3-T\nfaces are never used in training.",
     legend=("base", "ours", "mod"))


def m2_extra(d, b):
    norm = b[-1]
    test = d.box(COLS[1], norm.y, BW, BH, "Test images", "no augmentation:\nresize to 224×224 only", "base")
    d.arrow(test.l, norm.r)


page("module2_preprocessing", 2, "Preprocessing and Augmentation", "label-symmetric: every training image gets the same random degradations",
     ["Manifest entry: image path + label (0 = real, 1 = fake)"],
     [{"items": [("Read the JPEG", "256×256 RGB", "base"),
        ("Real and fake alike", "base paper degraded only\n1,000 fakes: 'degraded = fake'", "mod"),
        ("Random rotation", "up to ±180°, p = 0.5", "ours"),
        ("Random resized crop", "to 224×224,\nkeeping 50–100%", "ours")]},
      {"items": [("Random flips", "horizontal p = 0.5,\nvertical p = 0.2", "ours"),
        ("Down-up scaling", "×0.5 and back, p = 0.3", "ours"),
        ("Gaussian blur", "kernel 13, σ 0.1–2.3,\np = 0.3", "ours"),
        ("JPEG recompression", "quality 50–99, p = 0.5", "ours")]},
      {"items": [("Tensor and normalise", "ImageNet mean and std", "base")]}],
     "tensor x, 3×224×224 → Modules 3 and 4",
     S + "m2_perturb.png", "Degradations of the kind applied here, shown on real faces:\nrotation, scaling ×0.5 and ×2, blur, mirroring, JPEG.",
     legend=("base", "ours", "mod"), extra=m2_extra)

page("module3_spatial", 3, "Spatial Stream", "ResNet-18, as in the base paper",
     ["Tensor x, 3×224×224"],
     [{"items": [("Stem", "Conv 7×7, 64 ch, stride 2;\nmax-pool → 56×56", "base"),
        ("Stage 1", "2 residual blocks,\n64 ch, 56×56", "base"),
        ("Stage 2", "2 residual blocks,\n128 ch, 28×28", "base"),
        ("Stage 3", "2 residual blocks,\n256 ch, 14×14", "base")]},
      {"items": [("Stage 4", "2 residual blocks,\n512 ch, 7×7", "base"),
        ("Global average\npooling", "7×7×512 → 512", "base"),
        ("Drop the ImageNet\nclassifier", "fusion head replaces it", "mod"),
        ("Spatial embedding", "512-d vector", "base")]}],
     "512-d → Module 5",
     S + "m3_gradcam.png", "Grad-CAM from Stage 4: the face regions that\nraise each model's fake score.",
     legend=("base", "mod"))

page("module4_frequency", 4, "Frequency Stream (DSF-Net v2)", "new: keeps the periodic traces GAN upsampling leaves in every patch",
     ["Tensor x, 3×224×224"],
     [{"items": [("Grayscale", "mean of R, G, B\n→ 1×224×224", "ours"),
        ("Split into patches", "7×7 grid =\n49 patches of 32×32", "ours"),
        ("2D FFT per patch", "32×32 complex spectrum", "ours"),
        ("Log-power", "log(1 + |F|²)", "ours")]},
      {"items": [("fftshift", "zero frequency\nto the centre", "ours"),
        ("Average the\n49 spectra", "traces repeat and stay;\nface content averages out", "ours"),
        ("BatchNorm", "log-power is ≈ 10–20", "ours"),
        ("Conv block 1", "3×3, 32 ch, ReLU,\nmax-pool → 16×16", "ours")]},
      {"items": [("Conv block 2", "3×3, 64 ch, ReLU,\nmax-pool → 8×8", "ours"),
        ("Conv block 3", "3×3, 256 ch, ReLU", "ours"),
        ("Global average\npooling", "8×8×256 → 256", "ours"),
        ("Spectral embedding", "256-d vector", "ours")]}],
     "256-d → Module 5",
     S + "m4_spectra.png", "Mean 32×32 spectra the CNN sees. The faint grid of dots\noff the central cross in the StyleGAN spectrum is an\nupsampling trace; the cross itself appears in every image.",
     legend=("ours",))

page("module5_fusion", 5, "Fusion and Classification", "both streams and this head are trained together, end to end",
     ["512-d (Module 3)", "256-d (Module 4)"],
     [{"items": [("Concatenate", "512 + 256 = 768-d", "ours"),
        ("FC 768 → 128", "ReLU, then Dropout 0.3\n(training only)", "ours"),
        ("FC 128 → 2", "logits z_real, z_fake", "ours"),
        ("Score s", "s = z_fake − z_real;\ndoes not saturate like softmax", "ours")]},
      {"items": [("Cross-entropy loss", "softmax vs real/fake label", "base"),
        ("Backpropagation", "through the head and\nboth streams", "ours"),
        ("Adam update", "learning rate 1e-4, batch 64,\n10 epochs", "base")],
       "group": "Training (offline)", "start": 2, "dir": -1, "link": 2},
      ],
     "score s → Module 6",
     S + "m5_scores.png", "Scores of the proposed system on the 23,000\nperturbed test images, with the calibrated threshold θ.",
     legend=("base", "ours", "offline"), output_from=3, output_side=True)


def module6():
    from diagram_lib import Box
    W2 = 2 * BW + GAP_X
    rows_y = [Y0 + 110 + GROUP_PAD, None, None, None]
    yA = Y0 + 110 + GROUP_PAD
    yB = yA + BH + ROW_GAP + GROUP_PAD
    yC = yB + BH + ROW_GAP + 10
    yD = yC + BH + ROW_GAP + GROUP_PAD
    bottom = yD + BH + 45
    img = Image.open(S + "m6_results.png")
    iw = 730
    ih = int(iw * img.height / img.width)
    cap = "Fakes caught when the threshold flags 5% of real faces,\nby generator, on the 23,000 perturbed test images."
    panel_h = 70 + ih + 30 + 25 + 40
    d = Diagram(2400, bottom + 80, "MODULE 6: FPR-TARGETED DECISION AND EVALUATION")
    d.module(X0, Y0, CW, bottom - Y0, "6. FPR-Targeted Decision and Evaluation", "threshold θ chosen so only 5% of real faces are flagged")

    def group(y, c0, c1, title, dx=25):
        d.module(COLS[c0] - 20, y - GROUP_PAD - 5, COLS[c1] + BW + 20 - (COLS[c0] - 20), BH + GROUP_PAD + 25, title, None,
                 dashed=True, align="left", title_dx=dx)

    group(yA, 0, 3, "Test protocol, from the base paper (Sec. III-A3, offline)", dx=215)
    p1 = d.box(COLS[0], yA, BW, BH, "Pick one\ndegradation", "random, seeded per image", "base")
    p2 = d.box(COLS[1], yA, W2, BH, "Apply it", "rotation 45°–315° in 45° steps, scale ×0.5 or ×2,\nblur kernel 3, 9 or 13, mirror, JPEG quality 50–99", "base")
    p3 = d.box(COLS[3], yA, BW, BH, "Perturbed sets", "23,000 test images;\n10,000 real validation", "mod")
    d.arrow(p1.r, p2.l); d.arrow(p2.r, p3.l)

    group(yB, 0, 3, "Threshold calibration (offline)")
    c1 = d.box(COLS[3], yB, BW, BH, "Score validation\nfaces", "10,000 perturbed real", "ours")
    c2 = d.box(COLS[2], yB, BW, BH, "Sort the scores", "with the trained model", "ours")
    c3 = d.box(COLS[1], yB, BW, BH, "95th percentile", "5% of real faces\nscore above it", "ours")
    c4 = d.box(COLS[0], yB, BW, BH, "Threshold θ", "+0.767 for the\nproposed system", "ours")
    d.arrow(p3.b, c1.t, label="validation", at=(p3.cx + 12, yB - 30), anchor="start")
    d.arrow(c1.l, c2.r); d.arrow(c2.l, c3.r); d.arrow(c3.l, c4.r)

    d1 = d.box(COLS[0], yC, BW, BH, "Compare s with θ", "fake if s > θ", "ours")
    d.arrow((c4.cx - 60, c4.y + BH), (c4.cx - 60, yC), label="θ", at=(c4.cx - 50, yC - 25), anchor="start")
    gy = yC - 32
    d.arrow((1500, 130), (1500, gy), (d1.cx + 60, gy), (d1.cx + 60, yC))
    d.text(1510, 118, "Score s (Module 5)", size=20, anchor="end")
    pill = d.pill(COLS[1] + 30, yC + BH / 2 - 35, 260, 70, "Real / Fake")
    d.arrow(d1.r, pill.l)

    group(yD, 0, 3, "Evaluation (offline)", dx=215)
    e1 = d.box(COLS[0], yD, BW, BH, "Decide every\ntest image", "23,000 perturbed images", "ours")
    e2 = d.box(COLS[1], yD, BW, BH, "Overall metrics", "detection at 5% FPR,\nAUC, accuracy, F1", "ours")
    e3 = d.box(COLS[2], yD, BW, BH, "Breakdowns", "per generator and\nper perturbation", "ours")
    e4 = d.box(COLS[3], yD, BW, BH, "McNemar test", "proposed vs\naugmentation only", "ours")
    d.arrow(d1.b, e1.t, label="test set", at=(d1.cx + 12, yC + BH + 38), anchor="start")
    d.arrow(e1.r, e2.l); d.arrow(e2.r, e3.l); d.arrow(e3.r, e4.l)

    d.text(X0, 118, "Test and validation sets (Module 1)", size=20, anchor="start")
    d.arrow((p1.cx, 130), (p1.cx, p1.y))

    px = 1590
    d.module(px, Y0, 770, panel_h, "Snapshot", None, fill="#fafafa")
    d.image(px + 20, Y0 + 70, iw, ih, S + "m6_results.png")
    d.text(px + 385, Y0 + 70 + ih + 34, cap, size=18, italic=True)
    d.legend(px, Y0 + panel_h + 40, 770, ("base", "ours", "mod", "offline"))
    d.save("module6_decision")


module6()
