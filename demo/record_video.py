"""Records a captioned, voice-free demo video of the running demo app.

Start the app first (python demo/app.py), then:

    python demo/record_video.py --out runs/DSF-Net_Demo.mp4

Title, method and results cards are rendered as HTML in the same headless
browser, then the script clicks through the app with a visible cursor and
on-screen captions. Frames come from Chrome's screencast (sharper than
Playwright's built-in recorder) and are joined into an H.264 MP4 with the
ffmpeg binary bundled in imageio-ffmpeg."""
import argparse
import base64
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

W, H = 1920, 1080
APP_URL = "http://localhost:7860"

CARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { box-sizing: border-box; margin: 0; }
body { width: 1920px; height: 1080px; background: #0f1b2d; color: #e8eef7;
       font-family: Inter, 'Segoe UI', sans-serif; display: flex; flex-direction: column;
       justify-content: center; padding: 0 140px; }
.kicker { color: #f97316; font-weight: 600; font-size: 30px; letter-spacing: .08em; text-transform: uppercase; }
h1 { font-size: 76px; font-weight: 800; line-height: 1.1; margin: 22px 0 28px; }
h2 { font-size: 54px; font-weight: 800; margin: 14px 0 40px; }
p, li { font-size: 34px; line-height: 1.5; color: #c6d2e3; }
li { margin-bottom: 18px; }
b { color: #fff; }
.ours { color: #2dd4bf; } .base { color: #fb923c; }
.muted { color: #8395ad; font-size: 26px; }
.row { display: flex; align-items: center; gap: 26px; }
.box { border: 3px solid #36507a; border-radius: 18px; padding: 22px 28px; font-size: 28px;
       background: #16263e; text-align: center; line-height: 1.35; }
.box.t { border-color: #2dd4bf; } .box.o { border-color: #f97316; }
.arrow { font-size: 46px; color: #8395ad; }
table { border-collapse: collapse; font-size: 32px; width: 100%; margin-top: 10px; }
th, td { padding: 16px 22px; text-align: left; border-bottom: 2px solid #26395a; }
th { color: #8395ad; font-weight: 600; font-size: 26px; text-transform: uppercase; letter-spacing: .05em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.bars { display: grid; grid-template-columns: 200px 1fr; gap: 10px 22px; align-items: center; font-size: 26px; }
.bar { height: 26px; border-radius: 6px; display: flex; align-items: center; padding-left: 10px;
       font-size: 20px; font-weight: 600; color: #0f1b2d; white-space: nowrap; }
"""

CAPTION_JS = """
(() => {
  if (document.getElementById('demo-caption')) return;
  const style = document.createElement('style');
  style.textContent = `
    #demo-caption { position: fixed; left: 0; right: 0; top: 0; z-index: 99999; height: 96px;
      background: #0f1b2d; color: #fff; font: 600 34px Inter, 'Segoe UI', sans-serif;
      display: flex; align-items: center; padding: 0 64px; gap: 28px; font-size: 31px; }
    #demo-caption .step { color: #f97316; font-size: 26px; white-space: nowrap; text-transform: uppercase; letter-spacing: .08em; }
    .gradio-container { zoom: 0.84; margin-top: 110px !important; }
    #demo-cursor { position: fixed; z-index: 100000; width: 30px; height: 30px; pointer-events: none;
      left: -50px; top: -50px; }
    .demo-hl { outline: 6px solid #f97316 !important; outline-offset: 6px; border-radius: 10px;
      transition: outline .2s; }
  `;
  document.head.appendChild(style);
  const cap = document.createElement('div');
  cap.id = 'demo-caption';
  document.body.appendChild(cap);
  const cur = document.createElement('div');
  cur.id = 'demo-cursor';
  cur.innerHTML = `<svg viewBox="0 0 24 24" width="30" height="30"><path d="M3 2l7 19 2.5-7.5L20 11z"
    fill="#fff" stroke="#000" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
  document.body.appendChild(cur);
  document.addEventListener('mousemove', e => { cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; });
})();
"""


def card(body: str) -> str:
    return f"<html><head><style>{CARD_CSS}</style></head><body>{body}</body></html>"


def bars(rows: list[tuple[str, float, float]]) -> str:
    out = []
    for label, base, ours in rows:
        out.append(
            f"<div>{label}</div><div>"
            f"<div class='bar' style='width:{max(base, 12) * 6.5}px;background:#fb923c'>{base:.1f}%</div>"
            f"<div class='bar' style='width:{max(ours, 12) * 6.5}px;background:#2dd4bf;margin-top:6px'>{ours:.1f}%</div>"
            f"</div>"
        )
    return "<div class='bars'>" + "".join(out) + "</div>"


TITLE = card("""
<div class='kicker'>CE23E02 Artificial Intelligence · Mini Project Demo</div>
<h1>DSF-Net: Robust Deepfake Face Detection<br>beyond StyleGAN3 Augmentation</h1>
<p>Builds on Hafezi et al., <i>“Enhancing Deepfake Detection: Leveraging StyleGAN3…”</i>, IEEE Access 2026</p>
""")

PROBLEM = card("""
<div class='kicker'>The problem</div>
<h2>The base paper trades accuracy for false alarms</h2>
<ul>
<li>It adds StyleGAN3 faces and <b>degraded copies of fakes only</b> to training.</li>
<li>Accuracy rises, but <b class='base'>real faces get flagged as fake</b> more often (FPR 5.9% → 12.4% in the paper).</li>
<li>Our finding: the model learns a shortcut, <b>“degraded ⇒ fake”</b>.
    Our reproduction flags <b class='base'>79% of rotated</b> and <b class='base'>51% of mirrored</b> real faces.</li>
</ul>
""")

METHOD = card("""
<div class='kicker'>Proposed method</div>
<h2>Three fixes on top of the base paper</h2>
<div class='row' style='margin-bottom:46px'>
  <div class='box'>Face image<br><span class='muted'>224×224</span></div>
  <div class='arrow'>→</div>
  <div style='display:flex;flex-direction:column;gap:18px'>
    <div class='box o'>Spatial stream · ResNet-18<br><span class='muted'>512-d</span></div>
    <div class='box t'>Frequency stream · patch FFT → mean 32×32 spectrum → CNN<br><span class='muted'>256-d (DSF-Net v2)</span></div>
  </div>
  <div class='arrow'>→</div>
  <div class='box'>Fusion<br><span class='muted'>768 → 128 → 2</span></div>
  <div class='arrow'>→</div>
  <div class='box t'>Real / Fake<br><span class='muted'>threshold @ 5% FPR</span></div>
</div>
<ol>
<li><b class='ours'>Label-symmetric augmentation</b>: degrade real <i>and</i> fake training images alike, removing the shortcut.</li>
<li><b class='ours'>Spectrum-preserving frequency stream</b>: keeps the periodic GAN upsampling peaks.</li>
<li><b class='ours'>FPR-targeted calibration</b> on logit margins: the threshold is set for 5% false alarms on real faces.</li>
</ol>
""")

RESULTS = card(f"""
<div style='margin:0 -60px'><div class='kicker'>Results · base paper's own test protocol · same data for both</div>
<div class='row' style='align-items:flex-start;gap:90px;margin-top:26px'>
<div style='flex:0 0 900px'>
<table>
<tr><th>Metric</th><th style='text-align:right'>Base paper's method</th><th style='text-align:right'>Ours</th></tr>
<tr><td>Detection at 5% FPR</td><td class='num base'>54.8%</td><td class='num ours'>96.2%</td></tr>
<tr><td>AUC</td><td class='num base'>0.920</td><td class='num ours'>0.992</td></tr>
<tr><td>Accuracy (threshold 0.5)</td><td class='num base'>84.4%</td><td class='num ours'>95.5%</td></tr>
<tr><td>False-positive rate (0.5)</td><td class='num base'>30.0%</td><td class='num ours'>6.9%</td></tr>
<tr><td>Unseen StyleGAN2 detection</td><td class='num base'>20.2%</td><td class='num ours'>83.0%</td></tr>
</table>
<p class='muted' style='margin-top:22px'>23,000 perturbed test images · paired McNemar tests significant · one training run per model</p>
</div>
<div style='flex:1'>
<p style='margin-bottom:20px'><b>Real faces flagged as fake</b>, by perturbation<br>
<span class='muted'><span class='base'>■</span> base paper's method &nbsp; <span class='ours'>■</span> ours &nbsp; (threshold 0.5)</span></p>
{bars([("Rotation", 79.2, 5.8), ("Mirroring", 50.6, 5.0), ("JPEG", 11.8, 10.1), ("Scaling", 5.8, 6.6), ("Gaussian blur", 3.1, 6.9)])}
</div>
</div></div>
""")

CONCLUSION = card("""
<div class='kicker'>Conclusion</div>
<h2>Fewer false alarms, better generalisation</h2>
<ul>
<li>The base paper's false-positive problem comes from a <b>degradation shortcut</b>; symmetric augmentation removes it.</li>
<li>Detection at 5% FPR: <b class='base'>54.8%</b> → <b class='ours'>96.2%</b>; unseen StyleGAN2: <b class='base'>20.2%</b> → <b class='ours'>83.0%</b>.</li>
<li>Limitations: one seed, 10 epochs; clean StyleGAN3-T detection drops (99.4% → 90.8%);
    most of the gain comes from the augmentation fix.</li>
</ul>
<p class='muted' style='margin-top:30px'>Code: github.com/Vishak05/dual-stream-deepfake-detection</p>
""")


class Recorder:
    """Collects Chrome screencast frames with their timestamps."""

    def __init__(self, page, frame_dir: Path):
        self.frames: list[tuple[float, Path]] = []
        self.frame_dir = frame_dir
        self.session = page.context.new_cdp_session(page)
        self.session.on("Page.screencastFrame", self._on_frame)

    def _on_frame(self, event) -> None:
        path = self.frame_dir / f"{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(event["data"]))
        self.frames.append((time.monotonic(), path))
        self.session.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    def start(self) -> None:
        self.session.send("Page.startScreencast",
                          {"format": "jpeg", "quality": 92, "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})

    def stop(self) -> None:
        self.session.send("Page.stopScreencast")
        self.end = time.monotonic()


class Demo:
    def __init__(self, page, app_only: bool = False):
        self.page = page
        self.app_only = app_only   # skip the title/method/results cards; keep the captioned app walkthrough

    def hold(self, seconds: float) -> None:
        self.page.wait_for_timeout(seconds * 1000)

    def show_card(self, html: str, seconds: float) -> None:
        if self.app_only:
            return
        self.page.set_content(html, wait_until="networkidle")
        self.hold(seconds)

    def caption(self, step: str, text: str) -> None:
        self.page.evaluate(
            "([s, t]) => { document.getElementById('demo-caption').innerHTML ="
            " `<span class='step'>${s}</span><span>${t}</span>`; }", [step, text])

    def point_at(self, locator) -> None:
        box = locator.bounding_box()
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=30)

    def click(self, locator) -> None:
        self.point_at(locator)
        self.hold(0.4)
        locator.click()

    def block(self, label: str):
        """The Gradio block (panel) containing the given label text."""
        return self.page.get_by_text(label, exact=False).first.locator(
            "xpath=ancestor::div[contains(concat(' ', normalize-space(@class), ' '), ' block ')][1]")

    def highlight(self, *labels: str) -> None:
        self.page.evaluate("() => document.querySelectorAll('.demo-hl').forEach(e => e.classList.remove('demo-hl'))")
        for label in labels:
            self.block(label).evaluate("e => e.classList.add('demo-hl')")

    def wait_for_result(self, previous: str) -> None:
        self.page.wait_for_function(
            "prev => { const t = document.querySelector('.gradio-container').innerText;"
            " return t.includes('Decision threshold') && t !== prev; }",
            arg=previous, timeout=20000)
        self.hold(1.0)   # let the images finish painting

    def page_text(self) -> str:
        return self.page.evaluate("() => document.querySelector('.gradio-container').innerText")

    def run_example(self, button_text: str) -> None:
        before = self.page_text()
        self.click(self.page.get_by_text(button_text, exact=True))
        self.wait_for_result(before)


BASE_VERDICT = "Base paper's method (ResNet-18"
OURS_VERDICT = "Proposed: DSF-Net v2"
BASE_CAM = "Grad-CAM: where the base model looks"
OURS_CAM = "Grad-CAM: where DSF-Net v2 looks"
SPECTRUM = "DSF-Net v2 frequency input"
SEEN = "Image the models see"


def script(demo: Demo) -> None:
    page = demo.page
    demo.show_card(TITLE, 6)
    demo.show_card(PROBLEM, 11)
    demo.show_card(METHOD, 13)

    page.goto(APP_URL, wait_until="networkidle")
    page.evaluate(CAPTION_JS)
    page.mouse.move(W / 2, H / 2)
    demo.caption("Live demo", "Left: base paper's method. Right: proposed system. Both trained on the same data")
    demo.hold(4)

    demo.caption("1 · Real face", "Real face from the held-out test set, no perturbation")
    demo.run_example("Real face, clean")
    demo.highlight(BASE_VERDICT, OURS_VERDICT)
    demo.caption("1 · Real face", "Base paper's method: REAL (score −8.65). Proposed system: REAL (score −5.99)")
    demo.hold(5)

    demo.caption("2 · Same face, rotated", "Perturbation from the base paper's test protocol: rotation by 135°")
    demo.highlight()
    dropdown = page.get_by_label("Perturbation (base paper's test protocol)")
    demo.click(dropdown)
    demo.hold(0.6)
    demo.click(page.get_by_role("option", name="Rotation 135°"))
    demo.hold(0.6)
    before = demo.page_text()
    demo.click(page.get_by_role("button", name="Detect", exact=True))
    demo.wait_for_result(before)
    demo.highlight(SEEN, BASE_VERDICT)
    demo.caption("2 · Same face, rotated", "Base paper's method: FAKE (score +2.38, threshold 0.00). The image is a real face")
    demo.hold(5)
    demo.highlight(OURS_VERDICT)
    demo.caption("2 · Same face, rotated", "Proposed system: REAL (score −3.22, threshold +0.77)")
    demo.hold(5)
    demo.highlight(BASE_CAM, OURS_CAM)
    demo.caption("2 · Same face, rotated", "Grad-CAM: image regions that contribute to each model's fake score")
    demo.hold(4)

    demo.highlight()
    demo.caption("3 · Same face, mirrored", "Same real face, mirrored vertically")
    demo.run_example("Real face, mirrored vertically")
    demo.highlight(BASE_VERDICT, OURS_VERDICT)
    demo.caption("3 · Same face, mirrored", "Base paper's method: FAKE (score +11.43). Proposed system: REAL (score −6.14)")
    demo.hold(6)

    demo.highlight()
    demo.caption("4 · StyleGAN fake, blurred", "StyleGAN-generated face, Gaussian blur with a 9×9 kernel")
    demo.run_example("StyleGAN fake, blurred (kernel 9)")
    demo.highlight(BASE_VERDICT, OURS_VERDICT)
    demo.caption("4 · StyleGAN fake, blurred", "Base paper's method: REAL (score −0.02). Proposed system: FAKE (score +6.72)")
    demo.hold(6)

    demo.highlight()
    demo.caption("5 · Unseen generator", "StyleGAN2-generated face. StyleGAN2 is not in either model's training data")
    demo.run_example("StyleGAN2 fake (not in training data)")
    demo.highlight(BASE_VERDICT, OURS_VERDICT)
    demo.caption("5 · Unseen generator", "Base paper's method: REAL (score −1.22). Proposed system: FAKE (score +5.44)")
    demo.hold(5)
    demo.highlight(SPECTRUM)
    demo.caption("5 · Unseen generator", "Proposed system's frequency input: 32×32 log-power spectrum averaged over image patches")
    demo.hold(5)

    demo.highlight()
    demo.caption("6 · Unseen generator, rotated", "StyleGAN3-T-generated face (not in training data), rotated by 90°")
    demo.run_example("StyleGAN3-T fake (not in training data), rotated 90°")
    demo.highlight(BASE_VERDICT, OURS_VERDICT)
    demo.caption("6 · Unseen generator, rotated", "Base paper's method: REAL (score −1.57). Proposed system: FAKE (score +7.58)")
    demo.hold(6)

    demo.highlight()
    demo.caption("Note", "These six images were selected from the test set to show cases where the two models differ")
    demo.hold(4)

    demo.show_card(RESULTS, 15)
    demo.show_card(CONCLUSION, 11)


def encode(frames: list[tuple[float, Path]], end: float, out: Path, workdir: Path) -> None:
    lines = []
    for (t, path), nxt in zip(frames, frames[1:] + [(end, None)]):
        lines.append(f"file '{path.as_posix()}'\nduration {max(nxt[0] - t, 0.001):.4f}")
    lines.append(f"file '{frames[-1][1].as_posix()}'")   # concat demuxer needs the last file repeated
    listing = workdir / "frames.txt"
    listing.write_text("\n".join(lines))
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(out),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the captioned demo video.")
    parser.add_argument("--out", type=Path, default=Path("runs/DSF-Net_Demo.mp4"))
    parser.add_argument("--app-only", action="store_true",
                        help="Record only the captioned app walkthrough, without the title/method/results cards.")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="dsfnet-demo-"))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            recorder = Recorder(page, workdir)
            page.set_content(card(""))
            recorder.start()
            script(Demo(page, app_only=args.app_only))
            recorder.stop()
            browser.close()
        encode(recorder.frames, recorder.end, args.out, workdir)
        print(f"wrote {args.out} ({len(recorder.frames)} frames, {recorder.end - recorder.frames[0][0]:.0f} s)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
