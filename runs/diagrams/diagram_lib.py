"""Tiny SVG diagram helper in the style of the reference architecture diagram:
blue boxes = base paper, orange = our work, blue with orange dashed border =
base-paper step we changed, dashed container = offline stage."""
import base64
import html
from pathlib import Path

FONTS = Path("/mnt/c/Windows/Fonts")
STYLE = {
    "base": dict(fill="#dbe6f2", stroke="#2f3e4e", dash=None, sw=2),
    "ours": dict(fill="#f8c9a0", stroke="#4a3020", dash=None, sw=2),
    "mod": dict(fill="#dbe6f2", stroke="#c0601c", dash="9,6", sw=2.6),
    "plain": dict(fill="#ffffff", stroke="#222222", dash=None, sw=2),
}


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.cx, self.cy = x + w / 2, y + h / 2
        self.t, self.b = (self.cx, y), (self.cx, y + h)
        self.l, self.r = (x, self.cy), (x + w, self.cy)


class Diagram:
    def __init__(self, w, h, title=None):
        self.w, self.h, self.el = w, h, []
        if title:
            self.text(w / 2, 58, title, size=34, weight=700)

    def text(self, x, y, t, size=19, weight=400, italic=False, anchor="middle", fill="#111"):
        for i, line in enumerate(t.split("\n")):
            self.el.append(
                f'<text x="{x}" y="{y + i * size * 1.25:.1f}" font-size="{size}" font-weight="{weight}" '
                f'font-style="{"italic" if italic else "normal"}" text-anchor="{anchor}" fill="{fill}">'
                f"{html.escape(line)}</text>")

    def module(self, x, y, w, h, title, sub=None, dashed=False, fill="#fff", align="center", title_dx=25):
        dash = ' stroke-dasharray="14,9"' if dashed else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="{fill}" '
                       f'stroke="#222" stroke-width="2.4"{dash}/>')
        if align == "left":
            self.text(x + title_dx, y + 40, title, size=24, weight=700, anchor="start")
        else:
            self.text(x + w / 2, y + 40, title, size=26, weight=700)
        if sub:
            self.text(x + w / 2, y + 68, sub, size=18)
        return Box(x, y, w, h)

    def box(self, x, y, w, h, title, detail=None, kind="ours", ts=21, ds=16.5):
        s = STYLE[kind]
        dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{s["fill"]}" '
                       f'stroke="{s["stroke"]}" stroke-width="{s["sw"]}"{dash}/>')
        tl = title.split("\n")
        dl = detail.split("\n") if detail else []
        th, dh = ts * 1.22, ds * 1.25
        total = len(tl) * th + len(dl) * dh + (6 if dl else 0)
        y0 = y + h / 2 - total / 2 + ts * 0.8
        self.text(x + w / 2, y0, title, size=ts, weight=700)
        if dl:
            self.text(x + w / 2, y0 + len(tl) * th + 6, detail, size=ds, italic=True, fill="#222")
        return Box(x, y, w, h)

    def arrow(self, *pts, label=None, at=None, anchor="middle", head=True, dashed=False):
        d = " ".join(f"{px},{py}" for px, py in pts)
        mk = ' marker-end="url(#ah)"' if head else ""
        da = ' stroke-dasharray="8,6"' if dashed else ""
        self.el.append(f'<polyline points="{d}" fill="none" stroke="#222" stroke-width="2.2"{mk}{da}/>')
        if label:
            self.text(*at, label, size=18, italic=True, anchor=anchor)

    def dot(self, x, y):
        self.el.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#222"/>')

    def image(self, x, y, w, h, path, border=True):
        data = base64.b64encode(Path(path).read_bytes()).decode()
        self.el.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
                       f'href="data:image/png;base64,{data}"/>')
        if border:
            self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#888" stroke-width="1.5"/>')

    def pill(self, x, y, w, h, t):
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="#fff" stroke="#222" stroke-width="2.2"/>')
        self.text(x + w / 2, y + h / 2 + 8, t, size=22, weight=700)
        return Box(x, y, w, h)

    def legend(self, x, y, w, entries=("base", "ours", "mod", "offline")):
        names = {
            "base": "Base paper (Hafezi et al., 2026)",
            "ours": "Our work",
            "mod": "Base-paper step, changed by us",
            "offline": "Offline stage (before testing)",
        }
        h = 80 + 46 * len(entries)
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" fill="#f2f2f2" stroke="#222" stroke-width="2.2"/>')
        self.text(x + w / 2, y + 42, "Legend", size=24, weight=700)
        for i, e in enumerate(entries):
            yy = y + 68 + i * 46
            if e == "offline":
                self.el.append(f'<rect x="{x+28}" y="{yy}" width="56" height="32" rx="7" fill="#fff" stroke="#222" stroke-width="2" stroke-dasharray="7,5"/>')
            else:
                s = STYLE[e]
                dash = f' stroke-dasharray="{s["dash"]}"' if s["dash"] else ""
                self.el.append(f'<rect x="{x+28}" y="{yy}" width="56" height="32" rx="7" fill="{s["fill"]}" stroke="{s["stroke"]}" stroke-width="{s["sw"]}"{dash}/>')
            self.text(x + 104, yy + 23, names[e], size=19, anchor="start")

    def svg(self):
        faces = "".join(
            f'@font-face{{font-family:A;src:url(data:font/ttf;base64,'
            f'{base64.b64encode((FONTS / f).read_bytes()).decode()});font-weight:{wt};font-style:{st}}}'
            for f, wt, st in (("arial.ttf", 400, "normal"), ("arialbd.ttf", 700, "normal"),
                              ("ariali.ttf", 400, "italic"), ("arialbi.ttf", 700, "italic")))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}"><defs><style>{faces} text{{font-family:A,Arial,sans-serif}}</style>'
                '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#fff"/>' + "".join(self.el) + "</svg>")

    def save(self, stem, scale=1.0):
        """Renders stem.png with headless Chromium."""
        from playwright.sync_api import sync_playwright
        svg = self.svg()
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": self.w, "height": self.h}, device_scale_factor=scale)
            pg.set_content(f'<html><body style="margin:0">{svg}</body></html>')
            pg.wait_for_timeout(300)
            png = pg.screenshot(clip={"x": 0, "y": 0, "width": self.w, "height": self.h})
            b.close()
        import time
        for attempt in range(10):  # Windows can briefly lock a file a viewer has open
            try:
                Path(f"{stem}.png").write_bytes(png)
                return
            except OSError:
                time.sleep(1)
        raise OSError(f"could not write {stem}.png; close any program showing it")
