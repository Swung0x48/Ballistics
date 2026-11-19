"""Top views: the map's real geometry seen from above, with ways drawn over it."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Mesh

BG, FLOOR, EDGE, RAIL, PLACED, INK, WARN, FLY = (248, 249, 251), (150, 164, 182), (70, 84, 104), (158, 110, 46), (126, 92, 168), (20, 28, 40), (196, 60, 48), (212, 122, 28)


def _font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class TopView:
    """x to the right, z upwards on the page is NOT assumed: z increases downwards, as in the map's own top view."""

    def __init__(self, mesh: Mesh, box=None, width: int = 2000, title: str = "", margin: float = 15.0):
        lo = np.minimum(np.minimum(mesh.A, mesh.B), mesh.C).min(0)
        hi = np.maximum(np.maximum(mesh.A, mesh.B), mesh.C).max(0)
        x0, x1, z0, z1 = (lo[0], hi[0], lo[2], hi[2]) if box is None else box
        self.x0, self.z0 = x0 - margin, z0 - margin
        self.k = width / (x1 - x0 + 2 * margin)
        self.top = 70
        self.img = Image.new("RGB", (width, int((z1 - z0 + 2 * margin) * self.k) + self.top), BG)
        self.pen = ImageDraw.Draw(self.img, "RGBA")
        y = mesh.A[:, 1]
        order = np.argsort(np.minimum(np.minimum(mesh.A[:, 1], mesh.B[:, 1]), mesh.C[:, 1]))     # low first, high drawn over it
        lo_y, hi_y = float(y.min()), float(y.max())
        for t in order:
            if mesh.rail[t]:
                continue
            if mesh.N[t, 1] < 0.3:
                continue                                  # walls and undersides are not seen from above
            pts = [self.P(p[0], p[2]) for p in (mesh.A[t], mesh.B[t], mesh.C[t])]
            if mesh.placed[t]:
                self.pen.polygon(pts, fill=PLACED + (150,))
            else:
                shade = 0.75 + 0.5 * (float(mesh.A[t, 1]) - lo_y) / max(hi_y - lo_y, 1e-6)
                self.pen.polygon(pts, fill=tuple(int(min(255, c * shade)) for c in FLOOR))
        for s in mesh.tubes:
            self.pen.line([self.P(s[0], s[2]), self.P(s[3], s[5])], fill=RAIL, width=max(1, int(0.5 * self.k)))
        if title:
            self.pen.text((16, 14), title, fill=INK, font=_font(34))

    def P(self, x, z):
        return (float((x - self.x0) * self.k), float((z - self.z0) * self.k) + self.top)

    def line(self, pts, colour=INK, width: int = 5, dash: bool = False):
        q = [self.P(p[0], p[2]) for p in pts]
        if not dash:
            self.pen.line(q, fill=colour, width=width, joint="curve")
            return
        for a, b in zip(q[::2], q[1::2]):
            self.pen.line([a, b], fill=colour, width=width)

    def dot(self, p, label: str = "", colour=INK, r: int = 9, size: int = 22):
        x, y = self.P(p[0], p[2])
        self.pen.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline=BG)
        if label:
            self.pen.text((x + r + 4, y - size), label, fill=INK, font=_font(size))

    def legend(self, rows):
        x = 16
        f = _font(20)
        for colour, text in rows:
            self.pen.line([(x, 60), (x + 36, 60)], fill=colour, width=5)
            self.pen.text((x + 42, 48), text, fill=INK, font=f)
            x += 60 + int(self.pen.textlength(text, font=f))

    def save(self, path):
        self.img.save(path)
