"""Top views: the map's real geometry seen from above, with ways drawn over it."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Mesh

BG, FLOOR, EDGE, RAIL, PLACED, INK, WARN, FLY, SEEN = (248, 249, 251), (150, 164, 182), (70, 84, 104), (158, 110, 46), (126, 92, 168), (20, 28, 40), (196, 60, 48), (212, 122, 28), (16, 132, 120)


FAINT = (168, 170, 176)        # a route past a break that was left
SURFACE, INK2, MUTED, NEUTRAL, RAIL_GREY, CRITICAL = (252, 252, 251), (82, 81, 78), (150, 149, 144), (82, 81, 78), (128, 127, 122), (208, 59, 59)
RAMP = [(0xcd, 0xe2, 0xfb), (0xb7, 0xd3, 0xf6), (0x9e, 0xc5, 0xf4), (0x86, 0xb6, 0xef), (0x6d, 0xa7, 0xec), (0x55, 0x98, 0xe7), (0x39, 0x87, 0xe5),
        (0x2a, 0x78, 0xd6), (0x25, 0x6a, 0xbf), (0x1c, 0x5c, 0xab), (0x18, 0x4f, 0x95), (0x10, 0x42, 0x81), (0x0d, 0x36, 0x6b)]      # one hue, light to dark


def ramp(f: float):
    f = min(max(float(f), 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(f), len(RAMP) - 2)
    a, b, t = RAMP[i], RAMP[i + 1], f - i
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def _font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class TopView:
    """x to the right, z upwards on the page is NOT assumed: z increases downwards, as in the map's own top view."""

    def __init__(self, mesh: Mesh, box=None, width: int = 2000, title: str = "", margin: float = 15.0, height: bool = False):
        """height: colour the floor by its height, one hue from light (low) to dark (high), rails and placed objects
        in neutral greys; `height_key` then draws the scale."""
        lo = np.minimum(np.minimum(mesh.A, mesh.B), mesh.C).min(0)
        hi = np.maximum(np.maximum(mesh.A, mesh.B), mesh.C).max(0)
        x0, x1, z0, z1 = (lo[0], hi[0], lo[2], hi[2]) if box is None else box
        self.x0, self.z0 = x0 - margin, z0 - margin
        self.k = width / (x1 - x0 + 2 * margin)
        self.top = 104
        self.img = Image.new("RGB", (width, int((z1 - z0 + 2 * margin) * self.k) + self.top), BG)
        self.pen = ImageDraw.Draw(self.img, "RGBA")
        y = mesh.A[:, 1]
        order = np.argsort(np.minimum(np.minimum(mesh.A[:, 1], mesh.B[:, 1]), mesh.C[:, 1]))     # low first, high drawn over it
        lo_y, hi_y = float(y.min()), float(y.max())
        self.height = height
        if height:
            seen = (~mesh.rail) & (mesh.N[:, 1] >= 0.3) & (~mesh.placed)
            mean = (mesh.A[:, 1] + mesh.B[:, 1] + mesh.C[:, 1]) / 3.0
            self.y_lo, self.y_hi = float(mean[seen].min()), float(mean[seen].max())
            self.img.paste(SURFACE, (0, 0, self.img.size[0], self.img.size[1]))
        for t in order:
            if mesh.rail[t]:
                continue
            if mesh.N[t, 1] < 0.3:
                continue                                  # walls and undersides are not seen from above
            pts = [self.P(p[0], p[2]) for p in (mesh.A[t], mesh.B[t], mesh.C[t])]
            if height:
                if mesh.placed[t]:
                    self.pen.polygon(pts, fill=NEUTRAL + (200,))
                else:
                    self.pen.polygon(pts, fill=ramp((mean[t] - self.y_lo) / max(self.y_hi - self.y_lo, 1e-6)))
            elif mesh.placed[t]:
                self.pen.polygon(pts, fill=PLACED + (150,))
            else:
                shade = 0.75 + 0.5 * (float(mesh.A[t, 1]) - lo_y) / max(hi_y - lo_y, 1e-6)
                self.pen.polygon(pts, fill=tuple(int(min(255, c * shade)) for c in FLOOR))
        for s in mesh.tubes:
            self.pen.line([self.P(s[0], s[2]), self.P(s[3], s[5])], fill=RAIL_GREY if height else RAIL, width=max(1, int(0.5 * self.k)))
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
            self.pen.line([(x, 80), (x + 36, 80)], fill=colour, width=5)
            self.pen.text((x + 42, 66), text, fill=INK, font=f)
            x += 60 + int(self.pen.textlength(text, font=f))

    def height_key(self, x: int, y: int, w: int = 420, h: int = 22, label: str = "路面高度"):
        """The height scale: a bar from the lowest to the highest floor with their values."""
        for i in range(w):
            self.pen.line([(x + i, y), (x + i, y + h)], fill=ramp(i / (w - 1)))
        self.pen.rectangle((x, y, x + w, y + h), outline=MUTED)
        f = _font(20)
        self.pen.text((x, y - 28), label, fill=INK2, font=f)
        for frac in (0.0, 0.5, 1.0):
            v = self.y_lo + frac * (self.y_hi - self.y_lo)
            t = "%.0f" % v
            self.pen.line([(x + frac * w, y + h), (x + frac * w, y + h + 6)], fill=MUTED)
            self.pen.text((x + frac * w - self.pen.textlength(t, font=f) / 2, y + h + 8), t, fill=INK2, font=f)

    def halo_line(self, pts, colour, width: int, ring: int = 2, dash: float = 0.0, alpha: int = 255):
        """A line with a ring of the surface colour round it, so it reads on light and dark ground alike.
        dash: length in pixels of the drawn and the skipped pieces (0: solid)."""
        q = np.array([self.P(p[0], p[2]) for p in pts])
        if len(q) < 2:
            return
        runs = [q]
        if dash > 0:
            d = np.r_[0.0, np.cumsum(np.hypot(*(q[1:] - q[:-1]).T))]
            runs, k = [], 0
            while k * dash < d[-1]:
                m = (d >= k * dash) & (d <= (k + 1) * dash)
                if k % 2 == 0 and m.sum() >= 2:
                    runs.append(q[m])
                k += 1
        for r in runs:
            xy = [tuple(v) for v in r]
            self.pen.line(xy, fill=SURFACE + (alpha,), width=width + 2 * ring, joint="curve")
        for r in runs:
            self.pen.line([tuple(v) for v in r], fill=colour + (alpha,), width=width, joint="curve")

    def save(self, path):
        self.img.save(path)


def draw_staged(mesh: Mesh, res: dict, path, title: str | None = None, width: int = 2000):
    """Top view of one sector solved coarse-first: the stretches the plain follower rolled (thin), the crossings the
    fine level searched (thick, coloured by what the ball is doing), a gap where a break stayed unresolved, and the
    breaks by number."""
    GROUND, RAILC, CROSS = (20, 28, 40), (40, 90, 200), (30, 140, 70)
    pts = [sg["trace"][:, :3] for sg in res["segments"] if sg["type"] != "assumed" and len(sg["trace"])]
    pts += [np.array([b["near"], b["far"]]) for b in res["breaks"]] + [np.array([res["start"], res["goal"]])]
    A = np.concatenate(pts)
    box = (A[:, 0].min() - 8, A[:, 0].max() + 8, A[:, 2].min() - 8, A[:, 2].max() + 8)
    state = "到达目标" if res["solved"] else "不通，留有断点 " + "、".join(res["unresolved"])
    v = TopView(mesh, box=box, width=width, title=title or "第 %d 小节：先粗后细（%s，轨迹 %.1f 游戏秒）" % (res["sector"], state, res["ticks"] / 132), margin=2.0)
    v.legend([(GROUND, "跟随器沿粗层通路滚（细线）"), (RAILC, "在钢轨上"), (CROSS, "细搜出来的过断点段（粗线）"), (FLY, "腾空"), ((170, 175, 185), "断点之后（淡色，不算到达）")])
    after = False                                          # past a break that was left: drawn faint, it does not count as reached
    for sg in res["segments"]:
        if sg["type"] == "assumed":
            after = True
            continue
        if not len(sg["trace"]):
            continue
        tr = sg["trace"]
        cross = sg["type"] == "cross"
        kind = np.where(tr[:, 3] == 0, 0, np.where(tr[:, 8] > 0, 1, 4))
        colour = {0: FLY, 1: RAILC, 4: CROSS if cross else GROUND}
        i = 0
        while i < len(tr) - 1:
            j = i
            while j + 1 < len(tr) - 1 and kind[j + 1] == kind[i]:
                j += 1
            c = colour[int(kind[i])]
            if after:                                      # an opaque pale tint: see-through strokes pile up where they overlap
                c = tuple(int(0.3 * a + 0.7 * b) for a, b in zip(c, BG))
            v.line(tr[i:j + 2, :3], c, 7 if cross else 3)
            i = j + 1
    for b in res["breaks"]:
        if b["status"] == "crossed":
            v.dot(b["near"], "%s 已过" % b["no"], CROSS, 8, 22)
        elif b["status"] == "unresolved":
            w = b.get("would_pass")
            why = "" if not w else "（去掉道具球才能过）"
            v.dot(b["near"], "断点 %s 未过%s" % (b["no"], why), WARN, 10, 24)
            v.dot(b["far"], "", WARN, 6)
        elif b["status"] == "rolled":
            v.dot(b["near"], "", (120, 130, 140), 4)
    v.dot(res["start"], "起点")
    v.dot(res["goal"], "目标")
    v.save(path)


def draw_level(mesh: Mesh, results: list, path, names=None, anchors=None, title: str | None = None, width: int = 2400):
    """The whole level from above, coloured by height, with the route of every sector on it: thin where the plain
    follower rolled, thick where the fine level searched a crossing, dashed in the air, faint past a break that was
    left (that part does not count as reached). Breaks that were left are marked and named."""
    v = TopView(mesh, width=width, title=title or "整关俯视图（按路面高度上色）和现在的线路", margin=12.0, height=True)
    f = _font(20)
    x = 16
    for kind, text in (("thin", "跟随器沿粗层通路滚"), ("thick", "细搜出来的过断点段"), ("dash", "腾空"), ("faint", "留下的断点之后（不算到达）")):
        y = 80
        if kind == "dash":
            for k in range(3):
                v.pen.line([(x + 14 * k, y), (x + 14 * k + 8, y)], fill=INK, width=4)
        else:
            v.pen.line([(x, y), (x + 36, y)], fill=FAINT if kind == "faint" else INK, width=8 if kind == "thick" else 4)
        v.pen.text((x + 44, y - 14), text, fill=INK2, font=f)
        x += 70 + int(v.pen.textlength(text, font=f))
    v.pen.ellipse((x, 70, x + 20, 90), outline=CRITICAL, width=5)
    v.pen.text((x + 28, 66), "留下的断点", fill=INK2, font=f)
    x += 60 + int(v.pen.textlength("留下的断点", font=f))
    v.pen.rectangle((x, 72, x + 18, 90), fill=NEUTRAL + (200,))
    v.pen.text((x + 26, 66), "摆放的东西（机关、箱子、道具球）", fill=INK2, font=f)
    for res in results:
        after = False
        for sg in res["segments"]:
            if sg["type"] == "assumed":
                after = True
                continue
            tr = sg["trace"]
            if not len(tr):
                continue
            air = tr[:, 3] == 0
            wide = 9 if sg["type"] == "cross" else 4
            i = 0
            while i < len(tr) - 1:
                j = i
                while j + 1 < len(tr) - 1 and air[j + 1] == air[i]:
                    j += 1
                flying = bool(air[i]) and j - i >= 10
                v.halo_line(tr[i:j + 2, :3], FAINT if after else INK, wide, dash=10.0 if flying else 0.0)
                i = j + 1
    for res in results:
        for b in res["breaks"]:
            if b["status"] == "unresolved":
                px, py = v.P(b["near"][0], b["near"][2])
                v.pen.ellipse((px - 16, py - 16, px + 16, py + 16), fill=SURFACE, outline=CRITICAL, width=6)
                why = "（去掉道具球才能过）" if b.get("would_pass") else ""
                text, ft = "第 %d 小节 断点 %s 未过%s" % (res["sector"], b["no"], why), _font(24)
                tw = int(v.pen.textlength(text, font=ft))
                tx, ty = px - tw - 30, py - 62                    # up and to the left, on a plate, clear of the route
                v.pen.rectangle((tx - 8, ty - 4, tx + tw + 8, ty + 34), fill=SURFACE + (235,), outline=CRITICAL)
                v.pen.text((tx, ty), text, fill=INK, font=ft)
                v.pen.line([(tx + tw + 8, ty + 34), (px - 12, py - 12)], fill=CRITICAL, width=2)
    if anchors is not None:
        for n, p in zip(names, anchors):
            px, py = v.P(p[0], p[2])
            v.pen.ellipse((px - 10, py - 10, px + 10, py + 10), fill=INK, outline=SURFACE, width=3)
            v.pen.text((px + 14, py + 4), n, fill=INK, font=_font(24))
    v.height_key(v.img.size[0] - 480, v.img.size[1] - 90)
    v.save(path)
