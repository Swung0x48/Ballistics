"""Top views: the map's real geometry seen from above, with ways drawn over it."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import Mesh

BG, FLOOR, EDGE, RAIL, PLACED, INK, WARN, FLY, SEEN = (248, 249, 251), (150, 164, 182), (70, 84, 104), (158, 110, 46), (126, 92, 168), (20, 28, 40), (196, 60, 48), (212, 122, 28), (16, 132, 120)


FAINT = (168, 170, 176)        # a route past a break that was left
SURFACE, INK2, MUTED, NEUTRAL, RAIL_GREY, CRITICAL = (252, 252, 251), (82, 81, 78), (150, 149, 144), (82, 81, 78), (128, 127, 122), (208, 59, 59)
RAMP = [(214, 214, 214), (203, 203, 203), (192, 192, 192), (181, 181, 181), (170, 170, 170), (160, 160, 160), (149, 149, 149),
        (138, 138, 138), (127, 127, 127), (116, 116, 116), (106, 106, 106), (95, 95, 95), (84, 84, 84)]      # greys, light (low) to dark (high)
STEEL = (62, 100, 150)         # what is on the map but is not floor, on the grey height map: rails and placed objects


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
        """height: colour the floor by its height in greys from light (low) to dark (high), rails and placed objects
        in one blue; `height_key` then draws the scale.

        The picture has three bands: a header (title and line keys), the map, and a footer (scales). Keys and
        scales are drawn on a layer of their own (`ui`), and the header and footer are cleared before that layer
        goes on, so that nothing of the map or of a route ever lies under a key, whatever the map's shape."""
        lo = np.minimum(np.minimum(mesh.A, mesh.B), mesh.C).min(0)
        hi = np.maximum(np.maximum(mesh.A, mesh.B), mesh.C).max(0)
        x0, x1, z0, z1 = (lo[0], hi[0], lo[2], hi[2]) if box is None else box
        self.x0, self.z0 = x0 - margin, z0 - margin
        self.k = width / (x1 - x0 + 2 * margin)
        self.top, self.foot = 104, 100
        self.img = Image.new("RGB", (width, int((z1 - z0 + 2 * margin) * self.k) + self.top + self.foot), BG)
        self.pen = ImageDraw.Draw(self.img, "RGBA")
        self.chrome = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        self.ui = ImageDraw.Draw(self.chrome)             # the header and footer: title, keys, scales
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
                    self.pen.polygon(pts, fill=STEEL + (215,))
                else:
                    self.pen.polygon(pts, fill=ramp((mean[t] - self.y_lo) / max(self.y_hi - self.y_lo, 1e-6)))
            elif mesh.placed[t]:
                self.pen.polygon(pts, fill=PLACED + (150,))
            else:
                shade = 0.75 + 0.5 * (float(mesh.A[t, 1]) - lo_y) / max(hi_y - lo_y, 1e-6)
                self.pen.polygon(pts, fill=tuple(int(min(255, c * shade)) for c in FLOOR))
        for s in mesh.tubes:
            self.pen.line([self.P(s[0], s[2]), self.P(s[3], s[5])], fill=STEEL if height else RAIL, width=max(1, int(0.5 * self.k)))
        if title:
            self.ui.text((16, 14), title, fill=INK, font=_font(34))

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
            self.ui.line([(x, 80), (x + 36, 80)], fill=colour, width=5)
            self.ui.text((x + 42, 66), text, fill=INK, font=f)
            x += 60 + int(self.ui.textlength(text, font=f))

    def foot_y(self) -> int:
        """Where a scale's bar goes in the footer."""
        return self.img.size[1] - self.foot + 34

    def height_key(self, x: int, y: int | None = None, w: int = 420, h: int = 22, label: str = "路面高度"):
        """The height scale: a bar from the lowest to the highest floor with their values (in the footer)."""
        y = self.foot_y() if y is None else y
        for i in range(w):
            self.ui.line([(x + i, y), (x + i, y + h)], fill=ramp(i / (w - 1)))
        self.ui.rectangle((x, y, x + w, y + h), outline=MUTED)
        f = _font(20)
        self.ui.text((x, y - 28), label, fill=INK2, font=f)
        for frac in (0.0, 0.5, 1.0):
            v = self.y_lo + frac * (self.y_hi - self.y_lo)
            t = "%.0f" % v
            self.ui.line([(x + frac * w, y + h), (x + frac * w, y + h + 6)], fill=MUTED)
            self.ui.text((x + frac * w - self.ui.textlength(t, font=f) / 2, y + h + 8), t, fill=INK2, font=f)

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

    def speed_labels(self, segments, every: float = 70.0, size: int = 20):
        """The ball's horizontal speed written along a route, every `every` pixels of the way, where a label does not
        land on one already there. segments: traces in order (rows of position ...), None where the route breaks off.
        The speed is the distance covered over the last 8 ticks, so one tick's stop at a landing does not show."""
        f = _font(size)
        placed = []
        run = every * 0.5
        for tr in segments:
            if tr is None or len(tr) < 10:
                run = every * 0.5
                continue
            q = np.array([self.P(p[0], p[2]) for p in tr[:, :3]])
            step = np.r_[0.0, np.hypot(*(q[1:] - q[:-1]).T)]
            for k in range(8, len(tr)):
                run += step[k]
                if run < every:
                    continue
                run = 0.0
                speed = float(np.hypot(tr[k, 0] - tr[k - 8, 0], tr[k, 2] - tr[k - 8, 2])) * 66.0 / 8.0
                text = "%.0f" % speed
                w = int(self.pen.textlength(text, font=f))
                x, y = int(q[k, 0]) + 8, int(q[k, 1]) - size - 6
                box = (x - 3, y - 1, x + w + 3, y + size + 5)
                if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in placed):
                    continue
                placed.append(box)
                self.pen.ellipse((q[k, 0] - 4, q[k, 1] - 4, q[k, 0] + 4, q[k, 1] + 4), fill=SURFACE, outline=INK, width=2)
                self.pen.rectangle(box, fill=SURFACE + (225,))
                self.pen.text((x, y), text, fill=INK, font=f)

    def save(self, path):
        w, h = self.img.size
        ground = SURFACE if self.height else BG
        self.pen.rectangle((0, 0, w, self.top - 1), fill=ground)
        self.pen.rectangle((0, h - self.foot, w, h), fill=ground)
        Image.alpha_composite(self.img.convert("RGBA"), self.chrome).convert("RGB").save(path)


SPEED_TOP = 20.0                # speeds are coloured on a fixed scale from 0 to this (faster is drawn as this)
SPEED_RAMP = [(0xfb, 0xd2, 0xb9), (0xf3, 0x9a, 0x6a), (0xeb, 0x68, 0x34), (0xb8, 0x45, 0x1a), (0x7d, 0x2a, 0x0c), (0x4a, 0x17, 0x05)]      # one hue (not the map's), light slow to dark fast
PATTERN = {"floor": None, "rail": (18.0, 8.0), "air": (3.0, 8.0)}       # how the stroke is broken, in pixels: drawn, skipped


def speed_colour(v: float):
    f = min(max(float(v) / SPEED_TOP, 0.0), 1.0) * (len(SPEED_RAMP) - 1)
    k = min(int(f), len(SPEED_RAMP) - 2)
    a, b, t = SPEED_RAMP[k], SPEED_RAMP[k + 1], f - k
    return tuple(int(round(a[c] + (b[c] - a[c]) * t)) for c in range(3))


def _states(tr) -> np.ndarray:
    """Per tick: 0 on the floor, 1 on a rail, 2 in the air; runs shorter than 10 ticks take the state before them
    (a rolling ball loses touch for a tick or two all the time)."""
    st = np.where(tr[:, 3] == 0, 2, np.where(tr[:, 8] > 0, 1, 0))
    i = 0
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if j - i + 1 < 10 and i > 0:
            st[i:j + 1] = st[i - 1]
        i = j + 1
    return st


def _speeds(tr) -> np.ndarray:
    """Horizontal speed per tick, over the last 8 ticks (one tick's stop at a landing does not show)."""
    out = np.zeros(len(tr))
    k = np.arange(len(tr))
    b = np.maximum(k - 8, 0)
    out = np.hypot(tr[k, 0] - tr[b, 0], tr[k, 2] - tr[b, 2]) * 66.0 / np.maximum(k - b, 1)
    return out


def draw_route(v: "TopView", tr, width: int, faint: bool = False):
    """One stretch of a route: coloured by the ball's speed, solid on the floor, long dashes on a rail, dots in the
    air; in a pale grey when it lies past a break that was left."""
    if len(tr) < 2:
        return
    q = np.array([v.P(p[0], p[2]) for p in tr[:, :3]])
    st, sp = _states(tr), _speeds(tr)
    names = {0: "floor", 1: "rail", 2: "air"}
    pieces = []                                            # (pixel points, speeds) of every drawn piece
    i = 0
    while i < len(tr) - 1:
        j = i
        while j + 1 < len(tr) - 1 and st[j + 1] == st[i]:
            j += 1
        pts, spd = q[i:j + 2], sp[i:j + 2]
        pat = PATTERN[names[int(st[i])]]
        if pat is None:
            pieces.append((pts, spd))
        else:
            d = np.r_[0.0, np.cumsum(np.hypot(*(pts[1:] - pts[:-1]).T))]
            on, off = pat
            t = 0.0
            while t < d[-1]:
                a, b = t, min(t + on, d[-1])
                xs = np.r_[a, d[(d > a) & (d < b)], b]
                pieces.append((np.c_[np.interp(xs, d, pts[:, 0]), np.interp(xs, d, pts[:, 1])], np.interp(xs, d, spd)))
                t += on + off
        i = j + 1
    for pts, _spd in pieces:                               # the ring of the surface colour first, under every piece
        v.pen.line([tuple(x) for x in pts], fill=SURFACE, width=width + 4, joint="curve")
        if len(pts) == 2 and np.hypot(*(pts[1] - pts[0])) < width:      # a dot
            c = pts.mean(0)
            v.pen.ellipse((c[0] - width / 2 - 2, c[1] - width / 2 - 2, c[0] + width / 2 + 2, c[1] + width / 2 + 2), fill=SURFACE)
    for pts, spd in pieces:
        for k in range(0, len(pts) - 1, 6):                # short runs, each in the colour of its own speed
            run = pts[k:k + 7]
            colour = FAINT if faint else speed_colour(float(spd[k:k + 7].mean()))
            v.pen.line([tuple(x) for x in run], fill=colour, width=width, joint="curve")
        if len(pts) == 2 and np.hypot(*(pts[1] - pts[0])) < width:
            c = pts.mean(0)
            v.pen.ellipse((c[0] - width / 2, c[1] - width / 2, c[0] + width / 2, c[1] + width / 2), fill=FAINT if faint else speed_colour(float(spd.mean())))


def route_key(v: "TopView", x: int, y: int):
    """What the line styles and the line colour mean, drawn from (x, y) to the right; returns the x after it."""
    f = _font(20)
    mid = speed_colour(10.0)
    for name, text in (("floor", "在路面上"), ("rail", "在钢轨上"), ("air", "腾空")):
        pat = PATTERN[name]
        if pat is None:
            v.ui.line([(x, y), (x + 54, y)], fill=mid, width=5)
        else:
            t = 0.0
            while t < 54:
                v.ui.line([(x + t, y), (x + min(t + pat[0], 54), y)], fill=mid, width=5)
                t += pat[0] + pat[1]
        v.ui.text((x + 62, y - 14), text, fill=INK2, font=f)
        x += 92 + int(v.ui.textlength(text, font=f))
    v.ui.line([(x, y), (x + 40, y)], fill=mid, width=5)
    v.ui.text((x + 46, y - 14), "跟随器", fill=INK2, font=f)
    x += 76 + int(v.ui.textlength("跟随器", font=f))
    v.ui.line([(x, y), (x + 40, y)], fill=mid, width=10)
    v.ui.text((x + 46, y - 14), "细搜出来的", fill=INK2, font=f)
    x += 76 + int(v.ui.textlength("细搜出来的", font=f))
    v.ui.line([(x, y), (x + 40, y)], fill=FAINT, width=5)
    v.ui.text((x + 46, y - 14), "留下的断点之后（不算到达）", fill=INK2, font=f)
    return x + 76 + int(v.ui.textlength("留下的断点之后（不算到达）", font=f))


def speed_key(v: "TopView", x: int, y: int | None = None, w: int = 420, h: int = 22):
    y = v.foot_y() if y is None else y
    for k in range(w):
        v.ui.line([(x + k, y), (x + k, y + h)], fill=speed_colour(SPEED_TOP * k / (w - 1)))
    v.ui.rectangle((x, y, x + w, y + h), outline=MUTED)
    f = _font(20)
    v.ui.text((x, y - 28), "线的颜色：球的水平速度", fill=INK2, font=f)
    for frac, t in ((0.0, "0"), (0.5, "%.0f" % (SPEED_TOP / 2)), (1.0, "%.0f 以上" % SPEED_TOP)):
        v.ui.line([(x + frac * w, y + h), (x + frac * w, y + h + 6)], fill=MUTED)
        v.ui.text((x + frac * w - v.ui.textlength(t, font=f) / 2, y + h + 8), t, fill=INK2, font=f)


def _routes(v: "TopView", results, label_every: float, label_size: int, numbers: bool):
    """Every sector's route, its speed labels, and its breaks."""
    for res in results:
        after = False
        for sg in res["segments"]:
            if sg["type"] == "assumed":
                after = True
                continue
            draw_route(v, sg["trace"], 10 if sg["type"] == "cross" else 5, faint=after)
    for res in results:
        v.speed_labels([None if sg["type"] == "assumed" else sg["trace"] for sg in res["segments"]], every=label_every, size=label_size)
    for res in results:
        for b in res["breaks"]:
            px, py = v.P(b["near"][0], b["near"][2])
            if b["status"] == "unresolved":
                v.pen.ellipse((px - 16, py - 16, px + 16, py + 16), fill=SURFACE, outline=CRITICAL, width=6)
                why = "（去掉道具球才能过）" if b.get("would_pass") else ""
                text, ft = "第 %d 小节 断点 %s 未过%s" % (res["sector"], b["no"], why), _font(24)
                tw = int(v.pen.textlength(text, font=ft))
                tx, ty = max(px - tw - 30, 16), py - 62           # up and to the left, on a plate, clear of the route, inside the picture
                v.pen.rectangle((tx - 8, ty - 4, tx + tw + 8, ty + 34), fill=SURFACE + (235,), outline=CRITICAL)
                v.pen.text((tx, ty), text, fill=INK, font=ft)
                v.pen.line([(tx + tw + 8, ty + 34), (px - 12, py - 12)], fill=CRITICAL, width=2)
            elif b["status"] == "crossed" and numbers:
                ft = _font(20)
                tw = int(v.pen.textlength(b["no"], font=ft))
                v.pen.rectangle((px + 10, py + 8, px + 18 + tw, py + 34), fill=SURFACE + (235,), outline=INK2)
                v.pen.text((px + 14, py + 8), b["no"], fill=INK, font=ft)
                v.pen.ellipse((px - 6, py - 6, px + 6, py + 6), fill=SURFACE, outline=INK, width=3)


def draw_staged(mesh: Mesh, res: dict, path, title: str | None = None, width: int = 2200):
    """Top view of one sector solved coarse-first, coloured by height, with its route: the colour of the line is the
    ball's speed, its style what carries the ball (solid on the floor, dashes on a rail, dots in the air), its
    width who made it (thin: the plain follower; thick: the fine search). A gap where a break was left, and pale
    grey after it. The breaks that were crossed carry their numbers."""
    pts = [sg["trace"][:, :3] for sg in res["segments"] if sg["type"] != "assumed" and len(sg["trace"])]
    pts += [np.array([b["near"], b["far"]]) for b in res["breaks"]] + [np.array([res["start"], res["goal"]])]
    A = np.concatenate(pts)
    box = (A[:, 0].min() - 8, A[:, 0].max() + 8, A[:, 2].min() - 8, A[:, 2].max() + 8)
    state = "通" if res["solved"] else "不通，留有断点 " + "、".join(res["unresolved"])
    v = TopView(mesh, box=box, width=width, title=title or "第 %d 小节（%s，轨迹 %.1f 游戏秒）" % (res["sector"], state, res["ticks"] / 132), margin=2.0, height=True)
    route_key(v, 16, 82)
    _routes(v, [res], 90.0, 20, True)
    for name, p in (("起点", res["start"]), ("目标", res["goal"])):
        px, py = v.P(p[0], p[2])
        v.pen.ellipse((px - 10, py - 10, px + 10, py + 10), fill=INK, outline=SURFACE, width=3)
        v.pen.text((px + 14, py + 4), name, fill=INK, font=_font(24))
    speed_key(v, v.img.size[0] - 1010)
    v.height_key(v.img.size[0] - 480)
    v.save(path)


def draw_level(mesh: Mesh, results: list, path, names=None, anchors=None, title: str | None = None, width: int = 2400):
    """The whole level from above, coloured by height, with the route of every sector on it, drawn as in
    `draw_staged`. Breaks that were left are marked and named."""
    v = TopView(mesh, width=width, title=title or "整关俯视图和现在的线路", margin=12.0, height=True)
    route_key(v, 16, 82)
    _routes(v, results, 110.0, 18, False)
    if anchors is not None:
        for n, p in zip(names, anchors):
            px, py = v.P(p[0], p[2])
            v.pen.ellipse((px - 10, py - 10, px + 10, py + 10), fill=INK, outline=SURFACE, width=3)
            v.pen.text((px + 14, py + 4), n, fill=INK, font=_font(24))
    speed_key(v, v.img.size[0] - 1010)
    v.height_key(v.img.size[0] - 480)
    v.save(path)


# ---- the coarse level's plan alone
COARSE_STYLE = {      # group of edge kinds -> (colour, width, dash in pixels (0: solid), name)
    "roll": ((30, 34, 44), 5, 0.0, "滚动（相邻落脚点之间）"),
    "rail": ((150, 92, 20), 5, 14.0, "沿钢轨"),
    "launch": ((14, 138, 122), 6, 5.0, "起飞边（从静止助跑推出去，物理核算过落点）"),
    "step": ((126, 72, 176), 7, 0.0, "台阶（不高于 1.5，只是提议）"),
    "link": ((208, 59, 59), 6, 9.0, "链接（粗层接不上、硬连过去的地方，也就是断点）"),
}
COARSE_GROUP = {"roll": "roll", "drop": "roll", "rail": "rail", "rail end": "rail", "launch": "launch", "step": "step", "gap": "link", "bridge": "link"}


def draw_coarse(mesh: Mesh, g, plans: list, path, title: str, anchors=None, names=None, box=None, width: int = 2200, labels: bool = True):
    """The coarse level's plan alone (`solve.coarse_plans`) over the map, coloured by height: what kind of edge
    each stretch is, and the links (the places the coarse level does not connect) ringed and, with `labels`,
    described. It is a line of positions; it has no speed and nothing says a ball can drive it.
    box: (x0, x1, z0, z1) to show; default the whole map."""
    v = TopView(mesh, box=box, width=width, title=title, margin=8.0 if box is None else 3.0, height=True)
    P = g.pos
    for plan in plans:
        nodes, kinds = plan["nodes"], plan["kinds"]
        if "starts_off" in plan:                              # from the reset point to where the plan begins: no plan at all
            a, b = np.asarray(plan["reset"], np.float64), P[nodes[0]]
            t = np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) * v.k)))[:, None]
            v.halo_line(a + t * (b - a), CRITICAL, 10, ring=2, dash=4.0)
            x, y = v.P((a[0] + b[0]) / 2, (a[2] + b[2]) / 2)
            text = "这一段没有规划（%.0f）：加上链接也到不了" % plan["starts_off"]
            f = _font(22)
            tw = int(v.pen.textlength(text, font=f))
            x = min(max(x - tw / 2, 16), v.img.size[0] - tw - 16)
            v.pen.rectangle((x - 6, y - 46, x + tw + 6, y - 12), fill=SURFACE + (240,), outline=CRITICAL, width=2)
            v.pen.text((x, y - 44), text, fill=INK, font=f)
        runs = []                                             # [group, points]: consecutive edges of one group
        for e, k in enumerate(kinds):
            u, w = nodes[e], nodes[e + 1]
            pts = [P[u], P[w]]
            if k == "launch":                                 # the run-up and the landing of the kernel run that made the edge
                pts = [P[u]] + [np.asarray(q, np.float64) for q in g.info["launches"][(u, w)]["way"][1:]] + [P[w]]
            grp = COARSE_GROUP[k]
            if runs and runs[-1][0] == grp and grp in ("roll", "rail"):
                runs[-1][1].extend(pts[1:])
            else:
                runs.append([grp, list(pts)])
        n_link = 0
        for grp, pts in runs:
            colour, w, dash, _name = COARSE_STYLE[grp]
            q = np.array(pts, np.float64)                     # a point every pixel, so that the dashes come out even
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            keep = np.r_[True, np.diff(d) > 1e-9]
            q, d = q[keep], d[keep]
            if len(q) < 2:
                continue
            t = np.arange(0.0, d[-1], 1.0 / v.k)
            v.halo_line(np.stack([np.interp(t, d, q[:, c]) for c in range(3)], 1), colour, w, ring=2, dash=dash)
            if grp not in ("link", "step"):
                continue
            a, b = q[0], q[-1]
            for end in (a, b):
                x, y = v.P(end[0], end[2])
                v.pen.ellipse((x - 9, y - 9, x + 9, y + 9), fill=SURFACE, outline=colour, width=4)
            n_link += grp == "link"
            if labels:
                x, y = v.P((a[0] + b[0]) / 2, (a[2] + b[2]) / 2)
                text = "%s：横 %.1f，高差 %+.1f" % ("链接 %d" % n_link if grp == "link" else "台阶", math.hypot(b[0] - a[0], b[2] - a[2]), b[1] - a[1])
                f = _font(20)
                tw = int(v.pen.textlength(text, font=f))
                x = min(x + 26, v.img.size[0] - tw - 16)
                v.pen.rectangle((x, y - 44, x + 12 + tw, y - 14), fill=SURFACE + (235,), outline=colour)
                v.pen.text((x + 6, y - 42), text, fill=INK, font=f)
    for name, p in zip(names or (), anchors or ()):
        if p is None:
            continue
        x, y = v.P(p[0], p[2])
        if 0 <= x <= v.img.size[0] and v.top <= y <= v.img.size[1] - v.foot:
            v.pen.ellipse((x - 10, y - 10, x + 10, y + 10), fill=INK, outline=SURFACE, width=3)
            v.pen.text((x + 14, y + 4), name, fill=INK, font=_font(24))
    x, f = 16, _font(20)
    for grp in ("roll", "rail", "launch", "step", "link"):
        colour, w, dash, name = COARSE_STYLE[grp]
        t = 0.0
        while t < 44:
            v.ui.line([(x + t, 80), (x + min(t + (dash or 44), 44), 80)], fill=colour, width=w)
            t += 2 * dash if dash else 44
        v.ui.text((x + 52, 66), name, fill=INK2, font=f)
        x += 76 + int(v.ui.textlength(name, font=f))
    v.height_key(v.img.size[0] - 480)
    v.save(path)
