"""Exact static geometry: a level's collision triangles as the engine has them, and sphere queries against them.

Everything that goes into a proof is computed here, on the triangles themselves: where a ball of radius r touches
(face, edge or vertex), how far it is from the nearest surface, where it comes to rest when lowered. Kerbs,
chamfers, wall tops and rail tubes keep their real shape; nothing is rasterised.

A mesh file (`<name>-mesh.npz`) holds the fixed bodies the player ball collides with, in world coordinates:
triangles A, B, C; per triangle the owning body and whether it belongs to a rail; optionally analytic tube
segments fitted to the rail triangles (for rail topology; contacts use the triangles, as the game does); and per
body its name, groups, friction and elasticity. It is written from a map file by `ballistics.export`."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

BALL_R = 2.0
# Measured on a recorded level 1 run (20,483 frames): the engine keeps a rolling ball this far above the triangles
# (median 0.0141, never less than 0.0099).
CONTACT_MARGIN = 0.014
FACE, EDGE, VERTEX = 2, 1, 0
FEATURE = {FACE: "face", EDGE: "edge", VERTEX: "vertex"}
CELL = 4.0  # x-z grid cell of the triangle index


@njit(cache=True)
def _closest(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Closest point of triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5): (x, y, z, feature)."""
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az, 0
    bpx, bpy, bpz = px - bx, py - by, pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz, 0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if d1 != d3 else 0.0
        return ax + v * abx, ay + v * aby, az + v * abz, 1
    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz, 0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if d2 != d6 else 0.0
        return ax + w * acx, ay + w * acy, az + w * acz, 1
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        den = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / den if den != 0.0 else 0.0
        return bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz), 1
    den = va + vb + vc
    if den == 0.0:
        return ax, ay, az, 0
    v, w = vb / den, vc / den
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w, 2


@njit(cache=True)
def _near(px, py, pz, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, stamp, tick, out_i, out_d, out_q, out_f):
    """Every triangle within `reach` of p: fills out_i (triangle), out_d (distance), out_q (closest point), out_f
    (feature); returns how many (capped at the output size)."""
    i0 = int(math.floor((px - reach - ox) / cell))
    i1 = int(math.floor((px + reach - ox) / cell))
    j0 = int(math.floor((pz - reach - oz) / cell))
    j1 = int(math.floor((pz + reach - oz) / cell))
    n = 0
    for i in range(max(i0, 0), min(i1, nx - 1) + 1):
        for j in range(max(j0, 0), min(j1, nz - 1) + 1):
            c = i * nz + j
            for s in range(start[c], start[c + 1]):
                t = items[s]
                if stamp[t] == tick:
                    continue
                stamp[t] = tick
                if lo_y[t] > py + reach or hi_y[t] < py - reach:
                    continue
                qx, qy, qz, f = _closest(px, py, pz, A[t, 0], A[t, 1], A[t, 2], B[t, 0], B[t, 1], B[t, 2], C[t, 0], C[t, 1], C[t, 2])
                d = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
                if d <= reach and n < len(out_i):
                    out_i[n], out_d[n], out_f[n] = t, d, f
                    out_q[n, 0], out_q[n, 1], out_q[n, 2] = qx, qy, qz
                    n += 1
    return n


@njit(cache=True)
def _distance(px, py, pz, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """Distance from p to the nearest triangle within `reach` (reach itself if none), and that triangle (-1)."""
    i0 = int(math.floor((px - reach - ox) / cell))
    i1 = int(math.floor((px + reach - ox) / cell))
    j0 = int(math.floor((pz - reach - oz) / cell))
    j1 = int(math.floor((pz + reach - oz) / cell))
    best, tri = reach, -1
    for i in range(max(i0, 0), min(i1, nx - 1) + 1):
        for j in range(max(j0, 0), min(j1, nz - 1) + 1):
            c = i * nz + j
            for s in range(start[c], start[c + 1]):
                t = items[s]
                if lo_y[t] > py + best or hi_y[t] < py - best:
                    continue
                qx, qy, qz, _ = _closest(px, py, pz, A[t, 0], A[t, 1], A[t, 2], B[t, 0], B[t, 1], B[t, 2], C[t, 0], C[t, 1], C[t, 2])
                d = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
                if d < best:
                    best, tri = d, t
    return best, tri


@njit(cache=True)
def _distances(P, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    out = np.empty(len(P))
    tri = np.empty(len(P), np.int64)
    for k in range(len(P)):
        out[k], tri[k] = _distance(P[k, 0], P[k, 1], P[k, 2], reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
    return out, tri


@njit(cache=True)
def _lower(px, py, pz, radius, floor_y, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """Lower a sphere straight down from (px, py, pz) until it touches: the centre height at first contact, or nan
    if it gets below floor_y untouched. Steps by the free distance (it cannot overshoot), then closes in by
    bisection on the exact distance."""
    y = py
    for _ in range(400):
        d, _t = _distance(px, y, pz, radius + 8.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
        gap = d - radius
        if gap <= 1e-9:
            return y
        if gap < 1e-4:
            hi, lo = y, y - 1e-3
            dl, _t = _distance(px, lo, pz, radius + 8.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
            if dl - radius > 0.0:
                y = lo                 # grazing: no contact within the next thousandth, keep falling
                continue
            for _ in range(40):
                mid = 0.5 * (hi + lo)
                dm, _t = _distance(px, mid, pz, radius + 8.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                if dm - radius > 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi
        y -= gap
        if y < floor_y:
            return np.nan
    return y


def _index(lo, hi, cell: float):
    """Uniform x-z grid over the triangles' boxes: (origin x, origin z, cell, nx, nz, start, items), CSR per cell."""
    n = len(lo)
    if n == 0:
        return 0.0, 0.0, cell, 1, 1, np.zeros(2, np.int64), np.zeros(0, np.int64)
    lo2, hi2 = lo[:, [0, 2]], hi[:, [0, 2]]
    origin = lo2.min(0) - 1e-6
    c0 = np.floor((lo2 - origin) / cell).astype(np.int64)
    c1 = np.floor((hi2 - origin) / cell).astype(np.int64)
    nx, nz = int(c1[:, 0].max()) + 1, int(c1[:, 1].max()) + 1
    sx, sz = c1[:, 0] - c0[:, 0] + 1, c1[:, 1] - c0[:, 1] + 1
    counts = sx * sz
    item = np.repeat(np.arange(n), counts)
    k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    cid = (c0[item, 0] + k // sz[item]) * nz + c0[item, 1] + k % sz[item]
    order = np.argsort(cid, kind="stable")
    start = np.searchsorted(cid[order], np.arange(nx * nz + 1)).astype(np.int64)
    return float(origin[0]), float(origin[1]), float(cell), nx, nz, start, np.ascontiguousarray(item[order], np.int64)


@dataclass
class Contact:
    triangle: int
    point: np.ndarray       # the closest point of the triangle to the ball centre
    distance: float         # from the centre to that point
    gap: float              # distance minus the ball radius: 0 touching, negative penetrating
    feature: str            # face, edge or vertex
    normal: np.ndarray      # unit, from the surface point to the centre
    entity: int


@dataclass
class Mesh:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    owner: np.ndarray               # index into entities, per triangle
    rail: np.ndarray                # per triangle: part of a rail entity
    tubes: np.ndarray               # (S, 7) analytic rail segments x0 y0 z0 x1 y1 z1 radius (fitted to the rail triangles)
    entities: list = field(default_factory=list)   # [{name, groups, friction, elasticity, surface}]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = np.ascontiguousarray(self.A, np.float64).reshape(-1, 3)
        self.B = np.ascontiguousarray(self.B, np.float64).reshape(-1, 3)
        self.C = np.ascontiguousarray(self.C, np.float64).reshape(-1, 3)
        self.owner = np.asarray(self.owner, np.int64)
        self.rail = np.asarray(self.rail, bool)
        self.tubes = np.asarray(self.tubes, np.float64).reshape(-1, 7)
        lo = np.minimum(np.minimum(self.A, self.B), self.C)
        hi = np.maximum(np.maximum(self.A, self.B), self.C)
        self._lo_y, self._hi_y = np.ascontiguousarray(lo[:, 1]), np.ascontiguousarray(hi[:, 1])
        self._grid = _index(lo, hi, CELL)
        self._stamp = np.zeros(len(self.A), np.int64)
        self._tick = 0
        n = np.cross(self.B - self.A, self.C - self.A)
        ln = np.linalg.norm(n, axis=1)
        self.N = n / np.where(ln == 0, 1, ln)[:, None]
        # per triangle: part of an object placed on the map (a mechanism, a box, a loose ball) at its rest pose,
        # rather than of the map's own fixed geometry
        flag = np.array([bool(e.get("placed")) for e in self.entities], bool) if self.entities else np.zeros(0, bool)
        self.placed = flag[self.owner] if len(flag) and len(self.owner) else np.zeros(len(self.A), bool)

    def subset(self, keep) -> "Mesh":
        """The mesh with only the triangles where `keep` is true."""
        keep = np.asarray(keep, bool)
        return Mesh(self.A[keep], self.B[keep], self.C[keep], self.owner[keep], self.rail[keep], self.tubes, self.entities, dict(self.meta))

    def __len__(self):
        return len(self.A)

    def _args(self):
        return (*self._grid, self.A, self.B, self.C, self._lo_y, self._hi_y)

    # ---- queries
    def distance(self, p, reach: float = 12.0):
        """(distance to the nearest surface within reach, triangle index or -1)."""
        d, t = _distance(float(p[0]), float(p[1]), float(p[2]), float(reach), *self._args())
        return float(d), int(t)

    def distances(self, P, reach: float = 12.0):
        """Per point: (distance to the nearest surface within reach, triangle index or -1)."""
        return _distances(np.ascontiguousarray(P, np.float64).reshape(-1, 3), float(reach), *self._args())

    def contacts(self, centre, radius: float = BALL_R, margin: float = 0.05) -> list:
        """Everything a ball at `centre` touches or nearly touches (within margin), nearest first. Several triangles
        meeting at one edge or corner each report it; the caller merges by point where it matters."""
        cap = 256
        out_i, out_d, out_q, out_f = np.empty(cap, np.int64), np.empty(cap), np.empty((cap, 3)), np.empty(cap, np.int64)
        self._tick += 1
        ox, oz, cell, nx, nz, start, items = self._grid
        n = _near(float(centre[0]), float(centre[1]), float(centre[2]), radius + margin, ox, oz, cell, nx, nz, start, items,
                  self.A, self.B, self.C, self._lo_y, self._hi_y, self._stamp, self._tick, out_i, out_d, out_q, out_f)
        c = np.asarray(centre, np.float64)
        res = []
        for k in np.argsort(out_d[:n], kind="stable"):
            d = float(out_d[k])
            nrm = (c - out_q[k]) / d if d > 1e-12 else self.N[out_i[k]]
            f = int(out_f[k])
            if f != FACE and abs(float(nrm @ self.N[out_i[k]])) > 1.0 - 1e-9:
                f = FACE        # on the triangle's border but straight above its plane: a seam between coplanar triangles
            res.append(Contact(int(out_i[k]), out_q[k].copy(), d, d - radius, FEATURE[f], nrm, int(self.owner[out_i[k]])))
        return res

    def rest(self, x: float, z: float, from_y: float, radius: float = BALL_R, floor_y: float | None = None) -> float:
        """Centre height at which a ball lowered straight down from (x, from_y, z) first touches; nan if nothing."""
        if floor_y is None:
            floor_y = float(self._lo_y.min()) - 10.0 if len(self.A) else from_y - 100.0
        return float(_lower(float(x), float(from_y), float(z), float(radius), float(floor_y), *self._args()))

    # ---- files
    def save(self, path):
        np.savez_compressed(path, A=self.A, B=self.B, C=self.C, owner=self.owner.astype(np.int32), rail=self.rail,
                            tubes=self.tubes, entities=json.dumps(self.entities, ensure_ascii=False), meta=json.dumps(self.meta, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "Mesh":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["A"], z["B"], z["C"], z["owner"], z["rail"], z["tubes"], json.loads(str(z["entities"])), json.loads(str(z["meta"])))


def convex_hull(points, eps: float = 1e-4):
    """The convex hull of a small set of points (a body's collision hull is the hull of its mesh's vertices, whatever
    faces the mesh has): (vertices (n, 3), triangles (m, 3) as indices into them, wound outwards). Every plane through
    three of the points that has all of them on one side is a face; a face with more than three points on it is cut
    into a fan. Meant for a few dozen points."""
    P = np.unique(np.round(np.asarray(points, float), 6), axis=0)
    n = len(P)
    if n > 64:
        raise ValueError("convex_hull is for a few dozen points, not %d" % n)
    planes = []
    for i in range(n):
        for j in range(i + 1, n):
            K = np.arange(j + 1, n)
            if not len(K):
                continue
            N = np.cross(P[j] - P[i], P[K] - P[i])                   # one plane per third point
            ln = np.linalg.norm(N, axis=1)
            ok = ln > 1e-9
            N, K = N[ok] / ln[ok, None], K[ok]
            S = (P - P[i]) @ N.T                                       # (n, planes): every point's side
            out = (S <= eps).all(0)
            inn = (S >= -eps).all(0)
            for q in np.flatnonzero(out | inn):
                nrm = N[q] if out[q] else -N[q]
                d = float(nrm @ P[i])
                if not any(np.allclose(nrm, e[0], atol=1e-3) and abs(d - e[1]) < 1e-3 for e in planes):
                    planes.append((nrm, d))
    tris = []
    for nrm, d in planes:
        on = np.flatnonzero(np.abs(P @ nrm - d) < eps * 10)
        c = P[on].mean(0)
        u = P[on[0]] - c
        u /= np.linalg.norm(u)
        v = np.cross(nrm, u)
        ang = np.arctan2((P[on] - c) @ v, (P[on] - c) @ u)
        ring = on[np.argsort(ang)]
        for q in range(1, len(ring) - 1):
            t = [int(ring[0]), int(ring[q]), int(ring[q + 1])]
            if np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]) @ nrm < 0:
                t[1], t[2] = t[2], t[1]
            tris.append(t)
    return P, np.array(tris, np.int64).reshape(-1, 3)


def vertical_candidates(mesh, centre, reach):
    """Triangle indices in an XZ footprint, at any height, from the mesh grid."""
    ox, oz, cell, nx, nz, offsets, items = mesh._grid
    lo = np.floor((centre[[0, 2]] - reach - (ox, oz)) / cell).astype(int)
    hi = np.floor((centre[[0, 2]] + reach - (ox, oz)) / cell).astype(int)
    ids = [items[offsets[i * nz + j]:offsets[i * nz + j + 1]]
           for i in range(max(lo[0], 0), min(hi[0] + 1, nx))
           for j in range(max(lo[1], 0), min(hi[1] + 1, nz))]
    return np.unique(np.concatenate(ids)) if ids else np.empty(0, int)


def _sphere_top(tri, n, centre, radius):
    """Top of the triangle inflated by a sphere, on the vertical line through centre."""
    # Face interior: at tangency the centre is one radius along the upward normal.
    y = tri[0, 1] + (radius + (tri[0] - centre)[[0, 2]] @ n[[0, 2]]) / n[1]
    p = np.array([centre[0], y, centre[2]]) - radius * n
    q = _closest(*p, *tri.ravel())
    if np.linalg.norm(p - q[:3]) < 1e-7:
        return y, n[1]
    # Edge/vertex: maximize edge_y + sqrt(r^2 - horizontal_distance^2) over the segment.
    best = (-np.inf, 0.)
    for a, b in zip(tri, np.roll(tri, -1, axis=0)):
        d = (b - a)[[0, 2]]
        v = (centre - a)[[0, 2]]
        length = np.linalg.norm(d)
        if length < 1e-12:
            h2 = radius**2 - v @ v
            edge_y = max(a[1], b[1])
        else:
            proj = v @ (d / length)
            span2 = radius**2 - (v @ v - proj**2)
            if span2 <= 0.:
                continue
            span = np.sqrt(span2)
            lo, hi = max(0., proj - span), min(length, proj + span)
            if lo > hi:
                continue
            slope = (b[1] - a[1]) / length
            u = np.clip(proj + span * slope / np.hypot(1., slope), lo, hi)
            edge_y = a[1] + u * slope
            h2 = span2 - (u - proj)**2
        if h2 > 1e-12:
            h = np.sqrt(h2)
            best = max(best, (edge_y + h, h / radius))
    return best


def _box_span_y(tri, n, centre, axes, half):
    """Interval of centre heights with triangle/OBB overlap, by the separating axis theorem."""
    edges = np.roll(tri, -1, axis=0) - tri
    dirs = np.vstack((axes, n, np.cross(edges[:, None, :], axes).reshape(-1, 3)))
    lengths = np.linalg.norm(dirs, axis=1)
    dirs = dirs[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    dirs *= np.where(dirs[:, 1:2] < 0, -1., 1.)
    d = (tri - [centre[0], 0., centre[2]]) @ dirs.T
    r = half @ np.abs(axes @ dirs.T)
    lo, hi = d.min(0) - r, d.max(0) + r
    up = dirs[:, 1] > 1e-10
    if np.any(lo[~up] > 1e-8) or np.any(hi[~up] < -1e-8):
        return np.inf, -np.inf
    return np.max(lo[up] / dirs[up, 1]), np.min(hi[up] / dirs[up, 1])


def vertical_support(tris, friction, centre, *, radius=None, axes=None, half=None):
    """Highest upward contact on a downward sweep: (centre height, normal Y, surface friction), or None.

    Count a surface already intersecting the placed shape as contact at the start. Pure side grazes and
    vertical faces do not support weight. Triangle winding is immaterial for the two-sided collision mesh.
    """
    best = None
    reach = radius if radius is not None else np.asarray(half) @ abs(axes[:, [0, 2]])
    xz = tris[:, :, [0, 2]] - centre[[0, 2]]
    sel = np.flatnonzero(np.all(xz.min(1) <= reach + 1e-8, axis=1) & np.all(xz.max(1) >= -reach - 1e-8, axis=1))
    for i in sel:
        t = tris[i]
        n = np.cross(t[1] - t[0], t[2] - t[0])
        if abs(n[1]) < 1e-10:
            continue
        n *= np.sign(n[1]) / np.linalg.norm(n)
        if radius is None:
            lo, y = _box_span_y(t, n, centre, axes, half)
            if lo > y + 1e-8 or lo > centre[1] + 1e-8:
                continue
            ny = n[1]
        else:
            y, ny = _sphere_top(t, n, centre, radius)
            if y > centre[1] + 1e-8:
                delta = centre - _closest(*centre, *t.ravel())[:3]
                if delta @ delta > radius**2 + 1e-8:
                    continue
        if not np.isfinite(y):
            continue
        hit = (min(float(centre[1]), float(y)), float(ny), float(friction[i]))
        if best is None or hit[0] > best[0] + 1e-8 or (abs(hit[0] - best[0]) <= 1e-8 and hit[1:] > best[1:]):
            best = hit
    return best
