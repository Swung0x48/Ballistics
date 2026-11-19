"""Rails: from the rail triangles to tube axes, and from tube axes to the lanes a ball can ride.

The collision shape of a rail is its triangles, and contacts are computed on those (`ballistics.geometry`). This
module recovers what the triangles are a tessellation of, for everything that needs to know where a rail runs:

* `fit_tubes`: every rail is a tube. The vertices of one cross-section ring all lie one tube radius away from the
  same point of the axis, along their outward normals; moving each vertex back along its normal by the right radius
  makes each ring collapse into one axis point. The radius is the one that collapses them best, found per connected
  piece of mesh. The result is axis segments with a radius.
* `polylines`: the axis segments welded into runs between junctions and ends.
* `lanes`: two parallel tubes a ball fits between are a double rail, and the ball's centre runs above the middle
  of them; any other tube is a single rail with the ball's centre straight above it. Thick tubes (0.35) are the
  rails the levels are built to be ridden on; thin ones (0.25) are posts, guard rails and trim, and near-vertical
  pieces are posts."""
from __future__ import annotations

import math

import numpy as np

from .geometry import BALL_R

THIN_R = 0.3      # tubes thinner than this are guard rails, posts and trim
POST = 0.9        # a segment steeper than this (|dy| / length) is a post
PAIR_MIN, PAIR_MAX = 1.5, 5.5   # distance between the two axes of a double rail
SPACING = 1.0     # lane points are this far apart


# ---- tubes
def _weld(P, tol: float = 1e-3):
    """Merge coincident points: (unique points, index of each input point)."""
    key = np.round(np.asarray(P, np.float64) / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return np.asarray(P, np.float64)[first], inv.reshape(-1)


def _components(n: int, edges) -> np.ndarray:
    parent = np.arange(n)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _cluster(P, tol: float) -> np.ndarray:
    """Label points so that points closer than tol (directly or through others) share a label."""
    cell = np.floor(P / tol).astype(np.int64)
    grid: dict = {}
    for i, c in enumerate(map(tuple, cell)):
        grid.setdefault(c, []).append(i)
    edges = []
    for (cx, cy, cz), members in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = grid.get((cx + dx, cy + dy, cz + dz))
                    if other is None:
                        continue
                    for i in members:
                        for j in other:
                            if j > i and np.linalg.norm(P[i] - P[j]) < tol:
                                edges.append((i, j))
    root = _components(len(P), edges)
    _, label = np.unique(root, return_inverse=True)
    return label.reshape(-1)


def _collapse(V, N, radius: float, tol: float):
    """Axis points V - radius * N gathered where they fall together: (points, index per vertex, vertices per point)."""
    A = V - radius * N
    inv = _cluster(A, tol)
    counts = np.bincount(inv)
    pts = np.zeros((len(counts), 3))
    np.add.at(pts, inv, A)
    return pts / counts[:, None], inv, counts


def fit_tubes(A, B, C, radii=np.arange(0.20, 0.45, 0.01), tol: float = 0.06):
    """Tube axis segments of rail triangles. Returns (segments (S, 7): x0 y0 z0 x1 y1 z1 radius, info dict)."""
    A, B, C = (np.asarray(x, np.float64).reshape(-1, 3) for x in (A, B, C))
    if not len(A):
        return np.zeros((0, 7)), dict(pieces=0)
    V, inv = _weld(np.concatenate([A, B, C]))
    F = inv.reshape(3, len(A)).T
    fn = np.cross(B - A, C - A)                       # area-weighted face normals
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1)
    N = N / np.where(ln > 0, ln, 1)[:, None]
    comp = _components(len(V), np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]))
    segs, pieces, loose = [], 0, 0
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        if len(idx) < 6:
            loose += len(idx)
            continue
        pieces += 1
        v, n = V[idx], N[idx]
        best = None
        for sign in (1.0, -1.0):                      # the triangles may be wound either way
            for r in radii:
                pts, _, counts = _collapse(v, sign * n, float(r), tol)
                score = len(pts)                      # the right radius leaves the fewest axis points
                if best is None or score < best[0]:
                    best = (score, sign, float(r))
        _, sign, r = best
        _, node, counts = _collapse(v, sign * n, r, tol)
        # the collapse only tells which vertices form a ring. The radius is then the one that brings each ring's
        # vertices closest together when moved back along their normals (least squares over the piece's rings;
        # a rail's cross-section need not be a full circle, so the ring's centroid is not its centre), and the
        # axis point of a ring is where its vertices land
        vm, nm = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
        np.add.at(vm, node, v)
        np.add.at(nm, node, sign * n)
        vm, nm = vm / counts[:, None], nm / counts[:, None]
        ring = counts[node] >= 3
        dv, dn = v[ring] - vm[node[ring]], sign * n[ring] - nm[node[ring]]
        den = float((dn * dn).sum())
        if den > 1e-9:
            r = float((dv * dn).sum() / den)
        pts = vm - r * nm
        local = {int(g): k for k, g in enumerate(idx)}
        seen = set()
        f = F[np.isin(F[:, 0], idx)]
        for a, b in np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]):
            p, q = int(node[local[int(a)]]), int(node[local[int(b)]])
            if p == q or counts[p] < 3 or counts[q] < 3:      # a ring has at least three vertices; caps and seams do not
                continue
            key = (min(p, q), max(p, q))
            if key not in seen:
                seen.add(key)
                segs.append([*pts[key[0]], *pts[key[1]], r])
    segs = np.array(segs, np.float64).reshape(-1, 7)
    return segs, dict(pieces=pieces, segments=len(segs), radii=sorted({round(float(x), 3) for x in segs[:, 6]}) if len(segs) else [], loose_vertices=loose)


# ---- polylines and lanes
def length(line) -> float:
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def resample(line, spacing: float = SPACING):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(2, int(round(s[-1] / spacing)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def polylines(seg, weld: float = 0.35) -> list:
    """Weld the segment end points and split the graph into simple runs between junctions and tips (loops included)."""
    seg = np.asarray(seg, np.float64)
    pts = np.r_[seg[:, :3], seg[:, 3:6]]
    n = len(seg)
    grid: dict = {}
    for i, p in enumerate(pts):
        grid.setdefault(tuple(np.floor(p / 0.5).astype(int)), []).append(i)
    edges = []
    for i, p in enumerate(pts):
        g = np.floor(p / 0.5).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for m in grid.get((g[0] + dx, g[1] + dy, g[2] + dz), ()):
                        if m > i and np.linalg.norm(pts[m] - p) < weld:
                            edges.append((i, m))
    node = _components(len(pts), edges)
    pos: dict = {}
    for i, k in enumerate(node):
        pos.setdefault(int(k), []).append(pts[i])
    pos = {k: np.mean(v, axis=0) for k, v in pos.items()}
    adj: dict = {k: [] for k in pos}
    for s in range(n):
        a, b = int(node[s]), int(node[s + n])
        if a != b:
            adj[a].append((b, s))
            adj[b].append((a, s))
    used, out = set(), []

    def walk(a, b, s):
        line = [pos[a], pos[b]]
        used.add(s)
        cur = b
        while len(adj[cur]) == 2:
            nxt = [(c, e) for c, e in adj[cur] if e not in used]
            if not nxt:
                break
            c, e = nxt[0]
            used.add(e)
            line.append(pos[c])
            cur = c
        return np.array(line)

    for a in adj:
        if len(adj[a]) != 2:
            for b, s in adj[a]:
                if s not in used:
                    out.append(walk(a, b, s))
    for a in adj:                                     # closed loops
        for b, s in adj[a]:
            if s not in used:
                out.append(walk(a, b, s))
    return [resample(l) for l in out if length(l) >= 2.0]


def _tangents(P):
    t = np.gradient(P, axis=0)
    return t / np.maximum(1e-9, np.linalg.norm(t, axis=1, keepdims=True))


def _runs(mask, least: int):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if j - i + 1 >= least:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _lanes(seg, tube: float, ball: float) -> list:
    lines = polylines(seg)
    free = [np.ones(len(l), bool) for l in lines]
    out, cand = [], []
    for i, A in enumerate(lines):
        for j in range(i + 1, len(lines)):
            B = lines[j]
            if (A.min(0) - PAIR_MAX > B.max(0)).any() or (B.min(0) - PAIR_MAX > A.max(0)).any():
                continue
            D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
            ok = (D.min(1) >= PAIR_MIN) & (D.min(1) <= PAIR_MAX)
            if ok.sum() >= 4:
                cand.append((int(ok.sum()), i, j))
    for _, i, j in sorted(cand, reverse=True):
        A, B = lines[i], lines[j]
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        near = D.argmin(1)
        parallel = np.abs(np.sum(_tangents(A) * _tangents(B)[near], axis=1)) >= 0.8       # two rails that cross or meet are not a pair
        ok = (D.min(1) >= PAIR_MIN) & (D.min(1) <= PAIR_MAX) & free[i] & free[j][near] & (np.abs(A[:, 1] - B[near, 1]) <= 1.2) & parallel
        for a, b in _runs(ok, 4):
            idx = np.arange(a, b + 1)
            mid = (A[idx] + B[near[idx]]) / 2
            half = D[idx, near[idx]] / 2
            rise = np.sqrt(np.maximum(0.05, (ball + tube) ** 2 - half ** 2))               # the ball rests on both tubes
            out.append(dict(points=mid + np.c_[np.zeros(len(idx)), rise, np.zeros(len(idx))], double=True, gauge=float(np.median(2 * half))))
            free[i][idx] = False
            free[j][near[idx]] = False
    for i, A in enumerate(lines):
        for a, b in _runs(free[i], 5):
            out.append(dict(points=A[a:b + 1] + [0.0, ball + tube, 0.0], double=False, gauge=0.0))
    return out


def lanes(tubes, ball: float = BALL_R) -> list:
    """[{points (n, 3) ball-centre positions along the lane, double, thin, gauge, radius}] for tube segments
    (x0 y0 z0 x1 y1 z1 radius). Posts are dropped; thick and thin tubes are paired separately, so a ride rail never
    pairs with the guard rail beside it."""
    tubes = np.asarray(tubes, np.float64).reshape(-1, 7)
    if not len(tubes):
        return []
    d = tubes[:, 3:6] - tubes[:, :3]
    tubes = tubes[np.abs(d[:, 1]) <= POST * np.maximum(1e-9, np.linalg.norm(d, axis=1))]
    out = []
    for thin in (False, True):
        part = tubes[(tubes[:, 6] < THIN_R) == thin]
        if len(part):
            r = float(np.median(part[:, 6]))
            out += [dict(l, thin=thin, radius=r) for l in _lanes(part, r, ball)]
    return out


def climb(points, window: float = 4.0) -> tuple:
    """(steepest uphill slope as rise over run over about `window` of arc length, total ascent) riding the points
    in order."""
    P = np.asarray(points, float)
    if len(P) < 2:
        return 0.0, 0.0
    dy = np.diff(P[:, 1])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    h = np.r_[0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 2])))]
    j = np.minimum(len(P) - 1, np.searchsorted(s, s + window))
    run = np.maximum(h[j] - h, 0.5 * window)
    return float(np.max((P[j, 1] - P[:, 1]) / run)), float(dy[dy > 0].sum())
