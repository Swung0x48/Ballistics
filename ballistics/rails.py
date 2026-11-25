"""Rails: from the rail triangles to tube axes, and from tube axes to the lanes a ball can ride.

The collision shape of a rail is its triangles, and contacts are computed on those (`ballistics.geometry`). This
module recovers what the triangles are a tessellation of, for everything that needs to know where a rail runs:

* `fit_tubes`: every rail is a tube whose cross-section is a polygon with its corners on a circle. The corners of
  one ring are found together, the circle through them gives the ring's centre and the tube's radius, and the
  ring centres joined along the mesh are the axis. The result is axis segments with a radius.
* `polylines`: the axis segments welded into runs between junctions and ends.
* `lanes`: two parallel tubes a ball fits between are a double rail, and the ball's centre runs above the middle
  of them; any other tube is a single rail with the ball's centre straight above it. Thick tubes (0.35) are the
  rails the levels are built to be ridden on; thin ones (0.25) are posts, guard rails and trim, and near-vertical
  pieces are posts."""
from __future__ import annotations

import math

import numpy as np

from .geometry import BALL_R

# Where these come from: MEASURED on level 1's rails, or ASSUMED (a working threshold, not checked against anything).
THIN_R = 0.3      # MEASURED: the game's tubes come in two radii, 0.247 and 0.351; this only has to lie between them
POST = 0.9        # ASSUMED: a segment steeper than this (|dy| / length) is a post
PAIR_MIN, PAIR_MAX = 1.5, 5.5   # ASSUMED bounds; MEASURED: every double rail of level 1 has its two axes 3.80 apart
SPACING = 1.0     # a choice, not a fact: lane points are this far apart
# That thick tubes are the rails meant to be ridden and thin ones are guard rails and trim is a READING of level 1's
# layout, not something the game states; a ball can physically ride either.


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


def _circle(P):
    """Least-squares circle through points that lie in a plane: (centre, radius, rms distance of the points from the
    circle), or None when the points are too few or in a line."""
    if len(P) < 3:
        return None
    c0 = P.mean(0)
    _, sv, vt = np.linalg.svd(P - c0)
    if sv[1] < 1e-3:
        return None
    x, y = (P - c0) @ vt[0], (P - c0) @ vt[1]
    sol = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)[0]
    rad = math.sqrt(max(float(sol[2] + sol[0] ** 2 + sol[1] ** 2), 0.0))
    rms = float(np.sqrt(np.mean((np.hypot(x - sol[0], y - sol[1]) - rad) ** 2)))
    return c0 + sol[0] * vt[0] + sol[1] * vt[1], rad, rms, vt[2]


def _radius_events(v, n, lo, hi, tol):
    """Radius intervals with invariant adjacency of normal-offset vertices."""
    from scipy.spatial import cKDTree
    pairs = cKDTree(v).query_pairs(tol+2*max(abs(lo),abs(hi)), output_type='ndarray')
    d = v[pairs[:,0]]-v[pairs[:,1]]
    w = n[pairs[:,0]]-n[pairs[:,1]]
    den = np.einsum('ij,ij->i',w,w)
    use = den>1.e-15
    d,w,den = d[use],w[use],den[use]
    root = np.einsum('ij,ij->i',d,w)/den
    rem = d-root[:,None]*w
    span = (tol**2-np.einsum('ij,ij->i',rem,rem))/den
    span,root = span[span>0],root[span>0]
    limits = np.r_[root-np.sqrt(span),root+np.sqrt(span)]
    events = np.unique(np.r_[lo,limits[(limits>lo)&(limits<hi)],hi])
    return (events[1:]+events[:-1])*.5


def fit_tubes(A, B, C, radius_bounds=(.20,.45), tol: float = 0.06, gather: float = 0.15):
    """Tube axis segments of rail triangles. Returns (segments (S, 7): x0 y0 z0 x1 y1 z1 radius, info dict).

    A rail's cross-section is a polygon whose corners lie on a circle (in the game an octagon, or part of one where
    the underside is left open). Steps, per connected piece of mesh:

    1. which vertices form a ring: moved back along their normals by a shared radius, the corners of one ring
       fall together; stragglers (corners on an open rim or a mitred joint, whose normals lean) join the ring
       they land next to;
    2. the ring's centre and radius: the circle through its corners, which is exact for a polygon inscribed in a
       circle whether or not the ring is complete; the piece's radius is the median over its good rings;
    3. the axis: ring centres joined wherever the mesh has an edge from one ring to the next."""
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
    segs, pieces, loose, rings, corners = [], 0, 0, 0, {}
    for c in np.unique(comp):
        idx = np.flatnonzero(comp == c)
        if len(idx) < 6:
            loose += len(idx)
            continue
        pieces += 1
        v, n = V[idx], N[idx]
        best = None
        for sign in (1.0, -1.0):                      # the triangles may be wound either way
            for r in _radius_events(v, sign*n, *radius_bounds, tol):
                pts, _, counts = _collapse(v, sign * n, float(r), tol)
                if best is None or len(pts) < best[0]:      # the right radius leaves the fewest gathered points
                    best = (len(pts), sign, float(r))
        _, sign, r0 = best
        pts, node, counts = _collapse(v, sign * n, r0, tol)
        # corners whose normals lean (an open rim, a mitred joint) land apart: gather those among themselves with a
        # wider reach first, so that a ring made only of such corners is still found
        small = np.flatnonzero(counts[node] < 3)
        if len(small) >= 3:
            again = _cluster((v - sign * r0 * n)[small], gather)
            node[small] = len(counts) + again
            counts = np.bincount(node)
            pts = np.zeros((len(counts), 3))
            np.add.at(pts, node, v - sign * r0 * n)
            pts = pts / np.maximum(counts, 1)[:, None]
        # stragglers join the nearest ring they land next to
        big = np.flatnonzero(counts >= 3)
        if len(big):
            for g in np.flatnonzero(counts < 3):
                d = np.linalg.norm(pts[big] - pts[g], axis=1)
                k = int(d.argmin())
                if d[k] <= gather:
                    node[node == g] = big[k]
            counts = np.bincount(node, minlength=len(counts))
        # a corner still on its own belongs to the ring whose circle it lies on (same plane, same distance from the centre)
        left = np.flatnonzero(counts[node] < 3)
        if len(left):
            circles = [(g, _circle(v[node == g])) for g in np.flatnonzero(counts >= 3)]
            circles = [(g, c) for g, c in circles if c is not None and c[2] < 0.03]
            for i in left:
                for g, (cc, rad, _, axis) in circles:
                    d = v[i] - cc
                    off = float(d @ axis)
                    if abs(off) < 0.03 and abs(float(np.linalg.norm(d - off * axis)) - rad) < 0.03:
                        node[i] = g
                        break
            counts = np.bincount(node, minlength=len(counts))
        centre = np.full((len(counts), 3), np.nan)
        good = []
        for g in np.flatnonzero(counts >= 3):
            fit = _circle(v[node == g])
            if fit is not None and fit[2] < 0.03 and 0.1 < fit[1] < 0.8:
                centre[g] = fit[0]
                corners[int(counts[g])] = corners.get(int(counts[g]), 0) + 1
                if counts[g] >= 5 and fit[2] < 0.01:
                    good.append(fit[1])
            else:
                centre[g] = pts[g]                    # not a clean ring: keep where its vertices landed
        r = float(np.median(good)) if good else r0
        rings += int((counts >= 3).sum())
        local = {int(g): k for k, g in enumerate(idx)}
        seen = set()
        f = F[np.isin(F[:, 0], idx)]
        for a, b in np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]):
            p, q = int(node[local[int(a)]]), int(node[local[int(b)]])
            if p == q or counts[p] < 3 or counts[q] < 3:      # a ring has at least three corners; caps do not
                continue
            seen.add((min(p, q), max(p, q)))
        # an edge that skips a ring lying on it (a long triangle across a short ring spacing) is not a segment of its
        # own: longest first, an edge goes when two shorter ones that are still there cover the same stretch
        span = {e: float(np.linalg.norm(centre[e[0]] - centre[e[1]])) for e in seen}
        alive = set(seen)
        near: dict = {}
        for p, q in seen:
            near.setdefault(p, set()).add(q)
            near.setdefault(q, set()).add(p)
        for p, q in sorted(seen, key=lambda e: -span[e]):
            for m in near[p] & near[q]:
                e1, e2 = (min(p, m), max(p, m)), (min(m, q), max(m, q))
                if e1 in alive and e2 in alive and span[e1] + span[e2] < span[(p, q)] * 1.02 + 1e-6:
                    alive.discard((p, q))
                    break
        for p, q in sorted(alive):
            if span[(p, q)] > 1e-6:
                segs.append([*centre[p], *centre[q], r])
    segs = np.array(segs, np.float64).reshape(-1, 7)
    return segs, dict(pieces=pieces, rings=rings, segments=len(segs), radii=sorted({round(float(x), 3) for x in segs[:, 6]}) if len(segs) else [],
                      corners_per_ring=dict(sorted(corners.items())), loose_vertices=loose)


# ---- polylines and lanes
def length(line) -> float:
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def resample(line, spacing: float = SPACING):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(2, int(round(s[-1] / spacing)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, line[:, k]) for k in range(3)], axis=1)


def polylines(seg, weld: float = 0.35, *, sampled=True) -> list:
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
    return [resample(l) if sampled else l for l in out if length(l) >= 2.0]


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


def _arc(line):
    return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]


def _at(line,s,station):
    return np.column_stack([np.interp(station,s,line[:,i]) for i in range(3)])


def straight_pieces(line):
    """Remove only duplicate points and roundoff-collinear forward joints."""
    out=[]
    for p in line:
        if out and np.linalg.norm(p-out[-1])<=1.e-9:continue
        while len(out)>1:
            a,b=out[-1]-out[-2],p-out[-1]
            if a@b<=0. or np.linalg.norm(np.cross(a,b))>1.e-7*np.linalg.norm(a)*np.linalg.norm(b):break
            out.pop()
        out.append(p)
    return np.asarray(out,dtype=float).reshape(-1,3)


def _mesh_run(line, n, start, stop):
    """A pairing run identifies enclosing *mesh* pieces, not grid cuts."""
    s=_arc(line)
    a,b=sorted((s[-1]*start/(n-1),s[-1]*stop/(n-1)))
    i=max(0,int(np.searchsorted(s,a+1.e-8,side='right'))-1)
    j=min(len(s)-1,int(np.searchsorted(s,b-1.e-8,side='left')))
    points=line[i:j+1]
    return points[::-1] if start>stop else points


def _overlap_start(a,b):
    """Trim an overhanging tube at the other tube's initial cross section."""
    d,e=a[1]-a[0],b[1]-b[0]
    n=d/np.linalg.norm(d)+e/np.linalg.norm(e)
    where=(a-b[0])@n
    if where[0]>=0:return a
    for j in range(len(where)-1):
        if where[j]<0.<=where[j+1]:
            t=-where[j]/(where[j+1]-where[j])
            return straight_pieces(np.vstack((a[j]+t*(a[j+1]-a[j]),a[j+1:])))
    return a  # no transverse crossing on this run; leave it to mesh support checks


def paired_pieces(a,b,ball,tube):
    """Pair mesh rings on a shared arc parameter, subdividing only at ring events.

    ASSUMED: when the two tubes have different knot counts, use the same
    relative arc length between the overlap boundaries. On parallel straight
    runs this is the shared cross-section coordinate. Treat the centre line
    between supported ring pairs as straight, not a fitted circular bend.
    """
    a,b=straight_pieces(a),straight_pieces(b)
    if len(a)<2 or len(b)<2:return np.empty((0,3))
    a,b=_overlap_start(a,b),_overlap_start(b,a)
    if len(a)<2 or len(b)<2:return np.empty((0,3))
    a,b=_overlap_start(a[::-1],b[::-1])[::-1],_overlap_start(b[::-1],a[::-1])[::-1]
    if len(a)<2 or len(b)<2:return np.empty((0,3))
    u,v=_arc(a),_arc(b)
    u/=u[-1];v/=v[-1]
    if len(a)!=len(b):
        knots=np.union1d(u,v)
        a,b=_at(a,u,knots),_at(b,v,knots)
    gauge=np.linalg.norm(b-a,axis=1)
    rise=np.sqrt(np.maximum(.05,(ball+tube)**2-(gauge*.5)**2))
    return straight_pieces((a+b)*.5 + np.c_[np.zeros(len(rise)),rise,np.zeros(len(rise))])


def _lanes(seg, tube: float, ball: float) -> list:
    mesh_lines = polylines(seg,sampled=False)
    lines = [resample(p) for p in mesh_lines]
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
            raw_a=_mesh_run(mesh_lines[i],len(A),a,b)
            raw_b=_mesh_run(mesh_lines[j],len(B),int(near[a]),int(near[b]))
            track=paired_pieces(raw_a,raw_b,ball,tube)
            out.append(dict(points=mid + np.c_[np.zeros(len(idx)), rise, np.zeros(len(idx))], segments=track,
                            double=True, gauge=float(np.median(2 * half)), axes=(raw_a,raw_b)))
            free[i][idx] = False
            free[j][near[idx]] = False
    for i, A in enumerate(lines):
        for a, b in _runs(free[i], 5):
            raw=_mesh_run(mesh_lines[i],len(A),a,b)
            track=straight_pieces(raw+[0.,ball+tube,0.])
            out.append(dict(points=A[a:b + 1] + [0.0, ball + tube, 0.0], segments=track,double=False,gauge=0.0))
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


def project_pieces(points,pos):
    """Exact nearest projection on a finite chain of straight mesh pieces."""
    d=np.diff(points,axis=0)
    w=pos-points[:-1]
    square=np.einsum('ij,ij->i',d,d)
    f=np.clip(np.einsum('ij,ij->i',w,d)/square,0.,1.)
    q=points[:-1]+f[:,None]*d
    j=int(np.argmin(np.einsum('ij,ij->i',q-pos,q-pos)))
    return j,q[j]


def physical_lanes(g,mesh):
    """Rail geometry is independent of the mapping lattice."""
    if all('segments' in lane for lane in g.lanes):
        return [lane['segments'] for lane in g.lanes]
    if len(mesh.tubes):
        rebuilt=lanes(mesh.tubes)
        if len(rebuilt)!=len(g.lanes):raise ValueError('rail lane identity changed; rebuild the coarse graph')
        for a,b in zip(g.lanes,rebuilt):
            if not np.array_equal(a['points'],b['points']):raise ValueError('rail mapping grid changed; rebuild the coarse graph')
        return [lane['segments'] for lane in rebuilt]
    # Hand-supplied synthetic lanes in tests have explicit literal vertices.
    return [straight_pieces(lane['points']) for lane in g.lanes]


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


def axis_bends(tubes):
    """Circumcurvature from mesh ring centres, independent of coarse spacing."""
    from collections import defaultdict
    from scipy.spatial import cKDTree

    around = defaultdict(list)
    for a,b in zip(tubes[:,:3],tubes[:,3:6]):
        around[tuple(np.round(a,5))].append(b-a)
        around[tuple(np.round(b,5))].append(a-b)
    bends = []
    for p, ds in around.items():
        if len(ds)!=2:
            continue  # no unique continuation at a branch
        a,b = ds
        la,lb,span = np.linalg.norm(a),np.linalg.norm(b),np.linalg.norm(b-a)
        if min(la,lb,span)<=1.e-7:
            continue
        k = 2*np.linalg.norm(np.cross(a,b))/(la*lb*span)
        bends.append((np.asarray(p),-a/la,b/lb,float(k)))
    return (cKDTree([b[0] for b in bends]), bends) if bends else (None, ())
