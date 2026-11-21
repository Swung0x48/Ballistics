"""The coarse level: where a ball can be at rest on the map, and how those places connect.

This level only proposes. It answers "which way round" quickly, on a 1-unit sampling of the exact geometry; whether
a proposed way can actually be driven is decided by the physics kernel on the triangles themselves, and nothing
computed here goes into a proof.

Nodes are ball-centre positions: places where a ball lowered onto the map comes to rest on a face that can hold
it (sampled every `STEP` units, every level of the map), and the points along the rail lanes. Edges say a ball can
get from one node to the next: rolling between neighbouring rest places (down small steps too), along a lane, on
and off a lane's ends. Falls and flights are not edges of positions: where a ball comes down depends on the speed
it leaves with, so they are made by running the physics kernel (launch.py).

Every threshold here is a working choice for proposing ways, not a statement about the game: ASSUMED throughout,
except where a comment says otherwise."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import rails
from .geometry import BALL_R, Mesh, _distance as _distance_near, _lower

STEP = 1.0            # spacing of the rest samples
WALK_NY = 0.6         # a face this upright (normal y) or more can hold a resting ball
ROLL_SLOPE = 0.55     # rise over run between neighbouring rest places that still counts as rolling up (READ: a wood ball holds 0.75 from a standstill; kept well inside)
STEP_UP = 1.5         # MEASURED once: the recorded level 1 run takes a wood ball up a step 1.45 high (floor at -4.15, ramp edge at -2.70) beside the sector 2 ramp, arriving about 15 across with keys held, and the physics kernel replays that passage. Higher steps are not proposed
SNAP = 0.75           # a sample this close to the border of a floor is moved onto it (just over half the diagonal of a sample cell, so no floor is missed)
STEP_REACH = 3.6      # below a step the ball is held no closer than its radius to the riser, above it from the edge on; plus the sample spacing and SNAP
SMALL_DROP = 1.5      # a step down no deeper than this is rolled down in contact; a deeper fall is a launch (launch.py), which depends on the speed the ball leaves with
TRIGGER_R = 4.3       # READ (game script and wiki): a transformer acts on a ball of another kind within this distance


@njit(cache=True)
def _rest_columns(X, Z, top, floor_y, radius, levels, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """For each column: the centre heights at which a lowered ball rests, from the top down (nan-padded)."""
    out = np.full((len(X), levels), np.nan)
    for k in range(len(X)):
        y = top
        for l in range(levels):
            r = _lower(X[k], y, Z[k], radius, floor_y, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
            if np.isnan(r):
                break
            out[k, l] = r
            # go on down through what it touched, in small steps, to where the ball is free again (the next place
            # can be right below a grazed edge, or a ball's height and more below a floor)
            y = r - 0.05
            while y > floor_y:
                d, _t = _distance_near(X[k], y, Z[k], radius + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                if d - radius > 1e-6:
                    break
                y -= 0.05
    return out


@dataclass
class Graph:
    pos: np.ndarray                 # (n, 3) ball-centre positions
    kind: np.ndarray                # (n,) 0 floor, 1 lane point
    lane: np.ndarray                # (n,) lane index for lane points, -1 otherwise
    adj: list                       # per node: [(neighbour, cost, edge kind)]
    lanes: list
    info: dict = field(default_factory=dict)

    def nearest(self, p, reach: float = 6.0, floor_only: bool = True) -> int:
        d = np.linalg.norm(self.pos - np.asarray(p, float), axis=1)
        if floor_only:
            d = np.where(self.kind == 0, d, np.inf)
        k = int(d.argmin())
        return k if d[k] <= reach else -1

    def route(self, a: int, b: int):
        """Cheapest way from node a to node b: (node list, cost), or (None, the reached node nearest to b)."""
        dist, par, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            if u == b:
                break
            for v, c, _ in self.adj[u]:
                if d + c < dist.get(v, 1e18):
                    dist[v], par[v] = d + c, u
                    heapq.heappush(heap, (d + c, v))
        if b not in dist:
            seen = np.array(list(dist))
            return None, int(seen[np.linalg.norm(self.pos[seen] - self.pos[b], axis=1).argmin()])
        path = [b]
        while path[-1] != a:
            path.append(par[path[-1]])
        return path[::-1], dist[b]

    def edge_kind(self, u: int, v: int) -> str:
        return next((k for w, _, k in self.adj[u] if w == v), "?")


# What stands on the map besides the static geometry. A static analysis cannot move it, so it keeps out of it:
# ASSUMED keep-out distances around each placed object (its real shape is not used), except the transformers' own
# trigger distance. Objects that do not hinder a wood ball are not listed (score and life balls, the wood transformer).
# Loose balls (P_Ball_*) are not kept out of: a rolling ball knocks them aside. That is ASSUMED too, and a way that
# passes one is reported as passing it.
KEEP_OUT = {"P_Trafo_Paper": TRIGGER_R + 0.3}      # the paper ball is not solved: a route does not go into a paper transformer
LOOSE = ("P_Ball_",)       # loose balls: obstacles like everything placed, unless the caller takes them out with `without_loose`


def without_loose(mesh: Mesh) -> Mesh:
    """The mesh with the loose balls taken out: what the level is if a rolling ball knocks them aside (ASSUMED)."""
    loose = np.array([bool(e.get("placed")) and str(e.get("type", "")).startswith(LOOSE) for e in mesh.entities], bool)
    return mesh.subset(~loose[mesh.owner]) if loose.any() else mesh


def without_bodiless(mesh: Mesh) -> Mesh:
    """The mesh without what has no rigid body in the game but was exported as if it had (mesh files made before
    the export left them out): a fan's rotor and box, and a placed object's marker that the level file also lists
    as floor (one in the 13 levels: level 3's P_Modul_18_04, MEASURED: the ball goes through it in three of the
    original engine's records)."""
    from .export import SKIP_TEMPLATES
    drop = np.array([(bool(e.get("placed")) and str(e.get("type", "")).startswith(SKIP_TEMPLATES))
                     or (not e.get("placed") and any(str(g).startswith("P_") for g in e.get("groups", ()))) for e in mesh.entities], bool)
    keep = ~drop[mesh.owner] if len(drop) else np.ones(len(mesh.A), bool)
    if keep.all():
        return mesh
    out = mesh.subset(keep)
    out.meta["bodiless_dropped"] = int((~keep).sum())       # a coarse graph cached before this is of another mesh (`solve.graph_for`)
    return out


FIXED_TYPES = ("P_Dome",)      # placed objects that are fixed rigid bodies: plain geometry while their sector is current


def in_sector(mesh: Mesh, sector=None) -> Mesh:
    """The mesh as it is while `sector` is the current one, as far as the placed objects that never move go (the
    dome, P_Dome): those of the sector become plain fixed geometry, with the friction and elasticity the catalogue
    has for them; those of other sectors are taken out (the game takes their bodies away). sector None: all of
    them are there. Everything else placed stays as it is."""
    hit = [i for i, e in enumerate(mesh.entities) if e.get("placed") and str(e.get("type", "")).startswith(FIXED_TYPES)]
    if not hit:
        return mesh
    from . import things
    cat = things.load()
    ents = [dict(e) for e in mesh.entities]
    drop = np.zeros(len(ents), bool)
    for i in hit:
        e = ents[i]
        if sector is not None and e.get("sector") != sector:
            drop[i] = True
            continue
        body = cat.of_type(e["type"]).bodies[0]
        e.update(placed=False, fixed_body=True, friction=float(body["friction"]["v"]), elasticity=float(body["elasticity"]["v"]))
    keep = ~drop[mesh.owner]
    meta = dict(mesh.meta, fixed_bodies=list(FIXED_TYPES))
    return Mesh(mesh.A[keep], mesh.B[keep], mesh.C[keep], mesh.owner[keep], mesh.rail[keep], mesh.tubes, ents, meta)


def keep_out(placements) -> list:
    """[(x, y, z, distance)] for the placed objects a wood ball has to stay clear of."""
    out = []
    for p in placements:
        d = KEEP_OUT.get(p["type"])
        if d is not None:
            out.append((*p["position"], d))
    return out


_ROUND = np.stack([np.cos(np.arange(36) * math.pi / 18), np.sin(np.arange(36) * math.pi / 18)], axis=1)


def _held(mesh: Mesh, contacts) -> bool:
    """Whether what the ball touches can carry it. Statics, not a tuned rule:
    * a face no steeper than WALK_NY under it; or
    * a ridge: an edge or corner where two such faces meet, touched from between them (the push of the contact lies
      between the two faces' own normals); or
    * edges and corners on opposite sides (the horizontal parts of their pushes surround the centre).
    A ball that touches the border of a floor and hangs over it is not held: it would roll off."""
    if any(q.feature == "face" and q.normal[1] >= WALK_NY for q in contacts):
        return True
    groups: list = []
    for q in contacts:
        for grp in groups:
            if np.abs(grp[0].point - q.point).max() < 1e-3:
                grp.append(q)
                break
        else:
            groups.append([q])
    for grp in groups:
        N = mesh.N[[q.triangle for q in grp]]
        N = N[N[:, 1] >= WALK_NY]                         # the faces at this point that can carry a ball
        if len(N) < 2:
            continue
        C = np.clip(N @ N.T, -1.0, 1.0)
        a, b = np.unravel_index(int(C.argmin()), C.shape)                 # the two that lean furthest apart
        if C[a, b] > 1.0 - 1e-9:
            continue                                      # one flat floor: its border, not a ridge
        to = np.arccos(np.clip(N[[a, b]] @ grp[0].normal, -1.0, 1.0))
        if float(to.sum()) <= float(np.arccos(C[a, b])) + 0.02:
            return True                                   # pushed from between the two faces: on the ridge
    up = [q.normal for q in contacts if q.normal[1] > 0.05]
    if len(up) < 2:
        return False
    H = np.array([(n[0], n[2]) for n in up])
    return bool(((H @ _ROUND.T).max(axis=0) >= -1e-3).all())


def build(mesh: Mesh, avoid=(), radius: float = BALL_R, log=lambda *a: None) -> Graph:
    """avoid: [(x, y, z, distance)] places the ball must stay out of (see `keep_out`)."""
    floor = ~mesh.rail
    lo = np.minimum(np.minimum(mesh.A, mesh.B), mesh.C)[floor]
    hi = np.maximum(np.maximum(mesh.A, mesh.B), mesh.C)[floor]
    x0, z0 = math.floor(lo[:, 0].min()) - 1, math.floor(lo[:, 2].min()) - 1
    nx, nz = int(math.ceil(hi[:, 0].max()) - x0) + 2, int(math.ceil(hi[:, 2].max()) - z0) + 2
    occ = np.zeros((nx, nz), bool)                       # columns that have some floor triangle over or near them
    for a, b in zip(lo, hi):
        i0, i1 = int((a[0] - x0) / STEP) - 1, int((b[0] - x0) / STEP) + 1
        j0, j1 = int((a[2] - z0) / STEP) - 1, int((b[2] - z0) / STEP) + 1
        occ[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1] = True
    ii, jj = np.nonzero(occ)
    X, Z = x0 + ii * STEP, z0 + jj * STEP
    top, bottom = float(mesh._hi_y.max()) + radius + 1.0, float(mesh._lo_y.min()) - 5.0
    rest = _rest_columns(X.astype(np.float64), Z.astype(np.float64), top, bottom, radius, 6, *mesh._args())
    pos, cell = [], {}
    snapped = 0
    for k in range(len(X)):
        for l in range(rest.shape[1]):
            y = rest[k, l]
            if np.isnan(y):
                break
            x, z = float(X[k]), float(Z[k])
            c = mesh.contacts((x, y, z), radius, 0.02)
            if not c or c[0].gap < -0.02 or mesh.rail[c[0].triangle]:
                continue                                  # nothing there, squeezed in, or on a rail from the side
            if not _held(mesh, c):
                # hanging over the border of a floor. A floor narrower than the sample spacing, or one that runs
                # askew to it, would be missed altogether, so a sample within SNAP of a border moves onto the border
                q = max(c, key=lambda q: q.normal[1])
                h = math.hypot(q.normal[0], q.normal[2])
                if h * radius > SNAP or h < 1e-9:
                    continue
                x, z = float(q.point[0] - 0.02 * q.normal[0] / h), float(q.point[2] - 0.02 * q.normal[2] / h)
                y = mesh.rest(x, z, y + 0.5)
                c = mesh.contacts((x, y, z), radius, 0.02)
                if not c or c[0].gap < -0.02 or mesh.rail[c[0].triangle] or not _held(mesh, c):
                    continue
                snapped += 1
            if any(mesh.placed[q.triangle] for q in c):
                continue                                  # on or against a placed object: not a place to count on
            if any(math.hypot(x - t[0], z - t[2]) <= t[3] and abs(y - 2.0 - t[1]) <= 5.0 for t in avoid):
                continue
            cell.setdefault((int(ii[k]), int(jj[k])), []).append(len(pos))
            pos.append((x, y, z))
    n_floor = len(pos)
    log("rest places: %d over %d columns, %d of them moved onto a border" % (n_floor, len(X), snapped))
    lns = rails.lanes(mesh.tubes, radius)
    kind, lane_of, chains = [0] * n_floor, [-1] * n_floor, []
    for li, l in enumerate(lns):
        ids = list(range(len(pos), len(pos) + len(l["points"])))
        pos.extend(map(tuple, l["points"]))
        kind.extend([1] * len(ids))
        lane_of.extend([li] * len(ids))
        chains.append(ids)
    P = np.array(pos, np.float64).reshape(-1, 3)
    adj = [[] for _ in range(len(P))]

    def link(u, v, k, both=False, cost=None):
        c = float(np.linalg.norm(P[u] - P[v])) if cost is None else cost
        adj[u].append((v, c, k))
        if both:
            adj[v].append((u, c, k))

    # rolling and ledge drops between neighbouring rest places
    edge_node = np.zeros(len(P), bool)
    for (i, j), ids in cell.items():
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)):
            run = STEP * math.hypot(di, dj)
            for u in ids:
                level = [v for v in cell.get((i + di, j + dj), ()) if abs(P[v, 1] - P[u, 1]) <= ROLL_SLOPE * run]
                for v in level:
                    link(u, v, "roll")
                if level:
                    continue
                # no floor at this height next door: the ball drops onto the first floor below, which begins up to a
                # ball's width further out (next to a wall the ball cannot be held closer than its radius)
                for k in (1, 2, 3):
                    there = cell.get((i + k * di, j + k * dj), ())
                    if any(abs(P[v, 1] - P[u, 1]) <= 0.3 for v in there):
                        break                             # floor at this height again: a slot, not a ledge
                    under = [v for v in there if -SMALL_DROP <= P[v, 1] - P[u, 1] < 0]
                    if under:
                        link(u, max(under, key=lambda v: P[v, 1]), "drop")
                        break
        for u in ids:                                    # a rest place with a missing neighbour is at an edge
            if any((i + di, j + dj) not in cell for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                edge_node[u] = True
    # a placed object between two neighbouring rest places blocks the roll: the ball half way must clear it
    if mesh.placed.any():
        solid = mesh.subset(mesh.placed)
        for u in range(n_floor):
            keep = []
            for v, c, k in adj[u]:
                if k == "roll" and solid.distance((P[u] + P[v]) / 2, radius + 1.0)[0] < radius - 0.05:
                    continue
                keep.append((v, c, k))
            adj[u] = keep
    # lanes: along, and on / off at the ends
    F = P[:n_floor]
    for li, ids in enumerate(chains):
        for u, v in zip(ids, ids[1:]):
            link(u, v, "rail", both=True)
        thin = lns[li]["thin"] or not lns[li]["double"]
        for end in (ids[0], ids[-1]):
            if not n_floor:
                continue
            d = np.hypot(F[:, 0] - P[end, 0], F[:, 2] - P[end, 2])
            dy = F[:, 1] - P[end, 1]
            near = np.flatnonzero((d <= 3.0) & (np.abs(dy) <= 1.6))
            for v in near[np.argsort(d[near])[:4]]:
                link(end, int(v), "rail end", both=True, cost=float(d[v]) + (6.0 if thin else 0.0))      # riding one rail is the last resort
    return Graph(P, np.array(kind), np.array(lane_of), adj, lns, dict(rest_places=n_floor, lane_points=len(P) - n_floor, edge_places=int(edge_node.sum()),
                                                                    edge_node=edge_node))


# ---- flights
def add_steps(g: Graph, mesh: Mesh, radius: float = BALL_R, log=lambda *a: None) -> int:
    """Step edges: from a rest place up to one within STEP_REACH and at most STEP_UP higher, where rolling does
    not link them and nothing stands higher than the upper floor in between. A step is taken with speed; how much,
    and whether the ball can have it there, is for the fine level."""
    n = g.info["rest_places"]
    F = g.pos[:n]
    grid: dict = {}
    for k in range(n):
        grid.setdefault((int(round(F[k, 0] / STEP)), int(round(F[k, 2] / STEP))), []).append(k)
    solid = mesh
    added = 0
    for (i, j), ids in grid.items():
        for di in range(-4, 5):
            for dj in range(-4, 5):
                other = grid.get((i + di, j + dj))
                if not other or (di, dj) == (0, 0):
                    continue
                run = STEP * math.hypot(di, dj)
                for u in ids:
                    have = {e[0] for e in g.adj[u]}
                    for v in other:
                        rise = F[v, 1] - F[u, 1]
                        if v in have or not (ROLL_SLOPE * run < rise <= STEP_UP) or math.hypot(F[v, 0] - F[u, 0], F[v, 2] - F[u, 2]) > STEP_REACH:
                            continue
                        top = F[v, 1] + 0.05
                        if any(solid.distance((F[u, 0] + f * (F[v, 0] - F[u, 0]), top, F[u, 2] + f * (F[v, 2] - F[u, 2])), radius + 1.0)[0] < radius - 0.05
                               for f in (0.25, 0.5, 0.75)):
                            continue                          # a kerb, a wall or a placed object stands in the way
                        g.adj[u].append((v, run + 10.0, "step"))
                        added += 1
    log("steps: %d step edges of up to %.1f" % (added, STEP_UP))
    return added


def _lane_chains(g: Graph) -> list:
    out = []
    for li in range(len(g.lanes)):
        out.append(np.flatnonzero(g.lane == li))
    return [c for c in out if len(c)]


GAP_COST, BRIDGE_COST = 300.0, 1000.0      # a link is taken only where the coarse level's own edges do not get there (or only by a detour longer than this)


class Relaxed:
    """The coarse graph plus generous links where it does not connect: 'gap' between any two places within `near`
    that have no edge, and 'bridge' between ledges and lane points within `far` (dearer, more so upwards). A plan
    over it is the coarse level's ways joined by the links it would need: each link on a plan is a break."""

    def __init__(self, g: Graph, near: float = 4.0, far: float = 25.0):
        P = g.pos
        n = len(P)
        self.g = g
        fwd = [list(a) for a in g.adj]
        have = [set(e[0] for e in a) for a in g.adj]
        grid: dict = {}
        for i, p in enumerate(P):
            grid.setdefault((int(p[0] // near), int(p[1] // near), int(p[2] // near)), []).append(i)
        for (a, b, c), ids in grid.items():
            other = [k for da in (-1, 0, 1) for db in (-1, 0, 1) for dc in (-1, 0, 1) for k in grid.get((a + da, b + db, c + dc), ())]
            Q = P[other]
            for i in ids:
                d = np.sqrt(((Q - P[i]) ** 2).sum(1))
                for k in np.flatnonzero((d <= near) & (d > 0)):
                    v = other[k]
                    if v not in have[i]:
                        fwd[i].append((v, GAP_COST + 2.0 * float(d[k]) + 6.0 * max(0.0, float(P[v, 1] - P[i, 1])), "gap"))
        edge = np.zeros(n, bool)
        nf = g.info["rest_places"]
        edge[:nf] = g.info["edge_node"][:nf]
        edge[nf:] = True
        E = np.flatnonzero(edge)
        for lo in range(0, len(E), 512):
            chunk = E[lo:lo + 512]
            D = np.sqrt(((P[chunk][:, None, :] - P[E][None, :, :]) ** 2).sum(2))
            for r, i in enumerate(chunk):
                for k in np.flatnonzero((D[r] <= far) & (D[r] > near)):
                    v = int(E[k])
                    if v not in have[i]:
                        fwd[i].append((v, BRIDGE_COST + 3.0 * float(D[r, k]) + 6.0 * max(0.0, float(P[v, 1] - P[i, 1])), "bridge"))
        self.fwd = fwd
        self.back = [[] for _ in range(n)]
        for u in range(n):
            for v, c, _k in fwd[u]:
                self.back[v].append((u, c))

    def togo(self, goal: int) -> np.ndarray:
        """How far each place is from the goal over this graph."""
        togo = np.full(len(self.fwd), np.inf)
        togo[goal] = 0.0
        heap = [(0.0, goal)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > togo[u]:
                continue
            for v, c in self.back[u]:
                if d + c < togo[v]:
                    togo[v] = d + c
                    heapq.heappush(heap, (d + c, v))
        return togo

    def plan(self, u: int, togo: np.ndarray):
        """The way from place u to the goal: (places, kind of the edge leaving each but the last)."""
        nodes, kinds = [int(u)], []
        while togo[nodes[-1]] > 0 and np.isfinite(togo[nodes[-1]]):
            v, c, k = min(self.fwd[nodes[-1]], key=lambda e: e[1] + togo[e[0]])
            nodes.append(int(v))
            kinds.append(k)
        return nodes, kinds
