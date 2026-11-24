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
KEEP_OUT: dict = {}      # nothing placed is kept out of now (a paper transformer was, until the paper ball was solved)
LOOSE = ("P_Ball_", "P_Box", "P_Modul_34", "P_Modul_01", "P_Modul_25", "P_Modul_30")       # loose balls, boxes, the falling rock, the gate, the push board, the seesaw: rigid bodies of their own (`ballistics.props`), not fixed shapes


def without_loose(mesh: Mesh) -> Mesh:
    """The mesh with the loose balls' shapes taken out: they are bodies of their own (`ballistics.props`), or, in the
    diagnosis mode, not there at all."""
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
FIXED_PARTS = ("P_Modul_01_Filler",)   # parts of mechanisms that are fixed rigid bodies the ball rolls on: the gate's filler plate (READ)


def in_sector(mesh: Mesh, sector=None) -> Mesh:
    """The mesh as it is while `sector` is the current one, as far as the placed objects that never move go (the
    dome, P_Dome): those of the sector become plain fixed geometry, with the friction and elasticity the catalogue
    has for them; those of other sectors are taken out (the game takes their bodies away). sector None: all of
    them are there. Everything else placed stays as it is."""
    hit = [i for i, e in enumerate(mesh.entities) if e.get("placed") and (str(e.get("type", "")).startswith(FIXED_TYPES) or e.get("name") in FIXED_PARTS)]
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
        bodies = cat.of_type(e["type"]).bodies
        body = next((b for b in bodies if b["name"] == e.get("name")), bodies[0])
        e.update(placed=False, fixed_body=True, friction=float(body["friction"]["v"]), elasticity=float(body["elasticity"]["v"]))
    keep = ~drop[mesh.owner]
    meta = dict(mesh.meta, fixed_bodies=list(FIXED_TYPES) + list(FIXED_PARTS))
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

    def with_edges(self, edges) -> "Relaxed":
        """The same graph with these edges (u, v, cost, kind) added: a gate pushed open (`solve.gate_edges`)."""
        out = Relaxed.__new__(Relaxed)
        out.g, out.fwd, out.back = self.g, list(self.fwd), list(self.back)
        for u, v, c, k in edges:
            out.fwd[u] = out.fwd[u] + [(v, c, k)]
            out.back[v] = out.back[v] + [(u, c)]
        return out

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


# ---------------- the coarse level with the mechanisms' states
WOOD, STONE, PAPER = 0, 1, 2                 # the ball's kind, as a state of the coarse level
KIND_INDEX = {"wood": WOOD, "stone": STONE, "paper": PAPER}
GATE_THROUGH, GATE_OPERATE, BOARD_PUSH, BOARD_OVER, SEESAW, CORK = 1, 2, 3, 4, 5, 6
ROCK_OVER, T_OVER, T_CLEAR, ROCK_PUSH, T_PUSH = 7, 8, 9, 10, 11
FLOAT_ROW = 12
FLAP_OVER, FLAP_PUSH = 13, 14
FLOAT_KIND_BITS = len(KIND_INDEX)
FLOAT_KIND_MASK = (1 << FLOAT_KIND_BITS) - 1
FLOAT_THROUGH, FLOAT_MINUS, FLOAT_PLUS, FLOAT_END = range(4)


def float_side(n, direction, allowed):
    return ((n * 4 + direction) << FLOAT_KIND_BITS) | allowed


def float_exit(side):
    """(Box number from entry, exit direction); through edges use the full row length."""
    return divmod(side >> FLOAT_KIND_BITS, 4)
STATE_OVER = (ROCK_OVER, T_OVER, T_CLEAR)
OPERATE = (GATE_OPERATE, ROCK_PUSH, T_PUSH)
_INF = 1e300


@njit(cache=True)
def _state_togo(n, M, bptr, bsrc, bcost, bek, bmask, bclear, tk, tcost, lptr, lsrc, lcost, lk, lmask, lclear, cptr, csrc, ccost, cmech, ctype, cside, goal, lock):
    """How far each (state, place) is from the goal, which counts in any state. A state is kind * 2^M + bits (bit m:
    mechanism m's state). Backwards over: plain edges (u -> v; v inside a transformer's range makes the ball its
    kind there, at tcost unless it is that kind already; a link that passes through a range, bek, likewise; a link
    with bmask or bclear needs those bits set or clear respectively), the paper ball's lifts, and the mechanisms'
    edges, each with the state it needs and the state it leaves (see `StateGraph`)."""
    S = 3 * (1 << M)
    togo = np.full((S, n), _INF)
    heap = [(0.0, 0, 0)]
    heap.pop()
    for s in range(S):
        togo[s, goal] = 0.0
        heapq.heappush(heap, (0.0, s, goal))
    while len(heap) > 0:
        d, s, v = heapq.heappop(heap)
        if d > togo[s, v]:
            continue
        k = s >> M
        bits = s & ((1 << M) - 1)
        for e in range(bptr[v], bptr[v + 1]):
            if (bits & bmask[e]) != bmask[e] or bits & bclear[e]:
                continue
            t = tk[v] if tk[v] >= 0 else bek[e]
            u = bsrc[e]
            nd = d + bcost[e]
            if t < 0:
                if nd < togo[s, u]:
                    togo[s, u] = nd
                    heapq.heappush(heap, (nd, s, u))
            elif k == t:
                for k2 in range(3):
                    s2 = (k2 << M) | bits
                    nd2 = nd + (0.0 if k2 == t else tcost)
                    if nd2 < togo[s2, u]:
                        togo[s2, u] = nd2
                        heapq.heappush(heap, (nd2, s2, u))
        # a lift is a paper ball's; it lands as the kind of a transformer's range it lands in (or ends in on the way, lk)
        s2 = (2 << M) | bits
        for e in range(lptr[v], lptr[v + 1]):
            tl = tk[v] if tk[v] >= 0 else (lk[e] if lk[e] >= 0 else 2)
            if k != tl or (bits & lmask[e]) != lmask[e] or bits & lclear[e]:
                continue
            u = lsrc[e]
            nd = d + lcost[e] + (tcost if tl != 2 else 0.0)
            if nd < togo[s2, u]:
                togo[s2, u] = nd
                heapq.heappush(heap, (nd, s2, u))
        for e in range(cptr[v], cptr[v + 1]):
            u = csrc[e]
            nd = d + ccost[e]
            m = cmech[e]
            bit = (bits >> m) & 1
            typ = ctype[e]
            preds = (-1, -1)
            if typ == 1:                                   # through an open gate
                if bit == 1:
                    preds = (s, -1)
            elif typ == 2:                                 # a stone ball pushes the gate open
                if bit == 1 and k == 1 and not (lock & (1 << m)):
                    preds = (s & ~(1 << m), -1)
            elif typ == 3:                                 # a wood or stone ball pushes the board over, going across it
                if bit == 1 and k != 2:
                    preds = (s, s & ~(1 << m) if not (lock & (1 << m)) else -1)
            elif typ == 4:                                 # across a board that lies
                if bit == 1:
                    preds = (s, -1)
            elif typ == 5:                                 # up the seesaw from its low end (cside: which end is low): a stone ball tips it
                side = cside[e]
                if k == 1 and not (lock & (1 << m)):
                    if bit == 1 - side:
                        preds = ((s & ~(1 << m)) | (side << m), -1)
                elif bit == side:
                    preds = (s, -1)
            elif typ == 6:                                 # over a cork bridge: a wood or paper ball (a stone one breaks it and falls)
                if k != 1:
                    preds = (s, -1)
            elif typ == ROCK_OVER or typ == T_OVER or typ == T_CLEAR or typ == FLOAT_ROW:
                if typ == FLOAT_ROW or bit == 1:
                    t = tk[v]
                    if t < 0:
                        if typ != FLOAT_ROW or cside[e] & (1 << k):
                            preds = (s, -1)
                    elif k == t:
                        for k2 in range(3):
                            if typ == FLOAT_ROW and not (cside[e] & (1 << k2)):
                                continue
                            s2 = (k2 << M) | bits
                            nd2 = nd + (0.0 if k2 == t else tcost)
                            if nd2 < togo[s2, u]:
                                togo[s2, u] = nd2
                                heapq.heappush(heap, (nd2, s2, u))
            elif typ == FLAP_OVER or typ == FLAP_PUSH:
                side = cside[e]  # required state before crossing
                push = typ == FLAP_PUSH
                if bit == (1 - side if push else side) and not (push and lock & (1 << m)):
                    before = bits ^ (1 << m) if push else bits
                    t = tk[v]
                    for k2 in range(3):
                        if (push and k2 == PAPER) or k != (t if t >= 0 else k2):
                            continue
                        s2 = (k2 << M) | before
                        nd2 = nd + (tcost if k2 != k else 0.0)
                        if nd2 < togo[s2, u]:
                            togo[s2, u] = nd2
                            heapq.heappush(heap, (nd2, s2, u))
            elif typ == ROCK_PUSH or typ == T_PUSH:
                if bit == 1 and not (lock & (1 << m)) and (k == STONE or (typ == T_PUSH and k == WOOD)):
                    preds = (s & ~(1 << m), -1)
            for s2 in preds:
                if s2 >= 0 and nd < togo[s2, u]:
                    togo[s2, u] = nd
                    heapq.heappush(heap, (nd, s2, u))
    return togo


def _csr(n, edges):
    """Edges (src, dst, cost, *rest) as a CSR by dst: (ptr, src, cost, rest columns...)."""
    order = sorted(range(len(edges)), key=lambda i: edges[i][1])
    ptr = np.zeros(n + 1, np.int64)
    for i in order:
        ptr[edges[i][1] + 1] += 1
    ptr = np.cumsum(ptr)
    src = np.array([edges[i][0] for i in order], np.int64)
    cost = np.array([edges[i][2] for i in order], np.float64)
    rest = [np.array([edges[i][3 + j] for i in order], np.int64) for j in range(len(edges[0]) - 3)] if edges else []
    return ptr, src, cost, rest


class StateGraph:
    """The coarse level of one sector over states: the ball's kind (wood, stone, paper) and each mechanism's state
    (bit m of `bits`): a gate open, a push board down, a seesaw down at its other end. A state index is
    kind * 2^M + bits. Only the places of the sector's region take part.

    The edges: the relaxed graph's (for any state; one into a transformer's range of another kind makes the ball that
    kind, at the transformation's cost), the paper ball's lifts, and the mechanisms':
    * a gate (P_Modul_01): the edges it clears once open, only while it is open; at its push place an "operate"
      step by a stone ball opens it (a wood ball cannot open it all the way: MEASURED);
    * a push board (P_Modul_25): from the road in front of it across, a wood or stone ball pushes it over on the way
      (a paper ball cannot: READ); across it either way while it lies;
    * a seesaw (P_Modul_30): from its low end up and across, by any ball; a stone one tips it, so that the other end
      is low after (MEASURED).
    * falling rocks and T boards (region analysis only): a push changes their bit in place; crossing the new
      surface or the way a T-board foot clears needs the bit set. Existing blocked edges retain their own kinds,
      with guards {(u, v, edge kind): (required set bits, required clear bits)}.
    * two-way flaps (region analysis only): over the lane in the current pose, or push from the supported side
      and cross the other lane in the same move. The latter toggles the bit each time.
    mechs: [dict(name, type, ...edges)] as `solve` makes them. links: {(u, v): (kind
    index or -1, bits mask)} for the relaxed graph's links that pass through a transformer's range or a mechanism
    (`solve._link_rules`). lifts: (u, v, cost[, the kind index it ends as when that is a transformer's it enters on the
    way, not one v is in]). skip: (u, v) or (u, v, edge kind) edges of the relaxed graph omitted in this sector.
    bridges: (u, v, cost, name) over cork bridges (P_Modul_29), for a wood or paper ball. A bridge has no bit: a stone
    ball breaks it and falls with it (READ), and a broken bridge is whole again only once the ball is lost, so no ball
    that goes on ever finds it broken. floats: box float rows with a mask of ball kinds allowed at entry, no bit;
    transformation on arrival works like a plain edge."""

    def __init__(self, rel, g, region, trafo_kind, trafo_cost: float, lifts, mechs: list, operate_cost: float, links=None, skip=(), bridges=(), guards=None, floats=()):
        N = len(g.pos)
        self.g, self.N, self.mechs = g, N, list(mechs)
        self.M = len(self.mechs)
        self.S = 3 * (1 << self.M)
        ids = np.flatnonzero(region)
        self.glob = ids                                    # local -> global
        self.loc = np.full(N, -1, np.int64)
        self.loc[ids] = np.arange(len(ids))
        n = self.n = len(ids)
        loc = self.loc
        # plain edges (both ways of the relaxed graph): kept as python lists for the plan, as a CSR by target for togo
        links = links or {}
        skip = set(skip)
        self.fwd = [[] for _ in range(n)]
        self.guards = {}  # local (u, v, kind) -> mask of bits that must be clear; required bits live in fwd
        guards = guards or {}
        base, base_kind = [], []
        for gu in ids:
            u = int(loc[gu])
            for v_, c, k in rel.fwd[gu]:
                v = int(loc[v_])
                if v >= 0 and (int(gu), int(v_)) not in skip and (int(gu), int(v_), k) not in skip:
                    ek, mask = links.get((int(gu), int(v_)), (-1, 0)) if k in ("gap", "bridge") else (-1, 0)
                    on, off = guards.get((int(gu), int(v_), k), (0, 0))
                    mask |= on
                    if mask & off:
                        continue
                    if off:
                        self.guards[u, v, k] = off
                    self.fwd[u].append((v, float(c), k, int(ek), int(mask)))
                    base.append((u, v, float(c), int(ek), int(mask), off))
                    base_kind.append(k)
        self.tk = np.array([int(trafo_kind[gu]) for gu in ids], np.int64)
        self.tcost = float(trafo_cost)
        self.lfwd = [[] for _ in range(n)]
        lift = []
        self.lift_guards = {}
        for e in lifts:
            gu, gv, c = e[:3]
            lk = int(e[3]) if len(e) > 3 else -1
            u, v = int(loc[gu]), int(loc[gv])
            on, off = guards.get((gu, gv, "lift"), (0, 0))
            if u >= 0 and v >= 0 and (gu, gv, "lift") not in skip and not on & off:
                self.lfwd[u].append((v, float(c), lk))
                if on or off:
                    self.lift_guards[u, v] = (on, off)
                lift.append((u, v, float(c), lk, on, off))
        self.cfwd = [[] for _ in range(n)]
        cond, cond_kind = [], []
        for m, mc in enumerate(self.mechs):
            for typ, gu, gv, c, side, kind in mc["edges"]:
                u, v = int(loc[gu]), int(loc[gv])
                if u < 0 or v < 0:
                    continue
                c = float(c) + (operate_cost if typ in (*OPERATE, FLAP_PUSH) else 0.0)
                self.cfwd[u].append((v, c, kind, m, typ, side))
                cond.append((u, v, c, m, typ, side))
                cond_kind.append(kind)
        self.bridges = sorted({b[3] for b in bridges})
        for gu, gv, c, name in bridges:
            u, v = int(loc[gu]), int(loc[gv])
            if u < 0 or v < 0:
                continue
            j = self.bridges.index(name)                   # (kept where a mechanism's index goes; it is no bit)
            self.cfwd[u].append((v, float(c), "roll", j, CORK, 0))
            cond.append((u, v, float(c), j, CORK, 0))
            cond_kind.append("roll")
        self.floats = list(floats)
        self.float_roll_labels = None
        if any("departures" in row for row in self.floats):
            # Float exits are reduced using exactly the rolling regions used by the region analysis.
            from .regions import roll_labels
            self.float_roll_labels = roll_labels(self)
        for j, row in enumerate(self.floats):
            for u, v, c, side in self._float_edges(row):
                self.cfwd[u].append((v, c, "float_row", j, FLOAT_ROW, side))
                cond.append((u, v, c, j, FLOAT_ROW, side))
                cond_kind.append("float_row")
        self._base, self._lift, self._cond = base, lift, cond
        self._base_kind, self._cond_kind = base_kind, cond_kind
        self._dropped = set()
        self._arrays()

    def _float_edges(self, row):
        """Keep at most one exit per entry place, direction and destination region for each kind.

        Kind regions can split differently (and a landing can transform the ball). Combine masks only
        when they select the very same edge. Prefer a witness usable by more kinds, then lower cost.
        The old opposite-bank edges take precedence for the same kind and destination at the far end.
        """
        labs = self.float_roll_labels
        through = set()

        def target(v, k):
            t = int(self.tk[v]) if self.tk[v] >= 0 else k
            return t, int(labs[t, v])

        allowed = row["allowed"]
        for gu, gv, c in row["edges"]:
            u, v = int(self.loc[gu]), int(self.loc[gv])
            if u >= 0 and v >= 0:
                yield u, v, c, float_side(row["boxes"], FLOAT_THROUGH, allowed)
                if labs is not None:
                    through.update((u, k, target(v, k)) for k in range(FLOAT_KIND_BITS) if allowed & (1 << k))
        for banks, sign, candidates in row.get("departures", ()):
            # Costs differ between banks on this end only by their axial distance to the row.
            choices = {}
            for gv, c, side in candidates:
                v = int(self.loc[gv])
                n, direction = float_exit(side)
                mask = side & FLOAT_KIND_MASK
                rank = (-mask.bit_count(), c, v, n)
                for k in range(FLOAT_KIND_BITS):
                    if mask & (1 << k):
                        key = direction, k, target(v, k)
                        if key not in choices or rank < choices[key][0]:
                            choices[key] = rank, v, c, n
            for gu, x in banks:
                u = int(self.loc[gu])
                exits = {}
                for (direction, k, dst), (_, v, c, n) in choices.items():
                    if direction == FLOAT_END and (u, k, dst) in through:
                        continue
                    key = v, c, n, direction
                    exits[key] = exits.get(key, 0) | (1 << k)
                for (v, c, n, direction), mask in exits.items():
                    yield u, v, c - sign * x, float_side(n, direction, mask)

    def _arrays(self):
        n = self.n
        base = [e for e, k in zip(self._base, self._base_kind) if (e[0], e[1], k) not in self._dropped]
        self.bptr, self.bsrc, self.bcost, (self.bek, self.bmask, self.bclear) = _csr(n, base or [(0, 0, _INF, -1, 0, 0)])
        lift = self._lift or [(0, 0, _INF, -1, 0, 0)]
        self.lptr, self.lsrc, self.lcost, (self.lk, self.lmask, self.lclear) = _csr(n, lift)
        cond = self._cond or [(0, 0, _INF, 0, 0, 0)]
        self.cptr, self.csrc, self.ccost, (self.cmech, self.ctype, self.cside) = _csr(n, cond)

    def drop(self, gu: int, gv: int, kind: str):
        """Take an edge the plan took (gu -> gv, of that kind) out, wherever it is (a plain edge, or a mechanism's of that
        kind): the coarse level's own edges lead past it."""
        u, v = int(self.loc[gu]), int(self.loc[gv])
        if u < 0 or v < 0:
            return
        self.fwd[u] = [e for e in self.fwd[u] if not (e[0] == v and e[2] == kind)]
        self._dropped.add((u, v, kind))
        keep = [i for i, e in enumerate(self._cond) if not (e[0] == u and e[1] == v and self._cond_kind[i] == kind)]
        self._cond, self._cond_kind = [self._cond[i] for i in keep], [self._cond_kind[i] for i in keep]
        self.cfwd[u] = [e for e in self.cfwd[u] if not (e[0] == v and e[2] == kind)]
        self._arrays()

    def drop_mech_edge(self, m: int, typ: int):
        """Take a mechanism's edges of one type out (a step that was tried and did not change its state)."""
        keep = [i for i, e in enumerate(self._cond) if not (e[3] == m and e[4] == typ)]
        self._cond, self._cond_kind = [self._cond[i] for i in keep], [self._cond_kind[i] for i in keep]
        for u in range(self.n):
            self.cfwd[u] = [e for e in self.cfwd[u] if not (e[3] == m and e[4] == typ)]
        self._arrays()

    def togo(self, goal: int, lock: int = 0) -> np.ndarray:
        """(S, N): how far each place is from the goal in each state (inf outside the sector's region)."""
        out = np.full((self.S, self.N), np.inf)
        gl = int(self.loc[goal])
        if gl < 0:
            return out
        t = _state_togo(self.n, self.M, self.bptr, self.bsrc, self.bcost, self.bek, self.bmask, self.bclear, self.tk, self.tcost, self.lptr, self.lsrc, self.lcost, self.lk, self.lmask, self.lclear,
                        self.cptr, self.csrc, self.ccost, self.cmech, self.ctype, self.cside, gl, lock)
        t[t >= _INF / 2] = np.inf
        out[:, self.glob] = t
        return out

    def step(self, s: int, u: int, lock: int = 0):
        """Every way on from state s at local place u: [(state, local place, cost, edge kind)]."""
        M = self.M
        k, bits = s >> M, s & ((1 << M) - 1)
        out = []
        for v, c, kind, ek, mask in self.fwd[u]:
            if (bits & mask) != mask or bits & self.guards.get((u, v, kind), 0):
                continue
            t = self.tk[v] if self.tk[v] >= 0 else ek
            if t >= 0 and t != k:
                out.append(((int(t) << M) | bits, v, c + self.tcost, kind))
            else:
                out.append((s, v, c, kind))
        if k == PAPER:
            for v, c, lk in self.lfwd[u]:
                on, off = self.lift_guards.get((u, v), (0, 0))
                if bits & on != on or bits & off:
                    continue
                t = self.tk[v] if self.tk[v] >= 0 else lk
                if t >= 0 and t != PAPER:              # it lands in (or ends in, on the way) a transformer's range of another kind
                    out.append(((int(t) << M) | bits, v, c + self.tcost, "lift"))
                else:
                    out.append((s, v, c, "lift"))
        for v, c, kind, m, typ, side in self.cfwd[u]:
            bit = (bits >> m) & 1
            if typ == GATE_THROUGH and bit == 1:
                out.append((s, v, c, kind))
            elif typ in OPERATE and bit == 0 and not (lock & (1 << m)) and (k == STONE or (typ == T_PUSH and k == WOOD)):
                out.append((s | (1 << m), v, c, "operate" if typ == GATE_OPERATE else kind))
            elif typ == BOARD_PUSH and k != PAPER and (bit or not (lock & (1 << m))):
                out.append((s | (1 << m), v, c, kind))
            elif typ == BOARD_OVER and bit == 1:
                out.append((s, v, c, kind))
            elif typ == SEESAW and bit == side:
                out.append(((s & ~(1 << m)) | ((1 - side) << m) if k == STONE and not (lock & (1 << m)) else s, v, c, kind))
            elif typ in (FLAP_OVER, FLAP_PUSH) and bit == side:
                if typ == FLAP_PUSH and (k == PAPER or lock & (1 << m)):
                    continue
                after = bits ^ (1 << m) if typ == FLAP_PUSH else bits
                t = int(self.tk[v]) if self.tk[v] >= 0 else k
                out.append(((t << M) | after, v, c + (self.tcost if t != k else 0.0), kind))
            elif (typ in STATE_OVER and bit == 1) or (typ == FLOAT_ROW and side & (1 << k)):
                t = int(self.tk[v])
                if t >= 0 and t != k:
                    out.append(((t << M) | bits, v, c + self.tcost, kind))
                else:
                    out.append((s, v, c, kind))
            elif typ == CORK and k != STONE:
                out.append((s, v, c, "cork_bridge"))
        return out

    def plan(self, s: int, gu: int, togo: np.ndarray, most: int = 5000, lock: int = 0):
        """The way from global place gu in state s to the goal: (global places, the kind of each edge, the state at
        each place)."""
        u = int(self.loc[gu])
        if u < 0:
            return [int(gu)], [], [int(s)]
        T = togo[:, self.glob]
        nodes, kinds, states = [u], [], [int(s)]
        while T[states[-1], nodes[-1]] > 0 and np.isfinite(T[states[-1], nodes[-1]]) and len(nodes) < most:
            best = min(self.step(states[-1], nodes[-1], lock=lock), key=lambda e: e[2] + T[e[0], e[1]], default=None)
            if best is None or not np.isfinite(best[2] + T[best[0], best[1]]):
                break
            states.append(int(best[0]))
            nodes.append(int(best[1]))
            kinds.append(best[3])
        return [int(self.glob[x]) for x in nodes], kinds, states
