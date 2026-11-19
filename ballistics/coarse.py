"""The coarse level: where a ball can be at rest on the map, and how those places connect.

This level only proposes. It answers "which way round" quickly, on a 1-unit sampling of the exact geometry; whether
a proposed way can actually be driven is decided by the physics kernel on the triangles themselves, and nothing
computed here goes into a proof.

Nodes are ball-centre positions: places where a ball lowered onto the map comes to rest on a face that can hold
it (sampled every `STEP` units, every level of the map), and the points along the rail lanes. Edges say a ball can
get from one node to the next: rolling between neighbouring rest places, along a lane, on and off a lane's ends,
dropping off a ledge, and flying from a ledge or a lane end (`flights`).

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
DROP_MAX = 30.0       # a ledge drop no deeper than this is proposed
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
            y = r - 2.0 * radius - 0.3          # under what it rests on: a ball's height further down at least
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
KEEP_OUT = {"P_Trafo_Stone": TRIGGER_R + 0.3, "P_Trafo_Paper": TRIGGER_R + 0.3}
LOOSE = ("P_Ball_",)       # placed objects a rolling ball knocks aside (ASSUMED): not treated as obstacles, only reported


def keep_out(placements) -> list:
    """[(x, y, z, distance)] for the placed objects a wood ball has to stay clear of."""
    out = []
    for p in placements:
        d = KEEP_OUT.get(p["type"])
        if d is not None:
            out.append((*p["position"], d))
    return out


def build(mesh: Mesh, avoid=(), radius: float = BALL_R, log=lambda *a: None) -> Graph:
    """avoid: [(x, y, z, distance)] places the ball must stay out of (see `keep_out`)."""
    loose = np.array([bool(e.get("placed")) and str(e.get("type", "")).startswith(LOOSE) for e in mesh.entities], bool)
    if loose.any():
        mesh = mesh.subset(~loose[mesh.owner])           # loose balls are not obstacles here
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
    rest = _rest_columns(X.astype(np.float64), Z.astype(np.float64), top, bottom, radius, 4, *mesh._args())
    pos, cell = [], {}
    for k in range(len(X)):
        for l in range(rest.shape[1]):
            y = rest[k, l]
            if np.isnan(y):
                break
            c = mesh.contacts((X[k], y, Z[k]), radius, 0.02)
            if not c or mesh.rail[c[0].triangle] or max(q.normal[1] for q in c) < WALK_NY:
                continue                                  # resting on a rail from the side, on a wall top's edge only, ...
            if any(mesh.placed[q.triangle] for q in c):
                continue                                  # on or against a placed object: not a place to count on
            if any(math.hypot(X[k] - t[0], Z[k] - t[2]) <= t[3] and abs(y - 2.0 - t[1]) <= 5.0 for t in avoid):
                continue
            cell.setdefault((int(ii[k]), int(jj[k])), []).append(len(pos))
            pos.append((X[k], y, Z[k]))
    n_floor = len(pos)
    log("rest places: %d over %d columns" % (n_floor, len(X)))
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
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = cell.get((i + di, j + dj))
            run = STEP * math.hypot(di, dj)
            for a_ids, b_ids in ((ids, other or ()), (other or (), ids)):
                for u in a_ids:
                    level = [v for v in b_ids if abs(P[v, 1] - P[u, 1]) <= ROLL_SLOPE * run]
                    for v in level:
                        if a_ids is ids:
                            link(u, v, "roll", both=True)
                    if not level:                         # no floor at this height next door: the ball drops onto the first one below
                        under = [v for v in b_ids if -DROP_MAX <= P[v, 1] - P[u, 1] < 0]
                        if under:
                            link(u, max(under, key=lambda v: P[v, 1]), "drop")
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
            below = np.flatnonzero((d <= 3.5) & (dy < -1.6) & (dy >= -DROP_MAX))
            if len(below):                                # off the end of a lane the ball lands on the first floor below it
                link(end, int(below[np.argmax(F[below, 1] - 0.3 * d[below])]), "drop")
    return Graph(P, np.array(kind), np.array(lane_of), adj, lns, dict(rest_places=n_floor, lane_points=len(P) - n_floor, edge_places=int(edge_node.sum()),
                                                                    edge_node=edge_node))


# ---- flights
LAUNCH_SPEEDS = (10.0, 15.9, 22.5)      # READ: 15.9 and 22.5 are the wood ball's rolling plateaus with one and two keys; 10 is a working value
FLIGHT_TICKS = 220
AIR_DAMP = 1.0 - 0.9 / 66.0             # READ: the wood ball's linear damping per tick
GRAVITY_TICK = 20.0 / 66.0


@njit(cache=True)
def _fly(P0, D, speeds, radius, lam, g, ticks, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """Free flights with no keys held from each start along each direction at each speed, until the ball touches
    something. Returns rows (start index, direction index, speed index, landing x y z, ticks, vertical speed)."""
    out = np.empty((len(P0) * len(D) * len(speeds), 8))
    n = 0
    for i in range(len(P0)):
        for d in range(len(D)):
            for s in range(len(speeds)):
                x, y, z = P0[i, 0], P0[i, 1] + 0.05, P0[i, 2]
                vx, vy, vz = speeds[s] * D[d, 0], 0.0, speeds[s] * D[d, 1]
                left = False
                for t in range(ticks):
                    x += vx / 66.0; y += vy / 66.0; z += vz / 66.0
                    vx *= lam; vz *= lam
                    vy = vy * lam - g
                    dist, _tri = _distance_near(x, y, z, radius + 0.6, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                    if dist > radius + 0.3:
                        left = True                          # clear of the floor it started on
                    elif dist <= radius + 0.02:
                        if left and t > 6:
                            out[n, 0], out[n, 1], out[n, 2] = i, d, s
                            out[n, 3], out[n, 4], out[n, 5], out[n, 6], out[n, 7] = x, y, z, t + 1, vy
                            n += 1
                        break
                    if y < lo_y.min() - 20.0:
                        break
    return out[:n]


def add_flights(g: Graph, mesh: Mesh, radius: float = BALL_R, log=lambda *a: None) -> int:
    """Flight edges from ledges and lane ends: the ball leaves with one of LAUNCH_SPEEDS in one of 16 directions,
    no keys held, and the edge goes to the rest place where it comes down. Whether that speed and heading can be
    had at that ledge is not judged here."""
    n_floor = g.info["rest_places"]
    sites = np.flatnonzero(g.info["edge_node"][:n_floor])
    ends = np.array([c for l, c in enumerate(_lane_chains(g)) for c in (c[0], c[-1])], np.int64)
    sites = np.concatenate([sites, ends])
    ang = np.arange(16) * (math.pi / 8)
    D = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    rows = _fly(g.pos[sites], D, np.array(LAUNCH_SPEEDS), radius, AIR_DAMP, GRAVITY_TICK, FLIGHT_TICKS, *mesh._args())
    F = g.pos[:n_floor]
    grid: dict = {}
    for k in range(n_floor):
        grid.setdefault((int(round(F[k, 0] / STEP)), int(round(F[k, 2] / STEP))), []).append(k)
    best: dict = {}
    for r in rows:
        u = int(sites[int(r[0])])
        land = r[3:6]
        cand = [k for di in (-1, 0, 1) for dj in (-1, 0, 1) for k in grid.get((int(round(land[0] / STEP)) + di, int(round(land[2] / STEP)) + dj), ())
                if abs(F[k, 1] - land[1]) <= 0.8]
        if not cand:
            continue                                         # came down on a rail, a wall or a slope too steep to rest on
        v = min(cand, key=lambda k: (F[k, 0] - land[0]) ** 2 + (F[k, 2] - land[2]) ** 2)
        far = math.hypot(F[v, 0] - g.pos[u, 0], F[v, 2] - g.pos[u, 2])
        if far < 3.0:
            continue                                         # a hop onto the next cell is rolling, not a flight
        cost = far + 12.0 + 6.0 * int(r[2])                  # rolling is preferred, and slower launches over faster ones
        if (u, v) not in best or cost < best[(u, v)][0]:
            best[(u, v)] = (cost, float(LAUNCH_SPEEDS[int(r[2])]), float(ang[int(r[1])]), int(r[6]), far, float(g.pos[u, 1] - F[v, 1]))
    g.info["flights"] = {}
    for (u, v), (cost, speed, heading, ticks, far, fall) in best.items():
        g.adj[u].append((v, cost, "flight"))
        g.info["flights"][(u, v)] = dict(speed=speed, heading=heading, ticks=ticks, distance=far, fall=fall)
    log("flights: %d launches tried from %d ledges and lane ends, %d landed, %d flight edges" % (len(sites) * 16 * len(LAUNCH_SPEEDS), len(sites), len(rows), len(best)))
    return len(best)


def _lane_chains(g: Graph) -> list:
    out = []
    for li in range(len(g.lanes)):
        out.append(np.flatnonzero(g.lane == li))
    return [c for c in out if len(c)]
