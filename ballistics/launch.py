"""Launch edges: where a ball that takes a run-up and rolls off a ledge or off the end of a rail comes down.

A ball does not leave a ledge at a speed of the planner's choosing, straight down or anywhere else: it leaves with
the speed and heading its run-up gave it. So these edges are not proposed from positions; each one is a run of the
physics kernel. The ball starts at rest at a place some way back from a ledge (or back along a rail from its end),
one arrow-key push is held towards the ledge, the ball rolls off, flies (with the key still held, or released) and
lands. The edge goes from the place where it started to the place where it landed, and carries the keys, the
take-off speed and the landing speed. Nothing placed on the map may be touched on the way.

The run-ups are a wood ball's, except where the run-up starts inside a transformer's range: a ball there is that
transformer's kind (and a wood ball would stop on the range at once), so it runs as that kind.

What these edges do not cover yet: a ball that arrives already moving (every run-up here starts from rest), leaving a
rail sideways before its end, and steering in the air other than holding or releasing the one key.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import fine
from . import physics as ph
from .coarse import STEP, Graph, _lane_chains
from .geometry import CONTACT_MARGIN
from .kernel import OK, World
from .takeoff import FOLLOW_SPEED, STOP_COST

RUNUPS = (0.0, 8.0, 16.0, 32.0)        # how far back the run-up starts (0: the ball tips over the ledge from rest)


def _push_towards(dx: float, dz: float) -> int:
    """Index in fine.PUSHES of the arrow-key push nearest to the direction (dx, dz)."""
    best, at = -2.0, 0
    for a in range(1, len(fine.PUSHES)):
        p = fine.PUSHES[a]
        c = (p[0] * dx + p[2] * dz) / (math.hypot(p[0], p[2]) * max(math.hypot(dx, dz), 1e-9))
        if c > best:
            best, at = c, a
    return at


def add_launches(g: Graph, world: World, ball: ph.Ball = ph.WOOD, log=lambda *a: None, ball_at=None) -> int:
    """ball_at: {place: ball} for places where a run-up starts as another ball (inside a transformer's range)."""
    ball_at = ball_at or {}
    P = g.pos
    n = g.info["rest_places"]
    par = ph.Params()
    grid: dict = {}
    for k in range(n):
        grid.setdefault((int(round(P[k, 0] / STEP)), int(round(P[k, 2] / STEP))), []).append(k)

    def floor_near(x, z, y, dy, reach=0.8):
        best, at = reach, -1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for k in grid.get((int(round(x / STEP)) + di, int(round(z / STEP)) + dj), ()):
                    d = math.hypot(P[k, 0] - x, P[k, 2] - z)
                    if d <= best and abs(P[k, 1] - y) <= dy:
                        best, at = d, k
        return at

    def node_near(p, reach=1.5):
        d = np.sqrt(((P - p) ** 2).sum(1))
        k = int(d.argmin())
        return k if d[k] <= reach else -1

    jobs = []                                              # (start place, site, push index, run-up length)
    for u in np.flatnonzero(g.info["edge_node"][:n]):
        for a in range(1, len(fine.PUSHES)):
            d = fine.PUSHES[a, [0, 2]] / np.linalg.norm(fine.PUSHES[a, [0, 2]])
            if floor_near(P[u, 0] + 2.0 * d[0], P[u, 2] + 2.0 * d[1], P[u, 1], 1.2) >= 0:
                continue                                   # floor goes on in this direction: not a ledge this way
            for L in RUNUPS:
                w, ok = int(u), True
                if L > 0:
                    w = floor_near(P[u, 0] - L * d[0], P[u, 2] - L * d[1], P[u, 1], 0.6 * L + 0.3)
                    ok = w >= 0 and all(floor_near(P[u, 0] - s * d[0], P[u, 2] - s * d[1], P[u, 1], 0.6 * s + 0.6, 1.0) >= 0
                                        for s in np.arange(2.0, L, 2.0))
                if ok:
                    jobs.append((w, int(u), a, L))
    for chain in _lane_chains(g):
        if len(chain) < 3:
            continue
        for ids in (chain, chain[::-1]):                   # towards either end
            end = int(ids[-1])
            t = P[ids[-1]] - P[ids[-3]]
            a = _push_towards(t[0], t[2])
            along = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[ids[::-1]], axis=0), axis=1))]       # arc length back from the end
            for L in RUNUPS:
                k = int(np.searchsorted(along, L))
                if k >= len(ids):
                    k = len(ids) - 1
                jobs.append((int(ids[::-1][k]), end, a, L))
                if k == len(ids) - 1:
                    break
    log("launches: %d run-ups to try from %d ledges and lane ends" % (len(jobs), len(set(j[1] for j in jobs))))
    t0 = time.time()
    best: dict = {}
    for w, u, a, L in jobs:
        bw = ball_at.get(int(w), ball)
        start = P[w] + np.array([0.0, CONTACT_MARGIN if w < n else 0.0, 0.0])
        t1 = min(420, int(66 * (1.3 * math.sqrt(2.0 * max(L, 1.0) / 8.0) + 1.0)))
        st, done, _p, _v, _o, _s, tr = world.rollout(bw, par, start, np.zeros(3), np.zeros(3), fine.PUSHES[a:a + 1], [t1], trace=True)
        air = tr[:, 3] == 0
        run = np.convolve(air.astype(np.int64), np.ones(8, np.int64), "valid") if len(air) >= 8 else np.zeros(0)
        off = np.flatnonzero(run == 8)
        if not len(off):
            continue                                       # it never left the ground (or was stopped before it did)
        n1 = int(off[0])
        if n1 < 1:
            continue
        for b in (a, 0):                                   # the key still held in the air, or released
            st, done, pos, vel, omega, s, tr2 = world.rollout(bw, par, start, np.zeros(3), np.zeros(3), np.array([fine.PUSHES[a], fine.PUSHES[b]]),
                                                             [n1, 300], trace=True)
            c = tr2[n1 + 8:, 7] > 0
            held = np.convolve(c.astype(np.int64), np.ones(4, np.int64), "valid") if len(c) >= 4 else np.zeros(0)
            down = np.flatnonzero(held == 4)
            if not len(down):
                continue                                   # fell out, or touched something placed, before it was carried again
            j = n1 + 8 + int(down[0]) + 3
            v = node_near(tr2[j, :3])
            if v < 0 or v == w:
                continue
            if math.hypot(P[v, 0] - P[w, 0], P[v, 2] - P[w, 2]) < 2.5 and abs(P[v, 1] - P[w, 1]) < 1.0:
                continue                                   # it only hopped: rolling covers that
            ticks = j + 1
            if (w, v) not in best or ticks < best[(w, v)]["ticks"]:
                way, last = [], None                       # where the ball was carried on the way: the run-up, then the landing
                for q in tr2[:j + 1][tr2[:j + 1, 7] > 0, :3]:
                    if last is None or np.linalg.norm(q - last) >= 1.0:
                        way.append(q)
                        last = q
                best[(w, v)] = dict(tape=[(a, n1), (b, ticks - n1)], ticks=ticks, site=u, runup=L, launch_speed=float(tr2[n1, 6]),
                                    way=np.array(way, np.float32),
                                    land_speed=float(tr2[j, 6]), across=float(math.hypot(tr2[j, 0] - tr2[n1, 0], tr2[j, 2] - tr2[n1, 2])),
                                    down=float(tr2[n1, 1] - tr2[j, 1]), keys_in_air=bool(b), ball=bw.name)
    g.info["launches"] = best
    for (w, v), e in best.items():
        g.adj[w].append((v, e["ticks"] / 66.0 * FOLLOW_SPEED + STOP_COST, "launch"))
    log("launches: %d edges from kernel run-ups, %.0f s" % (len(best), time.time() - t0))
    return len(best)
