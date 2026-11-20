"""Worlds and modes for the fine level, and the whole-sector search (the first version: one search from a
checkpoint to the next, kept for comparison; `staged.py` is the coarse-first flow).

    python -m ballistics solve --mesh data/Level_01-mesh.npz --map data/Level_01-map.json --sector 2 --whole strict

Modes:
  strict   a rollout may not touch anything placed on the map (mechanisms, boxes, loose balls)
  free     the loose balls are left out (ASSUMED to be knocked aside); only a diagnosis of what blocks

Everything found is a trajectory of the hand-written kernel: a candidate until the engine has run it.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import coarse, fine
from . import physics as ph
from .geometry import Mesh
from .kernel import OK, World

MODES = {"strict": True, "free": False}
AIR, RAIL, FLOOR = 0, 1, 4
KIND_NAMES = {AIR: "air", RAIL: "on a rail", FLOOR: "on the floor"}


def world_for(mesh: Mesh, info: dict, mode: str) -> World:
    solid_balls = MODES[mode]
    # ASSUMED: a transformer acts when the ball's centre is within its trigger distance of a point one ball radius
    # above the transformer's origin (the distance is READ, the centre's height is not)
    keep = [(x, y + 2.0, z, d) for x, y, z, d in coarse.keep_out(info["placements"])]
    if not solid_balls:
        mesh = coarse.without_loose(mesh)
    return World.of(mesh, info["death_boxes"], keep)


def graph_for(mesh: Mesh, info: dict, cache: Path | None = None, log=print):
    """The coarse graph of the mesh as given: pass the full mesh to have the loose balls in it as obstacles, or
    `coarse.without_loose(mesh)` to have them out."""
    if cache is not None and cache.exists():
        return pickle.load(open(cache, "rb"))
    from .launch import add_launches
    g = coarse.build(mesh, avoid=coarse.keep_out(info["placements"]), log=log)
    coarse.add_steps(g, mesh, log=log)
    add_launches(g, world_for(mesh, info, "strict"), log=log)
    if cache is not None:
        pickle.dump(g, open(cache, "wb"))
    return g


def whole_sector(mesh: Mesh, info: dict, g, sector: int, mode: str = "strict", expansions: int = 3_000_000, seconds: float = 600.0,
                 seed: int = 0, log=print):
    """sector: 1-based. From rest under the sector's reset point to the places around the next checkpoint (for the
    last sector: around the last coarse place before the airship, whose own shape is not in the mesh)."""
    world = world_for(mesh, info, mode)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    n = g.info["rest_places"]
    goal = g.nearest(anchors[sector], 25.0)
    targets = g.pos[:n][np.linalg.norm(g.pos[:n] - g.pos[goal], axis=1) <= 2.5]
    reset = info["reset_points"][sector - 1]
    start = fine.rest_state(world, reset[0], reset[2], reset[1] + 0.5)
    s = fine.Search(world, targets, seed=seed, places=g.pos, togo=fine.guide(g, goal))
    s.add_start(start)
    c = s.run(max_expansions=expansions, max_seconds=seconds)
    log("sector %d, %s: %s after %d expansions, %.0f s" % (sector, mode, "found" if c.found else "NOT found", c.expansions, c.seconds))
    if not c.found:
        log("   nearest it got on the ground: %s" % np.round(c.nearest, 1))
    return world, c


def analyse(world: World, info: dict, c: fine.Crossing, trials: int = 12, log=print) -> dict:
    """Flights, loose balls passed through, and how sensitive each game second is."""
    P = ph.Params()
    start = ph.State(c.start, np.zeros(3), np.zeros(3))
    out = fine.replay(world, ph.WOOD, P, start, c.tape)
    tr = out["trace"]
    kind = np.full(len(tr), FLOOR)
    kind[tr[:, 8] > 0] = RAIL
    kind[tr[:, 3] == 0] = AIR

    def stretches(v, least):
        i, res = 0, []
        while i < len(tr):
            if kind[i] != v:
                i += 1
                continue
            j = i
            while j + 1 < len(tr) and kind[j + 1] == v:
                j += 1
            if j - i + 1 >= least:
                res.append((i, j))
            i = j + 1
        return res

    flights = stretches(AIR, 30)
    log("%d ticks = %.1f game s, %d key changes; ticks %s" % (len(tr), len(tr) / 132, len(c.tape), ", ".join("%s %d" % (KIND_NAMES[v], int((kind == v).sum())) for v in KIND_NAMES)))
    for i, j in flights:
        a, b = tr[max(i - 1, 0)], tr[min(j + 1, len(tr) - 1)]
        log("   flight, ticks %d-%d (%.2f game s): %s -> %s, across %.1f, down %.1f, speed at take-off %.1f"
            % (i, j, (j - i) / 132, np.round(a[:3], 1), np.round(b[:3], 1), np.hypot(b[0] - a[0], b[2] - a[2]), a[1] - b[1], a[6]))
    passed = []
    for p in info["placements"]:
        if p["type"].startswith(coarse.LOOSE):
            q = np.array(p["position"], float)
            d = np.linalg.norm(tr[:, :3] - q, axis=1)
            if d.min() < 4.0:
                passed.append((p["type"], q, float(d.min()), int(d.argmin())))
                log("   goes through a loose ball (%s at %s): centres %.1f apart" % (p["type"], np.round(q, 1), d.min()))
    flat = np.array([fine.PUSHES[a] for a, n in c.tape for _ in range(n)])
    rng = np.random.default_rng(0)
    state = [start.pos.copy(), start.vel.copy(), start.omega.copy()]
    within, at = [], []
    k, step, ahead = 0, 66, 132
    while k + ahead <= len(flat):
        ok = 0
        for _ in range(trials):
            pos = state[0] + rng.normal(0, 0.03, 3) * np.array([1, 0, 1])
            vel = state[1] + rng.normal(0, 0.3, 3)
            st, done, p2, _v, _o, _s, _ = world.rollout(ph.WOOD, P, pos, vel, state[2], flat[k:k + ahead], np.ones(ahead, np.int64))
            ok += st == OK and done == ahead and np.linalg.norm(p2 - tr[k + ahead - 1, :3]) <= 1.5
        within.append(ok / trials)
        at.append(k)
        st, done, state[0], state[1], state[2], s2, _ = world.rollout(ph.WOOD, P, state[0], state[1], state[2], flat[k:k + step], np.ones(step, np.int64))
        k += step
    within = np.array(within)
    sensitive = [at[i] for i in range(len(at)) if within[i] < 0.5]
    log("   sensitivity (state nudged 0.03 in place and 0.3 in speed, same keys, one game second on): %d windows, all stay within 1.5 in %d, fewer than half in %d"
        % (len(within), int((within == 1.0).sum()), len(sensitive)))
    return dict(trace=tr, kind=kind, flights=flights, passed=passed, sensitive=sensitive, within=within, ticks=len(tr),
                key_changes=len(c.tape), stats=out["stats"], expansions=c.expansions, seconds=c.seconds)
