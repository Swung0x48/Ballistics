"""Worlds and modes for the fine level, and the whole-sector search (the first version: one search from a
checkpoint to the next, kept for comparison; `staged.py` is the coarse-first flow).

    python -m ballistics solve --mesh data/Level_01-mesh.npz --map data/Level_01-map.json --sector 2 --whole strict

Modes:
  strict   a rollout may not touch anything placed on the map (mechanisms, boxes, loose balls)
  free     the loose balls are left out (ASSUMED to be knocked aside); only a diagnosis of what blocks

Everything found is a trajectory of the hand-written kernel: a candidate until the engine has run it.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from . import coarse, fine
from . import physics as ph
from .geometry import BALL_R, Mesh
from .kernel import OK, World

MODES = {"strict": True, "free": False}
AIR, RAIL, FLOOR = 0, 1, 4
KIND_NAMES = {AIR: "air", RAIL: "on a rail", FLOOR: "on the floor"}


def world_for(mesh: Mesh, info: dict, mode: str, sector: int | None = None) -> World:
    """sector: the sector the ball is solved in. Its fans push (`ballistics.fans`); without it no fan does, which is
    how the coarse level's launch edges are made (they are proposals)."""
    from . import fans, props, trafo
    solid_balls = MODES[mode]
    mesh = coarse.in_sector(mesh, sector)             # the sector's domes are fixed geometry; other sectors' are not there
    shapes = mesh                                     # with the placed shapes still in it: where the mechanisms' parts are
    mesh = coarse.without_loose(mesh)                 # the loose bodies are bodies (strict) or not there (free): never fixed shapes
    # letting them settle takes a moment: kept per sector, by what the loose bodies are and what they stand on
    key = (sector, len(mesh.A), float(mesh.A.sum()) if len(mesh.A) else 0.0,
           tuple((q.get("name"), q["type"], tuple(q["position"])) for q in info.get("placements", ())
                 if (q["type"] in props.KINDS or q["type"] in props.PARTS) and q.get("sector") == sector))
    if solid_balls and key not in _PROPS:
        _PROPS[key] = props.of(info, sector, mesh.subset(~mesh.placed) if mesh.placed.any() else mesh, shapes)
    # the transformers in place: the sector's own (`ballistics.trafo`); without a sector, every one of the map
    return World.of(mesh, info["death_boxes"], fans=fans.of(info, sector), trafos=trafo.of(info, sector),
                    props=_PROPS[key] if solid_balls else None)


_PROPS: dict = {}


def props_parts(info: dict, shapes) -> list:
    from . import props
    return props.part_boxes(info, shapes)


GATE_OPEN = 4.95          # how far a gate (P_Modul_01) pushed open along its groove has slid: MEASURED 4.97 to 4.98 at its stop


def _hits(pts, boxes, clear: float, pushed=None) -> bool:
    """Whether a ball centred at any of pts overlaps any of the pieces (centre, half, axes, hull triangles or None)
    by more than `clear`: by its box, and where the piece is a hull that is not its box, by the hull.
    pushed: (the index of a piece, a point of the face it is pushed on, the way it gives): a ball that touches
    that piece from in front of that face pushes it, it is not stopped by it."""
    from .kernel import PROP_WIDTH, _hull_point
    for i, (c, half, axes, tris) in enumerate(boxes):
        if pushed is not None and i == pushed[0]:
            ahead = (pts - pushed[1]) @ pushed[2] > -0.5     # past the face (or at it): the piece is in the way
            if not ahead.any():
                continue
            L = (pts[ahead] - c) @ axes.T
            q = np.clip(L, -half, half)
            if (np.linalg.norm(L - q, axis=1) < BALL_R - clear).any():
                return True
            continue
        L = (pts - c) @ axes.T
        q = np.clip(L, -half, half)
        near = np.linalg.norm(L - q, axis=1) < BALL_R - clear
        if not near.any():
            continue
        if not tris:
            return True
        row, prow = np.zeros(29), np.zeros(PROP_WIDTH)
        row[21:24] = c
        T = np.asarray(tris, float).ravel()
        prow[23], prow[24:24 + len(T)] = len(tris), T
        for p in pts[near]:
            if _hull_point(float(p[0]), float(p[1]), float(p[2]), row, prow, BALL_R)[0] < -clear:
                return True
    return False


def gate_edges(g, info: dict, parts: list) -> dict:
    """The sliding gates (P_Modul_01) on the coarse level. A closed gate stands in the upper road: the edges a ball
    on them would run into it with are taken out of the graph. Those of them that are clear once the gate is pushed
    open (slid GATE_OPEN along its own x) are kept per gate, to be put back for a plan that opens it first.
    Per gate: dict(sector, edges [(u, v, cost, kind)], axis (its x in world), face (the middle of the target
    plate's face that is pushed), operate (the place the ball pushes from, on the lower road, or -1))."""
    out = {}
    bodies: dict = {}
    for b in parts:
        if b["kind"] == "P_Modul_01_Pusher":
            bodies.setdefault(b["body"], []).append(b)
    launches = g.info.get("launches", {})
    for body, pieces in bodies.items():
        place = body.split("/")[0]
        p = next(q for q in info["placements"] if q["name"] == place)
        axis = np.array(p["x_axis"], float) / np.linalg.norm(p["x_axis"])
        closed = [(np.array(b["placed"], float), np.array(b["half"], float), np.array(b["axes"], float), b.get("tris")) for b in pieces]
        opened = [(c + GATE_OPEN * axis, h, a, t) for c, h, a, t in closed]
        centre = np.mean([c for c, _h, _a, _t in closed], axis=0)
        reach = max(float(np.linalg.norm(c - centre) + np.linalg.norm(h)) for c, h, _a, _t in closed) + 30.0
        # the target plate: the piece that sticks out furthest against the gate's x; it is pushed on its -x face.
        # A ball rolling into that face pushes the gate open (that is how it is opened), it is not stopped by it
        ti = min(range(len(closed)), key=lambda i: float((closed[i][0] - centre) @ axis))
        tc, th = closed[ti][0], closed[ti][1]
        face = tc - th[0] * axis
        edges = []
        for u in np.flatnonzero(np.linalg.norm(g.pos - centre, axis=1) < reach):
            keep = []
            for v, c, k in g.adj[u]:
                way = np.asarray(launches[(u, v)]["way"], float) if k == "launch" and (u, v) in launches else np.array([g.pos[u], g.pos[v]])
                seg = [way[0]]
                for a, b2 in zip(way[:-1], way[1:]):
                    n = int(np.linalg.norm(b2 - a) / 0.5) + 1
                    seg += [a + (b2 - a) * (i / n) for i in range(1, n + 1)]
                pts = np.array(seg)
                if _hits(pts, closed, 0.1, pushed=(ti, face, axis)):
                    if not _hits(pts, opened, 0.1):
                        edges.append((int(u), int(v), float(c), k))
                    continue
                keep.append((v, c, k))
            g.adj[u] = keep
        spot = face - (BALL_R + 0.3) * axis
        op = g.nearest(spot, 3.0)
        out[place] = dict(sector=p.get("sector"), edges=edges, axis=axis, face=face, operate=int(op))
    return out


def graph_for(mesh: Mesh, info: dict, cache: Path | None = None, log=print):
    """The coarse graph of a map: the domes are geometry, the loose balls are not obstacles but cost to go through,
    a stone transformer costs its transformation, a paper transformer is kept out of."""
    mesh = coarse.in_sector(mesh)                         # the domes are geometry for the coarse level too (in every sector: it has one graph)
    parts = props_parts(info, mesh)                       # the mechanisms' box-shaped parts, where their shapes are
    mesh = coarse.without_loose(mesh)                     # the loose bodies are bodies: no obstacles, but going through one costs (below)
    fixed = mesh.meta.get("fixed_bodies")
    from . import props, trafo
    from .launch import FOLLOW_SPEED
    marks = dict(fixed_bodies=fixed, transformers="wood and stone ones can be gone through, at the cost of the transformation",
                 loose_balls="bodies: gone through at a cost %s; parts of %s" % (sorted(props.COARSE_COST.items()), sorted(props.PARTS)),
                 gate_rule="a closed gate blocks the edges into it; the edges it clears when pushed open %.2f are kept per gate" % GATE_OPEN)
    if cache is not None and cache.exists():
        g = pickle.load(open(cache, "rb"))
        n = g.info.get("mesh_triangles")                  # graphs cached before this was recorded have none
        if (n == len(mesh.A) or (n is None and not mesh.meta.get("bodiless_dropped"))) and all(g.info.get(k) == v for k, v in marks.items()):
            return g
        log("the cached coarse graph %s was made by other rules (fans, markers, domes or transformers were obstacles in it): building it again" % cache.name)
    from .launch import add_launches
    g = coarse.build(mesh, avoid=coarse.keep_out(info["placements"]), log=log)
    coarse.add_steps(g, mesh, log=log)
    add_launches(g, world_for(mesh, info, "strict"), log=log)
    # a stone transformer can be gone through: the edges that lead into its range cost the transformation's time
    # more, as way at the follower's pace (the coarse level does not know the ball's kind: it starts as wood, so a
    # wood transformer costs nothing)
    R = trafo.rules()
    extra = (R.body_after + 1) / 66.0 * FOLLOW_SPEED
    for t in trafo.of(info):
        if t["kind"] != "stone":
            continue
        inside = np.linalg.norm(g.pos - t["pos"], axis=1) < R.radius
        for u in np.flatnonzero(~inside):
            g.adj[u] = [(v, c + extra if inside[v] else c, k) for v, c, k in g.adj[u]]
    # a loose ball can be knocked aside: the edges that lead into where it lies cost more, by its kind
    for p in info.get("placements", ()):
        kind = props.KINDS.get(p["type"])
        if kind is None:
            continue
        inside = np.linalg.norm(g.pos - np.array(p["position"], float), axis=1) < (4.5 if kind == "box" else 2.0 * props.RADIUS)
        for u in np.flatnonzero(~inside):
            g.adj[u] = [(v, c + props.COARSE_COST[kind] if inside[v] else c, k) for v, c, k in g.adj[u]]
    for b in parts:                                       # the same for a mechanism's parts (the block of a falling rock stands in the upper road)
        if b["kind"] not in props.COARSE_COST:
            continue
        inside = np.linalg.norm(g.pos - np.array(b["placed"], float), axis=1) < float(np.linalg.norm(b["half"])) + 1.0
        for u in np.flatnonzero(~inside):
            g.adj[u] = [(v, c + props.COARSE_COST[b["kind"]] if inside[v] else c, k) for v, c, k in g.adj[u]]
    g.info["gates"] = gate_edges(g, info, parts)
    for name, d in g.info["gates"].items():
        log("gate %s (sector %s): %d edges wait for it to be opened; pushed from place %d" % (name, d["sector"], len(d["edges"]), d["operate"]))
    g.info["mesh_triangles"] = len(mesh.A)
    g.info.update(marks)
    if cache is not None:
        pickle.dump(g, open(cache, "wb"))
    return g


def coarse_plans(mesh: Mesh, info: dict, g) -> list:
    """The coarse level's plan alone, per sector: the cheapest way from the sector's reset point to its goal over
    the coarse graph plus the links it needs, before anything is rolled or searched. Every sector starts at rest
    under its reset point. Each: dict(sector, ok, reason (when not ok), nodes, kinds (of the edge leaving each
    node but the last), length, counts (edges by kind), launches, steps, links); a link is a place the coarse
    level does not connect, with what is placed within 9 of it. `starts_off` (with `reset`): the plan does not
    begin at the reset point but that far from it, because nothing nearer reaches the goal even with links.
    `short_of_goal`: the plan ends that far from the goal's marker.
    This is only what the coarse level proposes: positions, no speeds, and no statement that a ball can drive it."""
    from collections import Counter

    from .staged import nearest_node

    rel = coarse.Relaxed(g)
    strict = world_for(mesh, info, "strict")
    P = g.pos
    anchors = [info["start"]] + list(info["checkpoints"]) + [info.get("airship")]
    out = []
    for sector in range(1, len(info["reset_points"]) + 1):
        plan = dict(sector=sector, ok=False, nodes=[], kinds=[], length=0.0, counts={}, launches=[], steps=[], links=[])
        out.append(plan)
        if sector >= len(anchors) or anchors[sector] is None:
            plan["reason"] = "the map gives no goal for this sector"
            continue
        goal = g.nearest(anchors[sector], 25.0)
        if goal < 0:
            # the road ends further from the goal's marker than the solver accepts (the airship's own shape is not in
            # the mesh): plan to the nearest place of any kind, a rail included, and say how far short of the marker it is
            goal = g.nearest(anchors[sector], 80.0, floor_only=False)
            if goal < 0:
                plan["reason"] = "no place a ball can be on within 80 of the sector's goal"
                continue
            plan["short_of_goal"] = float(np.linalg.norm(P[goal] - np.asarray(anchors[sector], float)))
        reset = info["reset_points"][sector - 1]
        try:
            s0 = fine.rest_state(strict, reset[0], reset[2], reset[1] + 0.5)
        except ValueError:
            plan["reason"] = "nothing under the sector's reset point"
            continue
        togo = rel.togo(goal)
        if not np.isfinite(togo).any():
            plan["reason"] = "the goal is not reached from anywhere"
            continue
        nodes, kinds = rel.plan(nearest_node(g, togo, s0.pos), togo)
        if not np.isfinite(togo[nodes[0]]) or nodes[-1] != goal:
            plan["reason"] = "no way from the reset point to the goal, not even with links"
            continue
        plan.update(ok=True, nodes=nodes, kinds=kinds, counts=dict(Counter(kinds)),
                    length=float(sum(np.linalg.norm(P[b] - P[a]) for a, b in zip(nodes, nodes[1:]))))
        off = float(np.linalg.norm(P[nodes[0]] - s0.pos))
        if off > 3.0:
            # nothing near the reset point reaches the goal, not even with links: the plan begins at the nearest place
            # that does, and the stretch from the reset point to there has no plan at all
            plan.update(starts_off=off, reset=np.array(s0.pos))
        for e, k in enumerate(kinds):
            a, b = P[nodes[e]], P[nodes[e + 1]]
            edge = dict(edge=e, kind=k, near=a.copy(), far=b.copy(), across=float(math.hypot(b[0] - a[0], b[2] - a[2])), rise=float(b[1] - a[1]))
            if k == "launch":
                L = g.info["launches"][(nodes[e], nodes[e + 1])]
                plan["launches"].append(dict(edge, runup=float(L["runup"]), speed=float(L["launch_speed"]), across=float(L["across"]), down=float(L["down"])))
            elif k == "step":
                plan["steps"].append(edge)
            elif k in ("gap", "bridge"):
                ab = b - a
                around = set()
                for q in info["placements"]:
                    c = np.array(q["position"], float)
                    f = float(np.clip(((c - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0))
                    if np.linalg.norm(c - (a + f * ab)) <= 9.0:
                        around.add(q["type"])
                plan["links"].append(dict(edge, placed_near=sorted(around)))
    return out


def whole_sector(mesh: Mesh, info: dict, g, sector: int, mode: str = "strict", expansions: int = 3_000_000, seconds: float = 600.0,
                 seed: int = 0, log=print):
    """sector: 1-based. From rest under the sector's reset point to the places around the next checkpoint (for the
    last sector: around the last coarse place before the airship, whose own shape is not in the mesh)."""
    world = world_for(mesh, info, mode, sector)
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
