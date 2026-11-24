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
    # Strict sectors replace the T-board's placed obstacle by its catalogue opening pose. Keep the coarse mesh
    # and its cache unchanged. Do not discard a placed board that has no exported hull.
    if solid_balls and sector is not None:
        t_boards = {h["placement"] for h in info.get("hulls", ()) if h.get("body") == "P_Modul_37_Bridge"
                    and h.get("frame") is not None and h.get("sector") == sector}
        if t_boards:
            posed = np.array([e.get("placement") in t_boards and e.get("name") == "P_Modul_37_Bridge"
                              and bool(e.get("placed")) for e in mesh.entities])
            mesh = mesh.subset(~posed[mesh.owner])
    bodies = props.of(info, sector, mesh.subset(~mesh.placed) if mesh.placed.any() else mesh, shapes) if solid_balls else None
    # The transformers in place: the sector's own (`ballistics.trafo`); without a sector, every one of the map.
    return World.of(mesh, info["death_boxes"], fans=fans.of(info, sector), trafos=trafo.of(info, sector), props=bodies)


def props_parts(info: dict, shapes) -> list:
    from . import props
    return props.part_boxes(info, shapes)


GATE_OPEN = 4.95          # how far a gate (P_Modul_01) pushed open along its groove has slid: MEASURED 4.97 to 4.98 at its stop


def _hits(pts, boxes, clear: float, pushed=None) -> bool:
    """Whether a ball centred at any of pts overlaps any of the pieces (centre, half, axes, hull triangles or None)
    by more than `clear`: by its box, and where the piece is a hull that is not its box, by the hull.
    pushed: (the index of a piece, a point of the face it is pushed on, the way it gives): a ball that touches
    that piece from in front of that face pushes it, it is not stopped by it."""
    from .kernel import BC_WIDTH, PROP_WIDTH, TRI0, _hull_point
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
        row, prow = np.zeros(BC_WIDTH), np.zeros(PROP_WIDTH)
        row[21:24] = c
        row[29:38] = np.eye(3).ravel()                    # as it was put
        T = np.asarray(tris, float).ravel()
        prow[23], prow[TRI0:TRI0 + len(T)] = len(tris), T
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
        out[place] = dict(sector=p.get("sector"), edges=edges, axis=axis, face=face, operate=int(op), pieces=closed)
    return out


BOARD_COST = 5.0          # going over a push board, on top of the way: ASSUMED, only to rank ways (pushing it over slows the ball)


def board_edges(g, info: dict, parts: list) -> dict:
    """The push boards (P_Modul_25) on the coarse level. Where a board is pushed is where it is crossed: a ball rolling
    from the road into the leaning board pushes it over and goes on across it as it lies on its stopper. So the
    places on the road in front of it (1.5 to 4 from the hinge, within 2 of its middle) get an edge across to the
    places on the far side (10 to 14 from the hinge), one way only: from behind it cannot be pushed over (READ: no
    road there). Per board: dict(sector, edges [(u, v, cost, kind)], fall (the way it falls, level))."""
    out = {}
    for b in parts:
        if not b.get("hinge") or not b["kind"].startswith("P_Modul_25") or b["body"] in out:
            continue
        hg = b["hinge"]
        a, o = np.array(hg["axis"], float), np.array(hg["anchor"], float)
        fall = np.cross(a, [0.0, 1.0, 0.0])                # where its top goes as it falls over (positive turn)
        fall /= np.linalg.norm(fall)
        side = np.cross([0.0, 1.0, 0.0], fall)
        rel = g.pos - o
        along, across, up = rel @ fall, rel @ side, rel[:, 1]
        floor = g.kind == 0
        near = np.flatnonzero(floor & (along > -4.0) & (along < -1.5) & (np.abs(across) <= 2.0) & (up > 0.5) & (up < 4.0))
        far = np.flatnonzero(floor & (along > 10.0) & (along < 14.0) & (np.abs(across) <= 2.0) & (np.abs(up - 2.2) < 2.5))
        edges, back = [], []
        for u in near:
            if not len(far):
                break
            v = int(far[np.argmin(np.abs(across[far] - across[u]) + 0.1 * np.abs(along[far] - 11.0))])
            c = float(np.linalg.norm(g.pos[v] - g.pos[u])) + BOARD_COST
            edges.append((int(u), v, c, "roll"))
            back.append((v, int(u), c, "roll"))
        # kept apart from the graph's own edges: whether they are there depends on the board's state (`coarse.StateGraph`)
        pieces = [(np.array(q["placed"], float), np.array(q["half"], float), np.array(q["axes"], float), q.get("tris"))
                  for q in parts if q["body"] == b["body"]]
        out[b["body"]] = dict(sector=b.get("sector"), edges=edges, back=back, fall=fall, pieces=pieces)
    return out


LIFT_WAITS = (0, 40, 80, 130)     # ticks a paper ball rises in the fan's column before it is pushed towards the way out
LIFT_ENTRIES, LIFT_EXITS = 3, 40  # places tried per fan: where it is entered from, where it is left to
LIFT_UNSURE = 1.5                 # the cost of a lift that only works with one of the two fan bounds is this many times its time
LIFT_CHAIN = 25.0                 # a lift may go on into another fan's column this near (two fans in a row: level 2 sector 3's way up to the stone transformer)
LIFT_CHAIN_WAITS = (0, 60, 120)   # ticks it rises in the second column before it is pushed on
LIFT_CHAIN_BRAKES = (0, 16, 32)   # ticks it is pushed back once near the second column (or it flies past it)
LIFT_CHAIN_EXITS = 20


def push_towards(d) -> int:
    """The push (`fine.PUSHES`) whose direction is nearest d's horizontal direction: by direction only, a diagonal
    push is longer than a straight one but not nearer."""
    d = np.asarray(d, float)
    d = d[[0, 2]] if len(d) == 3 else d
    d = d / max(np.linalg.norm(d), 1e-9)
    h = fine.PUSHES[:, [0, 2]]
    return int(np.argmax(h @ d / np.maximum(np.linalg.norm(h, axis=1), 1e-9)))


def _lift_world(world, edge: str):
    """The same world with the fans' other bound for a ball at a box's edge (`fans.Rules.params`)."""
    import copy
    w2 = copy.copy(world)
    w2.fan_edge, w2._fanpar, w2._ballc = edge, {}, {}
    return w2


def ride(world, ball, state, legs, reach: float = 1.0, scene=None, arrive_at=None):
    """From state (pos, vel, omega): each leg (push, ticks, until[, reach]) is held for up to `ticks` ticks, in chunks
    of 8, until the ball's centre is horizontally within reach of its `until` point (None: the whole time). With
    arrive_at it stops after the chunk in which the ball is carried within 2.5 of that point. scene: the loose
    bodies' state, changed in place. Returns (tape [(push, ticks)], trace, status, (pos, vel, omega), stats)."""
    from .kernel import OK
    pos, vel, om = (np.array(x, float) for x in state[:3])
    tape, traces, stats = [], [], np.zeros(10)

    def out(st):
        return tape, (np.concatenate(traces) if traces else np.zeros((0, 9))), st, (pos, vel, om), stats

    for leg in legs:
        a, ticks, until = leg[:3]
        near = leg[3] if len(leg) > 3 else reach
        done = 0
        while done < ticks:
            if until is not None and math.hypot(pos[0] - until[0], pos[2] - until[2]) < near:
                break
            st, d, pos, vel, om, stats, tr = world.rollout(ball, ph.Params(), pos, vel, om, fine.PUSHES[a:a + 1], [8], trace=True, scene=scene)
            traces.append(tr)
            if tape and tape[-1][0] == a:
                tape[-1] = (a, tape[-1][1] + d)
            else:
                tape.append((a, d))
            done += d
            if st != OK:
                return out(st)
            if arrive_at is not None and len(tr) and ((np.linalg.norm(tr[:, :3] - arrive_at, axis=1) <= 2.5) & (tr[:, 7] > 0)).any():
                return out(OK)
    return out(OK)


def _lift_try(world, start, legs, reach: float = 1.0):
    """A paper ball from rest at `start` along the legs (`ride`). Returns (tape, trace, status)."""
    tape, tr, st, _end, _stats = ride(world, ph.PAPER, (start, np.zeros(3), np.zeros(3)), legs, reach)
    return tape, tr, st


def fan_lifts(g, mesh: Mesh, info: dict, log=print) -> dict:
    """The fans' lift edges, for a paper ball only (`coarse.StateGraph`): from a place near a fan's axis up to a place
    beside or above it, found by rolling the paper ball in the kernel with that sector's fans: it rolls to the axis,
    rises in the column for a while (LIFT_WAITS), then is pushed towards the way out and let go there. An edge is
    kept where that ends on the place, carried. It is tried again with the fans' two bounds for a ball at a box's
    edge (`fans.Rules.params`): one that arrives with only one of them is unsure (it depends on the ball's
    orientation) and costs more. {(u, v): dict(sector, tape, ticks, way, both, bounds, cost)}."""
    from . import fans
    from .launch import FOLLOW_SPEED
    out = {}
    P = g.pos
    floor = g.kind == 0
    sectors = sorted({p["sector"] for p in info["placements"] if p["type"] == fans.TYPE and p.get("sector")})
    pushes = fine.PUSHES

    towards = push_towards

    def keep(u, v, start, best, bounds, s):
        ok = {}
        for e, wb in bounds.items():
            st, d, p2, _v, _o, _s2, tr2 = wb.rollout(ph.PAPER, ph.Params(), start, np.zeros(3), np.zeros(3),
                                                    np.array([pushes[a] for a, _k in best["tape"]]), [k for _a, k in best["tape"]], trace=True)
            ok[e] = bool(len(tr2) and np.linalg.norm(tr2[-1, :3] - P[v]) <= (5.0 if st == TRIGGER else 2.5))
        both = ok["least"] and ok["most"]
        way, last = [], None
        for q in best["trace"][:, :3]:
            if last is None or np.linalg.norm(q - last) >= 1.0:
                way.append(q)
                last = q
        way.append(P[v])
        cost = best["ticks"] / 66.0 * FOLLOW_SPEED * (1.0 if both else LIFT_UNSURE)
        key = (int(u), int(v))
        if key not in out or cost < out[key]["cost"]:
            out[key] = dict(sector=s, tape=best["tape"], ticks=best["ticks"], way=np.array(way, np.float32), both=both, bounds=ok, cost=cost,
                            chain=best.get("chain", False), fan=np.array(best["fan"], float), legs=best["legs"], ends_in=best.get("ends_in"))

    from .kernel import TRIGGER

    def arrive(tape, tr, v, status=None):
        near = np.flatnonzero((np.linalg.norm(tr[:, :3] - P[v], axis=1) <= 2.5) & (tr[:, 7] > 0))
        ends_in = None
        if not len(near) and status == TRIGGER and np.linalg.norm(tr[-1, :3] - P[v]) <= 5.0:
            near = np.array([len(tr) - 1])                 # it went into a transformer's range there (in the air, maybe): that is arriving
            ends_in = min(w.trafos, key=lambda t: float(np.linalg.norm(np.asarray(t["pos"], float) - tr[-1, :3])))["kind"]
        if not len(near):
            return None
        ticks = int(near[0]) + 1
        cut, n = [], 0
        for a, k in tape:                                  # the tape up to the arrival
            if n >= ticks:
                break
            cut.append((a, min(k, ticks - n)))
            n += k
        return dict(tape=cut, ticks=ticks, trace=tr[:ticks], ends_in=ends_in)

    from . import trafo as _trafo

    def with_trafos(pick, cand, s):
        """The exits picked, and always the places beside a transformer of the sector among the candidates (a lift is
        often the way to one)."""
        T = [t["pos"] for t in _trafo.of(info, s)]
        if not T or not len(cand):
            return pick
        d = np.min([np.linalg.norm(P[cand] - np.asarray(t, float), axis=1) for t in T], axis=0)
        near = cand[np.argsort(d)][:6]
        return list(pick) + [int(v) for v in near if d[list(cand).index(v)] < 6.0 and int(v) not in pick]

    for s in sectors:
        w = world_for(mesh, info, "free", s)
        bounds = {e: _lift_world(w, e) for e in ("least", "most")}
        for f in w.fans:
            o = f[:3]
            hd = np.hypot(P[:, 0] - o[0], P[:, 2] - o[2])
            ent = np.flatnonzero(floor & (hd <= 5.0) & (P[:, 1] >= o[1] - 2.0) & (P[:, 1] <= o[1] + 17.0))
            ent = ent[np.argsort(hd[ent])][:LIFT_ENTRIES]
            # where it may be left to: MEASURED, the paper ball's rides in level 2's original records land 7 to 29 from
            # the fan's axis (p10 to p90, at most 61) and -1 to 20 above its origin
            ex = np.flatnonzero(floor & (hd >= 4.0) & (hd <= 32.0) & (P[:, 1] >= o[1] - 1.5) & (P[:, 1] <= o[1] + 25.0))
            if not len(ent) or not len(ex):
                continue
            pick = [int(ex[0])]                            # exits spread over what is there (furthest point first)
            dmin = np.linalg.norm(P[ex] - P[pick[0]], axis=1)
            while len(pick) < min(LIFT_EXITS, len(ex)):
                k = int(dmin.argmax())
                if dmin[k] < 1.5:
                    break
                pick.append(int(ex[k]))
                dmin = np.minimum(dmin, np.linalg.norm(P[ex] - P[ex[k]], axis=1))
            pick = with_trafos(pick, ex, s)
            for u in ent:
                start = P[u] + np.array([0.0, 0.014, 0.0])
                for v in pick:
                    if P[v, 1] - P[u, 1] < 1.0 and math.hypot(P[v, 0] - P[u, 0], P[v, 2] - P[u, 2]) < 8.0:
                        continue                               # (rolling covers that)
                    best = None
                    for wait in LIFT_WAITS:
                        legs = []
                        if hd[u] > 1.0:
                            legs.append((towards(o[[0, 2]] - P[u, [0, 2]]), 90, o))
                        climb = [(0, wait, None), (towards(P[v, [0, 2]] - o[[0, 2]]), 300, P[v]), (0, 120, None)]
                        tape, tr, alive = _lift_try(w, start, legs + climb)
                        got = arrive(tape, tr, v, alive) if len(tr) else None
                        if got is not None and (best is None or got["ticks"] < best["ticks"]):
                            best = dict(got, legs=climb, fan=o)
                    if best is not None:
                        keep(u, v, start, best, bounds, s)
                # two fans in a row: up this one, across into the next one's column, up that, then on to a place
                for f2 in w.fans:
                    o2 = f2[:3]
                    if np.allclose(o2, o) or math.hypot(o2[0] - o[0], o2[2] - o[2]) > LIFT_CHAIN:
                        continue
                    hd2 = np.hypot(P[:, 0] - o2[0], P[:, 2] - o2[2])
                    ex2 = np.flatnonzero(floor & (hd2 >= 4.0) & (hd2 <= 32.0) & (P[:, 1] >= o2[1] - 1.5) & (P[:, 1] <= o2[1] + 25.0))
                    if not len(ex2):
                        continue
                    pick2 = [int(ex2[0])]
                    dmin2 = np.linalg.norm(P[ex2] - P[pick2[0]], axis=1)
                    while len(pick2) < min(LIFT_CHAIN_EXITS, len(ex2)):
                        k = int(dmin2.argmax())
                        if dmin2[k] < 1.5:
                            break
                        pick2.append(int(ex2[k]))
                        dmin2 = np.minimum(dmin2, np.linalg.norm(P[ex2] - P[ex2[k]], axis=1))
                    pick2 = with_trafos(pick2, ex2, s)
                    for v in pick2:
                        if P[v, 1] - P[u, 1] < 1.0 and math.hypot(P[v, 0] - P[u, 0], P[v, 2] - P[u, 2]) < 8.0:
                            continue
                        best = None
                        for wait in LIFT_WAITS:
                            for wait2, brake in [(a_, b_) for a_ in LIFT_CHAIN_WAITS for b_ in LIFT_CHAIN_BRAKES]:
                                legs = []
                                if hd[u] > 1.0:
                                    legs.append((towards(o[[0, 2]] - P[u, [0, 2]]), 90, o))
                                climb = [(0, wait, None), (towards(o2[[0, 2]] - o[[0, 2]]), 300, o2, 6.0), (towards(o[[0, 2]] - o2[[0, 2]]), brake, None),
                                         (0, wait2, None), (towards(P[v, [0, 2]] - o2[[0, 2]]), 300, P[v]), (0, 120, None)]
                                tape, tr, alive = _lift_try(w, start, legs + climb)
                                got = arrive(tape, tr, v, alive) if len(tr) else None
                                if got is not None and (best is None or got["ticks"] < best["ticks"]):
                                    best = dict(got, chain=True, legs=climb, fan=o)
                        if best is not None:
                            keep(u, v, start, best, bounds, s)
    log("fan lifts: %d edges for a paper ball (%d of them with both fan bounds, %d through two fans)" % (len(out), sum(e["both"] for e in out.values()),
                                                                                                  sum(e.get("chain", False) for e in out.values())))
    return out


def _link_rules(rel, g, region, tk, trafos, radius: float, mechs: list) -> dict:
    """What the relaxed graph's links (gap, bridge: a break the fine search tries) mean for the states. A link is
    taken as the straight way between its two places (ASSUMED: it says nothing of how it is crossed):
    * one that enters a transformer's range on the way arrives as that kind, as an edge into the range does;
    * one that runs into a closed gate or a standing push board (their pieces as put) is not there while it is so.
    {(global u, global v): (kind index or -1, mask of mechanism bits that must be set)}."""
    out = {}
    T = [(np.asarray(t["pos"], float), coarse.KIND_INDEX[t["kind"]]) for t in trafos]
    block = []
    for m, mc in enumerate(mechs):
        if mc["type"] in ("gate", "board") and mc.get("pieces"):
            cen = np.mean([c for c, _h, _a, _t in mc["pieces"]], axis=0)
            size = max(float(np.linalg.norm(c - cen) + np.linalg.norm(h)) for c, h, _a, _t in mc["pieces"])
            block.append((m, mc["pieces"], cen, size))
    P = g.pos
    for gu in np.flatnonzero(region):
        a = P[gu]
        for v, _c, k in rel.fwd[gu]:
            if k not in ("gap", "bridge") or not region[v]:
                continue
            b = P[v]
            ab = b - a
            L2 = max(float(ab @ ab), 1e-12)
            ek, best = -1, -1.0
            if tk[v] < 0:
                for c, kind in T:
                    t = min(max(float((c - a) @ ab) / L2, 0.0), 1.0)
                    if np.linalg.norm(a + t * ab - c) < radius and t > best:
                        ek, best = kind, t
            mask = 0
            for m, pieces, cen, size in block:
                t = min(max(float((cen - a) @ ab) / L2, 0.0), 1.0)
                if np.linalg.norm(a + t * ab - cen) > size + BALL_R:
                    continue
                n = int(np.sqrt(L2) / 0.5) + 1
                pts = a + ab * (np.arange(n + 1)[:, None] / n)
                if _hits(pts, pieces, 0.1):
                    mask |= 1 << m
            if ek >= 0 or mask:
                out[(int(gu), int(v))] = (ek, mask)
    return out


GATE_FULL = 4.5           # a gate counts as open once it has slid this far (the edges it clears are made for 4.95)
BOARD_DOWN = 60.0         # a push board counts as down once it has turned this far over (degrees)


def cork_edges(g, info: dict, sector: int | None = None) -> list:
    """The cork bridges (P_Modul_29) on the coarse level: (u, v, cost, name), both ways. A wood or paper ball walks
    over one, a stone ball breaks it (READ and MEASURED, catalogue P_Modul_29). Its ends are those of its planks'
    hulls along its long axis (the placement's own x, as they are put): the places on the road beyond each end (0.5
    to 4 out, within 2 of its middle line, within 2.5 of the end's height) get an edge to the place beyond the other
    end nearest across. Cost: the way's length (ASSUMED: the bridge is crossed at about the follower's pace; MEASURED
    0.62 to 1.02 game s over its 14.8)."""
    out = []
    for p in info.get("placements", ()):
        if p["type"] != "P_Modul_29" or (sector is not None and p.get("sector") != sector):
            continue
        V = [np.array(h["vertices"], float) for h in info.get("hulls", ()) if h.get("placement") == p["name"]]     # (in world, as put)
        if not V:
            continue
        V = np.vstack(V)
        o = np.array(p["position"], float)
        ax = np.array(p["x_axis"], float)
        ax[1] = 0.0
        ax /= np.linalg.norm(ax)
        side = np.cross([0.0, 1.0, 0.0], ax)
        t = (V - o) @ ax
        ends = []
        for sign, lim in ((-1.0, t.min()), (1.0, t.max())):
            e = o + lim * ax
            e[1] = float(V[np.abs(t - lim) < 1.0, 1].max())     # the end's top
            ends.append((sign, lim, e))
        rel = g.pos - o
        along, across = rel @ ax, rel @ side
        floor = g.kind == 0
        groups = []
        for sign, lim, e in ends:
            groups.append(np.flatnonzero(floor & (sign * (along - lim) > 0.5) & (sign * (along - lim) < 4.0) & (np.abs(across) <= 2.0)
                                         & (np.abs(g.pos[:, 1] - e[1] - 2.0) < 2.5)))
        a, b = groups
        for us, vs in ((a, b), (b, a)):
            for u in us:
                if not len(vs):
                    break
                v = int(vs[np.argmin(np.abs(across[vs] - across[u]))])
                out.append((int(u), v, float(np.linalg.norm(g.pos[v] - g.pos[u])), p["name"]))
    return out


# Tolerances for matching sampled coarse floor places to the hulls, not game rules (ASSUMED).
_MECH_HEIGHT = 0.8
_MECH_BANK = 2.5
_MECH_ALIGN = 1.25


def _span(g, P, floor, lo, hi, axis, top, bank=_MECH_BANK):
    """Opposite banks of a local footprint, matched along the other horizontal axis."""
    cross = 2 - axis
    keep = (floor & (abs(P[:, 1] - top - BALL_R) <= _MECH_HEIGHT)
            & (P[:, cross] >= lo[cross]) & (P[:, cross] <= hi[cross]))
    a = np.flatnonzero(keep & (P[:, axis] <= lo[axis]) & (P[:, axis] >= lo[axis] - bank))
    b = np.flatnonzero(keep & (P[:, axis] >= hi[axis]) & (P[:, axis] <= hi[axis] + bank))
    # Prefer the nearest aligned bank, not a diagonal across to some other road.
    pairs = set()
    for us, vs in ((a, b), (b, a)):
        for u in us:
            choices = vs[abs(P[vs, cross] - P[u, cross]) <= _MECH_ALIGN]
            if len(choices):
                v = int(choices[np.argmin(np.linalg.norm(P[choices] - P[u], axis=1))])
                pairs.add(tuple(sorted((int(u), v))))
    return [(u, v, float(np.linalg.norm(g.pos[v] - g.pos[u]))) for a, b in sorted(pairs) for u, v in ((a, b), (b, a))]


def _box_span(way, lo, hi):
    """Does a straight segment of this already available coarse edge touch a local box?"""
    a, d = way[:-1], np.diff(way, axis=0)
    flat = abs(d) < 1e-9
    denom = np.where(flat, 1.0, d)
    t, t2 = (lo - a) / denom, (hi - a) / denom
    enter = np.maximum(0, np.where(flat, -np.inf, np.minimum(t, t2)).max(axis=1))
    leave = np.minimum(1, np.where(flat, np.inf, np.maximum(t, t2)).min(axis=1))
    return bool(((enter <= leave) & ~np.any(flat & ((a < lo) | (a > hi)), axis=1)).any())


# ASSUMED: coarse geometry matching only, not game rules.
_FLOAT_PITCH_TOL = 0.25
_FLOAT_LINE_TOL = 0.25
_FLOAT_AXIS_COS = 0.98
_FLOAT_BANK_MIN, _FLOAT_BANK_MAX = 0.5, 4.0
_FLOAT_BANK_WIDTH, _FLOAT_BANK_HEIGHT = 2.0, 2.5
_FLOAT_TOP_TOL = 0.8
_FLOAT_EXIT_MARGIN = BALL_R  # ASSUMED: allow one ball radius beyond the ballistic bound


def _float_footprint(vertices):
    """Counterclockwise horizontal convex hull in the row frame (monotone chain)."""
    points = sorted(set(map(tuple, vertices[:, [0, 2]])))

    def half(seq):
        out = []
        for x, z in seq:
            while len(out) > 1:
                (ax, az), (bx, bz) = out[-2:]
                if (bx - ax) * (z - az) > (bz - az) * (x - ax):
                    break
                out.pop()
            out.append((x, z))
        return out[:-1]

    return np.array(half(points) + half(reversed(points)))


def _float_departures(g, region, rows):
    """Geometric witnesses only, from each end's banks to the screened coarse floor."""
    floor = np.flatnonzero(region & (g.kind == 0))
    for row in rows:
        if "banks" not in row:
            continue
        P = (g.pos - row["origin"]) @ row["frame"].T
        footprint = row["footprint"]
        if len(footprint) < 3:
            continue
        XZ = P[floor][:, [0, 2]]
        # A landing must be strictly outside the convex footprint. Compute distance to its edges.
        inside, dist = np.ones(len(XZ), bool), np.full(len(XZ), np.inf)
        for a, b in zip(footprint, np.roll(footprint, -1, axis=0)):
            d, q = b - a, XZ - a
            inside &= d[0] * q[:, 1] - d[1] * q[:, 0] >= -1e-8
            f = np.clip((q @ d) / (d @ d), 0, 1)
            dist = np.minimum(dist, np.linalg.norm(q - f[:, None] * d, axis=1))
        floor_out = floor[~inside]
        x, y, z = P[floor_out].T
        dist = dist[~inside]
        row["departures"] = []
        for end, sign in enumerate((1, -1)):
            banks = [(int(u), P[u, 0]) for u in row["banks"][end] if region[u]]
            if not banks:
                continue
            candidates = []
            boxes = row["bounds"] if end == 0 else row["bounds"][::-1]
            for n, (lo, hi) in enumerate(boxes, 1):
                drop = hi[1] + BALL_R - y
                # ASSUMED optimistic upper bound: use maximum departure speed, ignore drag and collisions.
                dt = np.sqrt(2 * np.maximum(drop, 0) / ph.GRAVITY)
                max_along = np.clip(x, lo[0], hi[0])
                faces = [(coarse.FLOAT_MINUS, z < lo[2], max_along, z - lo[2]),
                         (coarse.FLOAT_PLUS, z > hi[2], max_along, z - hi[2])]
                if n == row["boxes"]:
                    face = hi[0] if sign > 0 else lo[0]
                    faces.append((coarse.FLOAT_END, sign * (x - face) > 0, face, z - np.clip(z, lo[2], hi[2])))
                for direction, outside, along, across in faces:
                    # The side segment spans the box's axial half-width on either side of its centre.
                    air = np.hypot(x - along, across)
                    mask = np.zeros(len(floor_out), int)
                    for k, (limit, speed) in row["limits"].items():
                        if n <= limit:
                            reach = speed * dt + _FLOAT_EXIT_MARGIN
                            mask[outside & (drop >= -_FLOAT_BANK_HEIGHT) & (dist <= reach) & (air <= reach)] |= 1 << k
                    # ASSUMED cost: axial travel from the entry to the departure face, plus horizontal flight.
                    cost = sign * along + air
                    candidates.extend((int(floor_out[i]), float(cost[i]), coarse.float_side(n, direction, int(mask[i])))
                                      for i in np.flatnonzero(mask))
            if candidates:
                row["departures"].append((banks, sign, candidates))


def float_rows(g, info, sector):
    """Straight runs of adjacent P_Modul_41 placements, with geometry in the row's own frame."""
    boxes = sorted((p for p in info.get("placements", ()) if p["type"] == "P_Modul_41"
                    and p.get("sector") == sector), key=lambda p: p["name"])
    if len(boxes) < 2:
        return []
    from . import things
    catalogue = things.load()
    nums = catalogue["P_Modul_41"].numbers
    pitch = nums["pitch"]["v"]
    time_factor = catalogue.rules["units"]["numbers"]["time_factor"]["v"]

    def speed(kind):
        seconds = nums[kind + "_seconds_per_box"]["v"]
        # Upper bound on speed = shortest measured time, converted from game to physics seconds.
        return pitch / ((min(seconds) if isinstance(seconds, list) else seconds) * time_factor)
    # The catalogue has no separate numeric short-row limit. Take the lower bound of the
    # measured multi-box outcomes (not the failed, one-box run), as its short-row guidance does.
    wood_max = min(min(v) for v in nums["wood_boxes_by_entry"]["v"].values() if isinstance(v, list))
    paper_speed, wood_speed = speed("paper"), speed("wood")
    centers = np.array([p["position"] for p in boxes], float)
    axes = np.array([p["x_axis"] for p in boxes])[:, [0, 2]]  # local X runs across the gap; the hinge is transverse
    neighbors = [[] for _ in boxes]
    for i, a in enumerate(centers):
        for j in range(i + 1, len(boxes)):
            d = centers[j] - a
            flat = np.linalg.norm(d[[0, 2]])
            if (abs(np.linalg.norm(d) - pitch) <= _FLOAT_PITCH_TOL and flat > 0
                    and abs(d[1]) <= _FLOAT_LINE_TOL
                    and np.all(abs(axes[[i, j]] @ (d[[0, 2]] / flat)) >= _FLOAT_AXIS_COS)):
                neighbors[i].append(j)
                neighbors[j].append(i)
    hulls = {}
    for h in info.get("hulls", ()):
        if len(h.get("vertices", ())):
            hulls.setdefault(h.get("placement"), []).append(h["vertices"])
    rows, seen = [], set()
    for i in range(len(boxes)):
        if i in seen:
            continue
        group, todo = {i}, [i]
        while todo:
            for j in neighbors[todo.pop()]:
                if j not in group:
                    group.add(j)
                    todo.append(j)
        seen.update(group)
        ends = sorted(j for j in group if len(neighbors[j]) == 1)
        if len(ends) != 2 or any(len(neighbors[j]) > 2 for j in group):
            continue  # isolated boxes or ambiguous branches do not make a row
        first, last = ends
        o = centers[first]
        ax = centers[last] - o
        ax[1] = 0.0
        if not np.linalg.norm(ax):
            continue
        ax /= np.linalg.norm(ax)
        frame = np.array([ax, [0., 1., 0.], np.cross([0., 1., 0.], ax)])
        Q = (centers[list(group)] - o) @ frame.T
        if np.any(abs(Q[:, 2]) > _FLOAT_LINE_TOL) or np.ptp(Q[:, 1]) > _FLOAT_LINE_TOL:
            continue
        order = sorted(group, key=lambda j: (centers[j] - o) @ ax)
        if order[0] != first or order[-1] != last or any(b not in neighbors[a] for a, b in zip(order, order[1:])):
            continue
        names = [boxes[j]["name"] for j in order]
        prefix = "P_Modul_41_"
        end_name = names[-1][len(prefix):] if names[0].startswith(prefix) and names[-1].startswith(prefix) else names[-1]
        n = len(names)
        allowed = (1 << coarse.PAPER) | ((1 << coarse.WOOD) if n <= wood_max else 0)
        row = dict(name=names[0] + "-" + end_name, placements=names, boxes=n, allowed=allowed,
                   pos=centers[order].mean(axis=0).tolist(), centers=centers[order].tolist(), edges=[], frame=frame, origin=o)
        # The prose mentions one stone box but there is no numeric limit: continue to allow none.
        row["limits"] = {coarse.PAPER: (n, paper_speed), coarse.WOOD: (wood_max, wood_speed), coarse.STONE: (0, 0.0)}
        rows.append(row)
        if not all(name in hulls for name in names):
            continue  # report the row, but do not guess the outline of a missing hull
        V = [(np.vstack(hulls[name]) - o) @ frame.T for name in names]
        lo, hi = V[0][:, 0].min(), V[-1][:, 0].max()
        # The top plane of each end box supplies its bank height.
        tops = [v[:, 1].max() + BALL_R for v in V]
        all_v = np.vstack(V)
        row["footprint"] = _float_footprint(all_v)
        row["bounds"] = [(v.min(axis=0), v.max(axis=0)) for v in V]
        row["top_slab"] = (np.array([lo, min(tops) - _FLOAT_TOP_TOL, all_v[:, 2].min()]),
                           np.array([hi, max(tops) + _FLOAT_TOP_TOL, all_v[:, 2].max()]))
        P = (g.pos - o) @ frame.T
        banks = []
        for dist, top in ((lo - P[:, 0], tops[0]), (P[:, 0] - hi, tops[-1])):
            banks.append(np.flatnonzero((g.kind == 0) & (dist >= _FLOAT_BANK_MIN) & (dist <= _FLOAT_BANK_MAX)
                                        & (abs(P[:, 2]) <= _FLOAT_BANK_WIDTH) & (abs(P[:, 1] - top) <= _FLOAT_BANK_HEIGHT)))
        row["banks"] = banks
        for a, b in (banks, banks[::-1]):
            if not len(b):
                continue
            for u in a:
                # Lateral alignment first; use length to break ties along the same lane.
                v = min(b, key=lambda v: (abs(P[v, 2] - P[u, 2]), np.linalg.norm(P[v] - P[u])))
                row["edges"].append((int(u), int(v), float(np.linalg.norm(P[v] - P[u]))))
    return rows


def _float_screen(g, rel, region, lifts, rows):
    """Exclude top places and pre-existing crossings of a row, even in the without counterfactual.

    Screen at ball-centre height (ASSUMED), not at the base of the boxes: unrelated lower roads
    and paths over the row stay. For launches and lifts use the cached way, never run one again.
    """
    slabs = [r for r in rows if "top_slab" in r]
    if not slabs:
        return region, set()
    keep = region.copy()
    skip = set()
    edges = [(int(u), v, k) for u in np.flatnonzero(region) for v, _c, k in rel.fwd[u] if region[v]]
    flight = [(u, v, k) for u, v, k in edges if k == "launch"] + [(u, v, "lift") for u, v, *_ in lifts if region[u] and region[v]]
    direct = [(u, v, k) for u, v, k in edges if k != "launch"]
    uv = np.array([(u, v) for u, v, _k in direct], dtype=int).reshape(-1, 2)
    for r in slabs:
        o, A = r["origin"], r["frame"]
        lo, hi = r["top_slab"]
        P = (g.pos - o) @ A.T
        keep &= ~np.all((P >= lo) & (P <= hi), axis=1)
        a, b = P[uv[:, 0]], P[uv[:, 1]]
        candidates = np.flatnonzero(np.all((np.minimum(a, b) <= hi) & (np.maximum(a, b) >= lo), axis=1))
        for i in candidates:
            if _box_span(P[uv[i]], lo, hi):
                skip.add(direct[i])
        for u, v, k in flight:
            e = g.info.get("launches" if k == "launch" else "lifts", {}).get((u, v), {})
            way = e.get("way", ())
            if _box_span((np.asarray(way) - o) @ A.T if len(way) > 1 else P[[u, v]], lo, hi):
                skip.add((u, v, k))
    return keep, skip


def _t_clear_edges(g, P, floor, pieces, fallen, mesh, name):
    """Conservatively restore a short supported floor crossing of the vacated upright foot.

    The cache has no floor places inside the wall. Without fixed-floor evidence do not invent an edge.
    Return also the crossings if the board is removed entirely (the fallen board need not be clear then).
    """
    if mesh is None:
        return [], []
    foot = min(pieces, key=lambda v: v[:, 1].min())
    lo, hi = foot.min(0), foot.max(0)
    # ASSUMED: the floor at the foot is near placement height (local 0); match only nearby banks.
    choices = {(u, v, c) for ax in (0, 2) for u, v, c in _span(g, P, floor, lo, hi, ax, 0.0, BALL_R + _MECH_BANK) if u < v}
    clear, removed = [], []
    for u, v, c in sorted(choices):
        if any(w == v and kind == "roll" for w, _c, kind in g.adj[u]) and any(w == u and kind == "roll" for w, _c, kind in g.adj[v]):
            continue
        # ASSUMED: sample every 0.5 and allow 0.35 in floor height for the grid discretisation.
        t = np.linspace(0, 1, int(np.ceil(c / 0.5)) + 1)[:, None]
        way = g.pos[u] + t * (g.pos[v] - g.pos[u])
        for point in way:
            contacts = [q for q in mesh.contacts(point, radius=BALL_R, margin=0.35)
                        if mesh.entities[q.entity].get("placement") != name]
            if not coarse._held(mesh, contacts) or any(q.gap < -0.35 for q in contacts):
                break
        else:
            both = [(u, v, c), (v, u, c)]
            removed.extend(both)
            local = P[u] + t * (P[v] - P[u])
            # ASSUMED: conservative ball clearance against each fallen piece's local bounding box.
            if not any(_box_span(local, f.min(0) - BALL_R + 0.2, f.max(0) + BALL_R - 0.2) for f in fallen):
                clear.extend(both)
    return clear, removed


def road_state_edges(g, rel, info, sector, region, mesh=None):
    """Region-only rules for 34/37. No new graph places, no simulated motion: only endpoints in two fixed poses."""
    types = {"P_Modul_34", "P_Modul_37"}
    from . import things
    cat = things.load()
    rock_fall = float(cat["P_Modul_34"].numbers["drop_distance"]["v"])        # MEASURED, the catalogue's
    t_angle = float(cat["P_Modul_37"].numbers["fallen_angle_deg"]["v"])       # MEASURED, the catalogue's
    hulls = {}
    for h in info.get("hulls", ()):
        hulls.setdefault(h.get("placement"), []).append(h)
    floor = (g.kind == 0) & region
    launches = g.info.get("launches", {})
    for p in info.get("placements", ()):
        if p["type"] not in types or p.get("sector") != sector:
            continue
        name = p["name"]
        o = np.asarray(p["position"], float)
        A = np.array([p[ax + "_axis"] for ax in "xyz"], float)
        A /= np.linalg.norm(A, axis=1)[:, None]
        P = (g.pos - o) @ A.T
        parts = [(h, (np.asarray(h["vertices"], float) - o) @ A.T) for h in hulls.get(name, ()) if len(h["vertices"]) >= 4]
        edges, guards, removed = [], {}, []

        def push_at(lo, hi, axis, sign, top, face_high=False):
            cross = 2 - axis
            face = hi[axis] if sign > 0 else lo[axis]
            across = P[:, cross] - (lo[cross] + hi[cross]) / 2
            height = P[:, 1] - top - BALL_R
            d = sign * (P[:, axis] - face) - BALL_R
            # face_high: a ball whose centre is between the face's bottom and top meets the face, whatever the road
            # under it is (on a rail or in a groove the centre sits lower than on a flat floor); otherwise the
            # centre is within _MECH_HEIGHT of resting on the piece's own floor height (ASSUMED)
            level = ((P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1])) if face_high else (abs(height) <= _MECH_HEIGHT)
            # ASSUMED: contact band outside the face; one coarse place per usable approach side.
            ids = np.flatnonzero(floor & (d >= -0.2) & (d <= _MECH_BANK)
                                 & (abs(across) <= _MECH_ALIGN) & level)
            return int(ids[np.argmin(d[ids] ** 2 + across[ids] ** 2 + height[ids] ** 2)]) if len(ids) else None

        if p["type"] == "P_Modul_34":
            typ = "falling_rock"
            rock = next((v for h, v in parts if "Schiebestein" in h.get("body", "")), None)
            box = next((v for h, v in parts if "Kiste" in h.get("body", "")), None)
            if rock is not None and box is not None:
                lo, hi = rock.min(0), rock.max(0)
                # Catalogue: the block drops along its slide; its top fills the upper road across local X.
                edges += [(coarse.ROCK_OVER, u, v, c, 0, "over_falling_rock")
                          for u, v, c in _span(g, P, floor, lo, hi, 0, hi[1] - rock_fall)]
                for sign in (-1, 1):         # MEASURED: a stone ball can push along either local Z direction.
                    u = push_at(box.min(0), box.max(0), 2, sign, box[:, 1].min(), face_high=True)
                    if u is not None:
                        edges.append((coarse.ROCK_PUSH, u, u, 0.0, 0, "drop_falling_rock"))
                # The lower way is sealed by the fallen block. ASSUMED: its box padded by the radius less 0.2
                # (clearance) at the sides and below; not above its top: a ball whose centre is above the fallen
                # block's top goes over it (the block's top is then part of the upper road, READ), it is not sealed.
                pad = BALL_R - 0.2
                fall = np.array([0, rock_fall, 0])
                lower_lo, lower_hi = lo - fall - pad, hi - fall + pad
                lower_hi[1] = hi[1] - rock_fall
                # Existing relaxed straight links are at most 25 long; inspect longer launch paths separately.
                # Read their stored geometry only. Do not make a new launch or certify any unknown break.
                near = np.all((P >= lower_lo - 25) & (P <= lower_hi + 25), axis=1)
                for u in np.flatnonzero(region):
                    for v, _c, kind in rel.fwd[u]:
                        if not region[v] or (kind != "launch" and not near[u]):
                            continue
                        stored = launches.get((u, v)) if kind == "launch" else None
                        way = (np.asarray(stored["way"]) - o) @ A.T if stored is not None and len(stored.get("way", ())) > 1 else P[[u, v]]
                        off = _box_span(way, lower_lo, lower_hi)
                        on = kind not in ("gap", "bridge") and _box_span(way, lo - pad, hi + pad)
                        if on or off:
                            guards[int(u), int(v), kind] = (on, off)
        else:
            typ = "t_board"
            if parts:
                pieces = [v for _h, v in parts]
                head = max(pieces, key=lambda v: np.ptp(v[:, 2]))  # catalogue: the crossbar is the wide piece along Z
                # Catalogue joints.37.geom (READ): the Bridge frame origin is the hinge, along local Z.
                # MEASURED final angle, toward local -X: one static pose transform, not a trajectory.
                frame = next((h["frame"] for h, _v in parts if h.get("frame") is not None), None)
                pivot = (np.asarray(frame[3][:3]) - o) @ A.T if frame is not None else np.array([-2.381, -0.625, 0.0])
                a = np.radians(t_angle)
                c, s = np.cos(a), np.sin(a)
                turn = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
                head_down = (head - pivot) @ turn + pivot
                lo, hi = head_down.min(0), head_down.max(0)
                edges += [(coarse.T_OVER, u, v, c, 0, "over_t_board") for u, v, c in _span(g, P, floor, lo, hi, 2, hi[1])]
                foot = min(pieces, key=lambda v: v[:, 1].min())
                # Catalogue: from the centre of the narrow road on +X, toward -X. ASSUMED wood run-up exists.
                u = push_at(foot.min(0), foot.max(0), 0, 1, 0.0)
                if u is not None:
                    edges.append((coarse.T_PUSH, u, u, 0.0, 0, "push_t_board"))
                clear, removed = _t_clear_edges(g, P, floor, pieces, [(v - pivot) @ turn + pivot for v in pieces], mesh, name)
                edges += [(coarse.T_CLEAR, u, v, c, 0, "past_t_board") for u, v, c in clear]
        yield dict(name=name, type=typ, edges=edges), guards, removed


def flap_state_edges(g, rel, info, sector, region, lifts):
    """Catalogue lane crossings for P_Modul_19 and guards on the cached edges, in the placement frame."""
    placements = [p for p in info.get("placements", ()) if p["type"] == "P_Modul_19" and p.get("sector") == sector]
    if not placements:
        return
    from . import things
    nums = things.load()["P_Modul_19"].numbers
    gaps, lanes = (nums[k]["v"] for k in ("gap_x", "lane_z"))
    height = nums["standing_flap_height"]["v"]
    floor = (g.kind == 0) & region
    ground, flights = [], []
    for u in np.flatnonzero(region):
        for v, _c, kind in rel.fwd[u]:
            if region[v]:
                (flights if kind == "launch" else ground).append((int(u), int(v), kind))
    flights += [(u, v, "lift") for u, v, *_ in lifts if region[u] and region[v]]
    uv = np.array([(u, v) for u, v, _ in ground], dtype=int).reshape(-1, 2)
    for p in placements:
        name = p["name"]
        o = np.asarray(p["position"], float)
        frame = np.array([p[ax + "_axis"] for ax in "xyz"], float)
        frame /= np.linalg.norm(frame, axis=1)[:, None]
        P = (g.pos - o) @ frame.T
        parts = [(np.asarray(h["vertices"]) - o) @ frame.T for h in info.get("hulls", ())
                 if h.get("placement") == name and len(h["vertices"]) >= 4]
        plates = []
        for z0, z1 in (lanes["lane_A"], lanes["lane_B"]):
            # The plate spans its lane; the wire and central steel piece do not.
            parts_in_lane = [v for v in parts if z0 < v[:, 2].mean() < z1]
            if parts_in_lane:
                v = max(parts_in_lane, key=lambda v: np.ptp(v[:, 2]))
                plates.append((v.min(0), v.max(0)))
        edges, guards, deck_edges = [], {}, {}
        if len(plates) != 2:
            continue  # no geometry for this placement; leave its breaks unresolved
        alo, ahi = plates[0]
        blo, bhi = plates[1]
        # B pose: the A plate stands at its minus-X end. ASSUMED box thickness from the other
        # upright plate's X extent; use the catalogue upright height over the local road datum.
        wall_a = (alo.copy(), np.array([alo[0] + bhi[0] - blo[0], height, ahi[2]]))
        walls = (wall_a, (blo, bhi))
        top = (alo[1] + ahi[1]) / 2
        # ASSUMED: only the ball-centre band for the cached A deck, not the road underneath it.
        slab_lo = alo + [0, BALL_R - _MECH_HEIGHT, 0]
        slab_hi = ahi + [0, BALL_R + _MECH_HEIGHT, 0]
        supported = floor & np.all((P > slab_lo) & (P < slab_hi), axis=1)
        # Height of the filled B gap is inferred from A; do not count a hypothetical bridge
        # or a flight between the banks as rolling on this surface.
        b_slab_lo = np.array([gaps["lane_B"][0], top + BALL_R - _MECH_HEIGHT, lanes["lane_B"][0]])
        b_slab_hi = np.array([gaps["lane_B"][1], top + BALL_R + _MECH_HEIGHT, lanes["lane_B"][1]])
        for side, lane in enumerate(("lane_A", "lane_B")):
            x0, x1 = gaps[lane]
            z0, z1 = lanes[lane]
            lo, hi = np.array([x0, top, z0]), np.array([x1, top, z1])
            # ASSUMED: use the same end/height/alignment tolerances on either lane. Before
            # tipping there may be no rest sample right at the lip (the ball meets the wall first).
            bank = BALL_R + _MECH_BANK
            if side or not supported.any():
                for u, v, c in _span(g, P, floor, lo, hi, 0, top, bank):
                    if side or not any(w == v and k not in ("gap", "bridge", "launch") for w, _c, k in g.adj[u]):
                        edges.append((coarse.FLAP_OVER, u, v, c, side, "over_flap"))
            wl, wh = walls[side]
            sign, face, lip = (1, wh[0], x1) if side else (-1, wl[0], x0)
            d = sign * (P[:, 0] - face) - BALL_R
            x, y, z = P.T
            along_lane = (z >= z0) & (z <= z1)
            # ASSUMED contact band at the upright face. A rail/groove ball need not be at
            # ball-radius height over the road: any centre between the face bottom and top can push.
            sources = np.flatnonzero(region & along_lane & (sign * (x - lip) >= -1e-8)
                                     & (d >= -0.2) & (d <= _MECH_BANK) & (y >= wl[1]) & (y <= wh[1]))
            far_lip = x0 if side else x1
            to_end = sign * (far_lip - x)
            banks = np.flatnonzero(floor & along_lane & (to_end >= -1e-8) & (to_end <= bank)
                                   & (abs(y - top - BALL_R) <= _MECH_HEIGHT))
            for u in sources:
                aligned = banks[abs(z[banks] - z[u]) <= _MECH_ALIGN]
                if len(aligned):
                    v = int(aligned[np.argmin(np.linalg.norm(P[aligned] - P[u], axis=1))])
                    edges.append((coarse.FLAP_PUSH, int(u), v, float(np.linalg.norm(P[v] - P[u])), 1 - side, "push_flap"))

        # ASSUMED: test passage through the faces' actual X bands. Do not seal rolling
        # along a bank alongside the wall. Allow ball clearance below and at the sides,
        # but do not extend above the face. Flights use only their stored paths.
        pad = BALL_R - 0.2
        wall_boxes = [(lo - [0, pad, pad], hi + [0, 0, pad]) for lo, hi in walls]
        low = np.minimum(P[uv[:, 0]], P[uv[:, 1]])
        high = np.maximum(P[uv[:, 0]], P[uv[:, 1]])
        near = supported[uv].any(axis=1)
        for lo, hi in [*wall_boxes, (slab_lo, slab_hi), (b_slab_lo, b_slab_hi)]:
            near |= ((low <= hi) & (high >= lo)).all(axis=1)

        def screen(edge, way):
            u, v, kind = edge
            # A steel rail has its own support even if its centreline is at deck height.
            # Still guard it below if the new standing face intersects that centreline.
            rolling = kind in ("roll", "step") and g.kind[u] == g.kind[v] == 0
            deck_a = supported[u] or supported[v] or (rolling and _box_span(way, slab_lo, slab_hi))
            deck_b = rolling and _box_span(way, b_slab_lo, b_slab_hi)
            # Dependent edges disappear without the flap. Relaxed links between two
            # independent banks remain breaks, not proof of a crossing.
            if deck_a or deck_b:
                deck_edges[edge] = 0 if deck_a else 1
            on = deck_b or (len(way) > 1 and _box_span(way, *wall_boxes[1]))  # B upright blocks in A
            off = deck_a or (len(way) > 1 and _box_span(way, *wall_boxes[0]))
            if on or off:
                guards[edge] = (on, off)

        for i in np.flatnonzero(near):
            screen(ground[i], P[uv[i]])
        for edge in flights:
            u, v, kind = edge
            stored = g.info.get("launches" if kind == "launch" else "lifts", {}).get((u, v), {})
            way = stored.get("way", ())
            local = (np.asarray(way) - o) @ frame.T if len(way) > 1 else ()
            screen(edge, local)
        yield dict(name=name, type="flap", edges=edges, deck_edges=deck_edges), guards


def _pendulum_shape(vertices):
    """Local XZ convex footprint (monotone chain), extruded between bottom and top."""
    points = sorted(set(map(tuple, vertices[:, [0, 2]])))

    def half(points):
        hull = []
        for p in points:
            while len(hull) >= 2:
                (x, z), (u, v) = hull[-2:]
                if (u - x) * (p[1] - z) - (v - z) * (p[0] - x) > 1e-9:
                    break
                hull.pop()
            hull.append(p)
        return hull[:-1]

    rim = np.array(half(points) + half(reversed(points)))
    if len(rim) < 3:
        return None
    d = np.roll(rim, -1, axis=0) - rim
    n = np.c_[d[:, 1], -d[:, 0]] / np.linalg.norm(d, axis=1)[:, None]
    return (np.r_[np.c_[n[:, 0], np.zeros(len(n)), n[:, 1]], [[0, 1, 0], [0, -1, 0]]],
            np.r_[-(n * rim).sum(axis=1), -vertices[:, 1].max(), vertices[:, 1].min()])


def _pendulum_hit(a, b, shape):
    """Clip a centre line to a rod piece's local XZ projection and its centre-height band."""
    normals, offsets = shape
    d = b - a
    side, step = normals @ a + offsets, normals @ d
    parallel = abs(step) < 1e-9
    if np.any(parallel & (side > 1e-8)):
        return None
    t = -side[~parallel] / step[~parallel]
    enters = step[~parallel] < 0
    first = max(0., t[enters].max(initial=-np.inf))
    last = min(1., t[~enters].min(initial=np.inf))
    return a + ((first + last) / 2) * d if first <= last else None


def _placement_frame(p):
    a = np.array([p[ax + "_axis"] for ax in "xyz"], float)
    return a / np.linalg.norm(a, axis=1)[:, None]


def _pendulum_contacts(vertices, reach, end, height):
    """First circle/AABB contacts in the local half on this end, for each signed turn.

    ASSUMED: bounds of each target hull and overlap with the wooden target's height band.
    Phase zero is local +/-X. Positive angles follow the right hand about placement +Y.
    """
    lo, hi = np.min(vertices, axis=0), np.max(vertices, axis=0)
    if lo[1] > height[1] or hi[1] < height[0]:
        return
    # The circle enters the box at rest or on one of its four boundary lines.
    bounds = np.array([lo[[0, 2]], hi[[0, 2]]]) * end / reach
    low, high = bounds.min(0), bounds.max(0)
    for sign in (-1, 1):
        angles = [0.0]
        angles += [np.arccos(x) for x in (low[0], high[0]) if 0 <= x <= 1]
        angles += [np.arcsin(-sign * z) for z in (low[1], high[1]) if 0 <= -sign * z <= 1]
        for angle in sorted(angles):
            xz = np.array([np.cos(angle), -sign * np.sin(angle)])
            if np.all(xz >= low - 1e-9) and np.all(xz <= high + 1e-9):
                x, z = end * reach * xz
                yield sign, angle, sign * np.array([z, 0., -x])
                break


def _fan_box_stacks(info, sector):
    """Placement centres above a fan and within its catalogue horizontal box, in its frame."""
    from . import fans

    half = fans.rules().half
    placed = [p for p in info["placements"] if p.get("sector") == sector]
    boxes = [p for p in placed if p["type"] == "P_Box"]
    for fan in placed:
        if fan["type"] != fans.TYPE:
            continue
        o, frame = np.asarray(fan["position"], float), _placement_frame(fan)
        stack = []
        for p in boxes:
            x, y, z = (p["position"] - o) @ frame.T
            if y > 0 and max(abs(x), abs(z)) <= half:
                stack.append(p["name"])
        if stack:
            yield fan, tuple(sorted(stack))


def pendulum_edges(g, rel, info, sector, region, mesh, without, lifts, guards, skipped):
    """Static, supported crossings of P_Modul_17. No angle state, sweep simulation or new places."""
    placements = [p for p in info["placements"] if p.get("sector") == sector and p["name"] not in without]
    rods = [p for p in info["placements"] if p.get("sector") == sector and p["type"] == "P_Modul_17"]
    if not rods:
        return [], [], set()
    from . import things
    from .export import SKIP_TEMPLATES

    nums = things.load()["P_Modul_17"].numbers
    reach = float(nums["arm_reach"]["v"])
    # The catalogue table is scoped to this map: placement names are reused across maps.
    limits = nums["l13_stopper_deg"]["v"] if info.get("map") == "Level_13.NMO" else {}
    hulls = {}
    for h in info.get("hulls", ()):
        if len(h.get("vertices", ())):
            hulls.setdefault(h["placement"], []).append(h)
    rows, restored, skip = [], [], set()
    places = np.flatnonzero(region)
    skipped = set(skipped)
    for p in rods:
        name = p["name"]
        o = np.asarray(p["position"], float)
        A = _placement_frame(p)
        parts = [h for h in hulls.get(name, ()) if h.get("body") == "Modul17_Dreharme"]
        if not parts:
            continue
        boxes, shapes = [], []
        for part in parts:
            V = (np.asarray(part["vertices"], float) - o) @ A.T
            lo, hi = V.min(0), V.max(0)
            shape = _pendulum_shape(V)
            if shape is not None:
                boxes.append((lo, hi))
                shapes.append(shape)
        if not shapes:
            continue
        lo = np.min([a for a, _ in boxes], axis=0)
        hi = np.max([b for _, b in boxes], axis=0)
        # The central convex piece is centred on the hinge (only its height is needed).
        mid_lo, mid_hi = min(boxes, key=lambda box: abs((box[0][0] + box[1][0]) / 2))
        y = (mid_lo[1] + mid_hi[1]) / 2
        centre = o + y * A[1]
        P = (g.pos - o) @ A.T
        edges, blocked, crossings, supported = {}, set(), set(), {}
        pushes = {}
        limit = limits.get(name)
        angle = np.deg2rad(limit) if limit is not None else None
        if angle is not None:
            cs, sn = np.cos(angle), np.sin(angle)
            turn = np.array([[cs, -sn], [sn, cs]])  # row vectors (local X, Z), about +Y

        def allowed(a, b, hit):
            if angle is None:
                return True
            push = (b - a)[[0, 2]]
            x, z = r = hit[[0, 2]]
            torque = z * push[0] - x * push[1]
            if torque * angle <= 0 or not np.linalg.norm(push):
                return True
            # ASSUMED: movement of the struck part along the push must clear one diameter.
            return (r @ turn - r) @ push / np.linalg.norm(push) >= 2 * BALL_R

        def hits(a, b):
            return [h for s in shapes if (h := _pendulum_hit(a, b, s)) is not None]

        def held(point, clearance=True):
            if mesh is None:
                return False
            # ASSUMED: the same height clearance as a vacated T-board foot. Keep the post, base,
            # other placements and any fixed stopper in the test for obstruction.
            contacts = [q for q in mesh.contacts(point, radius=BALL_R, margin=0.35)
                        if not (mesh.entities[q.entity].get("placement") == name
                                and mesh.entities[q.entity].get("name") == "Modul17_Dreharme")]
            fixed = [q for q in contacts if not mesh.entities[q.entity].get("placed")
                     or mesh.entities[q.entity].get("type") in coarse.FIXED_TYPES]
            return (not clearance or not any(q.gap < -0.35 for q in contacts)) and coarse._held(mesh, fixed)

        def add(u, v, c, points):
            if not all(allowed(P[u], P[v], q) for q in points):
                blocked.add((u, v))
            else:
                edges[u, v] = c
                dx, dz = (P[v] - P[u])[[0, 2]]
                for x, _, z in points:
                    torque = z * dx - x * dz
                    if abs(torque) > 1e-9 and abs(x) > 1e-9:
                        pushes.setdefault((int(np.sign(x)), int(np.sign(torque))), set()).add(int(u))

        # A cached rolling/rail edge can already cross (notably analytic rails). Turn just that
        # crossing into a push. Leave cached locations, overpasses and hypothetical breaks alone.
        for u in places:
            for v, c, kind in g.adj[u]:
                if (kind not in ("roll", "rail", "rail end", "step") or not region[v] or (u, v, kind) in guards
                        or (u, v) in skipped or (u, v, kind) in skipped):
                    continue
                if np.any(np.maximum(P[u], P[v]) < lo) or np.any(np.minimum(P[u], P[v]) > hi):
                    continue
                points = hits(P[u], P[v])
                if not points:
                    continue
                if kind == "step":
                    if name not in without and not all(allowed(P[u], P[v], q) for q in points):
                        skip.add((u, v, kind))
                        blocked.add((u, v))
                    continue
                crossings.add((u, v))
                if name not in without:
                    skip.add((u, v, kind))
                    add(u, v, c, points)

        # ASSUMED bank matching as for a T-board foot, but with the actual ball-centre heights
        # on the two banks (including rails/grooves). Try the nearest supported opposite bank.
        for low, high in boxes:
            for ax in (0, 2):
                cross = 2 - ax
                bank = (region & (abs(P[:, 1] - (low[1] + high[1]) / 2) <= (high[1] - low[1]) / 2 + _MECH_HEIGHT)
                        & (P[:, cross] >= low[cross] - _MECH_ALIGN) & (P[:, cross] <= high[cross] + _MECH_ALIGN))
                l = np.flatnonzero(bank & (P[:, ax] < low[ax]) & (P[:, ax] >= low[ax] - BALL_R - _MECH_BANK))
                r = np.flatnonzero(bank & (P[:, ax] > high[ax]) & (P[:, ax] <= high[ax] + BALL_R + _MECH_BANK))
                for starts, ends in ((l, r), (r, l)):
                    for u in starts:
                        vs = ends[(abs(P[ends, cross] - P[u, cross]) <= _MECH_ALIGN)
                                  & (abs(P[ends, 1] - P[u, 1]) <= _MECH_HEIGHT)]
                        for v in sorted(vs, key=lambda v: np.linalg.norm(P[v] - P[u])):
                            if (u, v) in crossings:
                                break
                            if any(k in ("roll", "rail", "rail end") and w == v for w, _, k in rel.fwd[u]):
                                continue   # a genuinely existing, independent edge
                            points = hits(P[u], P[v])
                            if not points:
                                continue
                            pair = tuple(sorted((u, v)))
                            if pair not in supported:
                                c = float(np.linalg.norm(g.pos[v] - g.pos[u]))
                                way = g.pos[u] + np.linspace(0, 1, int(np.ceil(c / 0.5)) + 1)[:, None] * (g.pos[v] - g.pos[u])
                                supported[pair] = c if all(held(q) for q in way) else None
                            c = supported[pair]
                            if c is None:
                                continue
                            for a, b in ((u, v), (v, u)):
                                if (a, b) in crossings or any(k in ("roll", "rail", "rail end") and w == b for w, _, k in rel.fwd[a]):
                                    continue
                                if name in without:
                                    restored.append((int(a), int(b), c, "roll"))
                                else:
                                    add(a, b, c, points)
                            break
        if name in without:
            continue
        # For trajectories only consider a catalogue limit and actual contact along a recorded path.
        if limit is not None:
            for pairs, kind in ((g.info.get("launches", {}), "launch"),
                                ({(u, v): g.info["lifts"][u, v] for u, v, *_ in lifts}, "lift")):
                for (u, v), e in pairs.items():
                    if (u, v) in skipped or (u, v, kind) in skipped or not (region[u] and region[v]) or e.get("way") is None:
                        continue
                    path = (np.asarray(e["way"]) - o) @ A.T
                    if np.any(path.max(0) < lo) or np.any(path.min(0) > hi):
                        continue
                    if any(not allowed(a, b, q) for a, b in zip(path, path[1:]) for q in hits(a, b)):
                        skip.add((u, v, kind))
                        blocked.add((u, v))
        candidates, contacts = [], {}
        board_fall = {nm.split("/")[0]: d["fall"] for nm, d in g.info.get("boards", {}).items()
                      if d["sector"] == sector and "fall" in d}
        # Use each wooden end's vertical envelope, not the pivot's or the fixed base's.
        tips = {end: max(boxes, key=lambda b: end * (b[1] if end > 0 else b[0])[0])
                for end in (-1, 1)}
        for other in placements:
            typ, nm = other["type"], other["name"]
            if nm == name or typ.startswith(SKIP_TEMPLATES) or not typ.startswith(("P_Modul_", "P_Box", "P_Ball_")):
                continue
            geometry = [h["vertices"] for h in hulls.get(nm, ())] or [[other["position"]]]
            for vertices in geometry:
                V = (np.asarray(vertices, float) - o) @ A.T
                if nm in board_fall or (typ == "P_Box" and nm in hulls):
                    for end, (tip_lo, tip_hi) in tips.items():
                        for sign, a, tangent in _pendulum_contacts(V, reach, end, (tip_lo[1], tip_hi[1])):
                            key = (nm, end, sign)
                            if key not in contacts or a < contacts[key][0]:
                                contacts[key] = a, tangent
            for vertices in geometry:
                V = (np.asarray(vertices, float) - o) @ A.T
                mn, mx = V.min(0), V.max(0)
                # ASSUMED: broad-phase sweep disk and height envelope; a placement without
                # hulls supplies only its origin, with a ball radius of proximity margin.
                margin = 0. if hulls.get(nm) else BALL_R
                if (np.linalg.norm(np.clip(0., mn[[0, 2]], mx[[0, 2]])) <= reach + margin
                        and mn[1] <= hi[1] + margin and mx[1] >= lo[1] - margin):
                    candidates.append(dict(name=nm, type=typ, basis="ASSUMED"))
                    break
        operations = {}
        for (nm, end, sign), (a, tangent) in contacts.items():
            if angle is not None and sign * angle > 0 and a >= abs(angle):
                continue
            if nm in board_fall and (tangent @ A) @ board_fall[nm] <= 1e-9:
                continue
            if sites := pushes.get((-end, sign)):
                operations.setdefault(nm, set()).update(sites)
        theta = np.linspace(0, 2 * np.pi, 73)
        sweep = centre + reach * (np.cos(theta)[:, None] * A[0] + np.sin(theta)[:, None] * A[2])
        # ASSUMED: a clear ring around the axle is not needed for routing; report unsupported
        # ball-centre samples here as a hole warning, not as a constraint on any edge.
        circle = centre + (reach / 2) * (np.cos(theta[::9])[:, None] * A[0] + np.sin(theta[::9])[:, None] * A[2])
        # ASSUMED: flag a broad gap (at least half the ring), not a local notch.
        # Terrain changes up to one radius lower are not a hole.
        below = [np.array([0., h, 0.]) for h in np.linspace(0, BALL_R, 5)]
        hole = mesh is not None and 2 * sum(not any(held(q - d, clearance=False) for d in below)
                                             for q in circle[:-1]) >= len(circle) - 1
        rows.append(dict(name=name, pos=o.tolist(), sweep=sweep, reach=reach, limit=limit, hole=hole, candidates=candidates,
                         edges=[(int(u), int(v), c) for (u, v), c in sorted(edges.items())], operations=operations,
                         blocked=[(int(u), int(v)) for u, v in sorted(blocked)]))
    return rows, sorted(set(restored)), skip



def _pendulum_mechanisms(g, sector, without, pendulums, stacks, mechs, readers, guards, lifts):
    """Attach in-place operations to target bits and require cleared stacks for the cached lifts."""
    boards = {d["name"].split("/")[0]: d for d in mechs if d["type"] == "board"}
    by_board = {nm.split("/")[0]: (nm, d) for nm, d in g.info.get("boards", {}).items() if d["sector"] == sector}
    lift_data = g.info.get("lifts", {})
    by_box = {}
    for fan, stack in stacks:
        if fan["name"] in without or without.intersection(stack):
            continue
        bit = 1 << len(mechs)
        mc = dict(name=stack[0], type="box", edges=[], placements=stack, fan=fan["name"])
        mechs.append(mc)
        readers.append(lambda _scene: 0)
        for nm in stack:
            by_box.setdefault(nm, []).append(mc)
        o = np.asarray(fan["position"], float)
        for u, v, *_ in lifts:
            e = lift_data[u, v]
            columns = [e["fan"]]
            if e.get("chain") and e.get("legs"):
                columns.append(e["legs"][1][2])  # this leg leads to the second fan's recorded origin
            if any(np.allclose(o, p, rtol=0, atol=1e-6) for p in columns):
                edge = (u, v, "lift")
                on, off = guards.get(edge, (0, 0))
                guards[edge] = (on | bit, off)
    for j, rod in enumerate(pendulums):
        linked, by_target = set(), {}
        for nm, sites in sorted(rod["operations"].items()):
            if nm in by_board:
                if nm not in boards:
                    # A target can fall even if the cache contains no crossing over it.
                    name, d = by_board[nm]
                    boards[nm] = dict(name=name, type="board", edges=[], pieces=d.get("pieces"))
                    mechs.append(boards[nm])
                    readers.append(lambda _scene: 0)
                targets = [boards[nm]]
            else:
                targets = by_box.get(nm, ())
            for mc in targets:
                _, push_sites = by_target.setdefault(mc["name"], (mc, set()))
                push_sites.update(sites)
                linked.update(mc.get("placements", (nm,)))
        for mc, sites in by_target.values():
            typ, kind = ((coarse.PENDULUM_BOARD, "pendulum_board") if mc["type"] == "board"
                         else (coarse.PENDULUM_BOX, "pendulum_box"))
            mc["edges"].extend((typ, u, u, 0., j, kind) for u in sorted(sites))
        rod["candidates"] = [c for c in rod["candidates"] if c["name"] not in linked]



# ASSUMED coarse geometry: allow a one-sample band at a road lip, and try exits across
# up to four ball radii. Beyond one radius without support we report an unmeasured flight.
_ELEVATOR_LIP = 1.5
_ELEVATOR_REACH = 4 * BALL_R
_ELEVATOR_STEP = 1.4  # ASSUMED: catalog prose estimate for a ball coming with speed, not a tabulated limit


def _elevator_floor_lip(tri, direction, support, rim):
    """First inward road boundary on this radial line, with the height of the actual face there."""
    if not len(tri):
        return support
    q = tri[:, :, [0, 2]]
    across = q @ np.array([-direction[1], direction[0]])
    nxt = np.roll(across, -1, axis=1)
    hit = ((across <= 0) & (nxt > 0)) | ((across > 0) & (nxt <= 0))
    valid = hit.sum(axis=1) == 2
    a, b = tri[valid], np.roll(tri[valid], -1, axis=1)
    rows = across[valid]
    t = np.divide(rows, rows - nxt[valid], out=np.zeros_like(rows), where=hit[valid])
    p = (a + (b - a) * t[:, :, None])[hit[valid]].reshape(-1, 2, 3)
    along = p[:, :, [0, 2]] @ direction
    order = along.argsort(axis=1)
    x, y = np.take_along_axis(along, order, 1), np.take_along_axis(p[:, :, 1], order, 1)
    r = float(support[[0, 2]] @ direction)
    h = float(support[1])
    # Walk adjacent faces on the same surface; a lower/upper deck is not the continuation of this lip.
    while True:
        f = np.clip((r - x[:, 0]) / np.maximum(x[:, 1] - x[:, 0], 1e-9), 0, 1)
        match = ((x[:, 0] < r - 1e-6) & (x[:, 1] >= r - 1e-6) &
                 (abs(y[:, 0] + f * (y[:, 1] - y[:, 0]) - h) <= 0.05))  # ASSUMED contact tolerance
        if not match.any():
            return np.array([r * direction[0], h, r * direction[1]])
        j = np.argmin(np.where(match, x[:, 0], np.inf))
        if x[j, 0] <= rim:
            h = float(np.interp(rim, x[j], y[j]))
            return np.array([rim * direction[0], h, rim * direction[1]])
        r, h = x[j, 0], y[j, 0]


def _elevator_sites(g, P, o, frame, region, deck, low, high, mesh):
    """Fixed floor/rail lips near the footprint, sampled by the unchanged coarse graph."""
    span = -deck[1][:-2].min() + _ELEVATOR_REACH + BALL_R
    radial = np.linalg.norm(P[:, [0, 2]], axis=1)
    trial = np.flatnonzero(region & (radial > 0) & (radial <= span) &
                           (P[:, 1] >= low - BALL_R) & (P[:, 1] <= high + BALL_R * 2))
    triangles = np.empty((0, 3, 3))
    if mesh is not None and len(trial):
        keep = (~mesh.placed & ~mesh.rail & (mesh.N[:, 1] > 0.5))
        A, B, C = (v[keep] - o for v in (mesh.A, mesh.B, mesh.C))
        keep = ((np.minimum(np.minimum(A, B), C) <= [span, high + BALL_R, span]) &
                (np.maximum(np.maximum(A, B), C) >= [-span, low - BALL_R, -span])).all(axis=1)
        triangles = np.stack([A[keep], B[keep], C[keep]], axis=1) @ frame.T
    for u in trial:
        q = P[u].copy()
        if g.kind[u] == 1:
            lane = g.lanes[g.lane[u]]
            if any(g.lane[v] == g.lane[u] and radial[v] < radial[u] - 1e-6 for v, _c, _k in g.adj[u]):
                continue  # not the platform-facing end of this rail
            tube = lane["radius"]
            rise = np.sqrt(max(0.05, (BALL_R + tube) ** 2 - (lane["gauge"] / 2) ** 2))
            q[1] += tube - rise
        elif mesh is not None:
            contacts = [c for c in mesh.contacts(g.pos[u], margin=0.05)
                        if not (mesh.placed[c.triangle] or mesh.rail[c.triangle]) and c.normal[1] > 0.5
                        and mesh.N[c.triangle, 1] > 0.5]
            if not contacts:
                continue
            q = (contacts[0].point - o) @ frame.T
        else:
            q[1] -= BALL_R
        r = np.linalg.norm(q[[0, 2]])
        if r < 1e-6:
            continue
        direction = q[[0, 2]] / r
        v = np.array([direction[0], 0., direction[1]])
        normals, offsets = deck
        advance = normals @ v
        rim = (-offsets[advance > 1e-9] / advance[advance > 1e-9]).min()
        if g.kind[u] == 0:
            q = _elevator_floor_lip(triangles, direction, q, rim)
        gap = float(q @ v - rim)
        # A cached edge sample can be a ball radius back from the lip (e.g. on a ramp).
        edge_nodes = g.info.get("edge_node", ())
        snap = _ELEVATOR_LIP + (BALL_R if u < len(edge_nodes) and edge_nodes[u] else 0.0)
        if -1e-6 <= gap <= _ELEVATOR_REACH and r - q @ v <= snap and low <= q[1] <= high:
            yield int(u), q, max(0., gap)


def _elevator_edges(g, info, sector, region, mesh, without, first_bit):
    """Build region-only ride edges. Temporary places stand for the loaded platforms in a chain."""
    from . import things
    type_id = "P_Modul_03"
    placed = [p for p in info.get("placements", ()) if p["type"] == type_id and p.get("sector") == sector]
    if not placed:
        return [], []
    catalog = things.load()[type_id]
    nums = {k: n["v"] for k, n in catalog.numbers.items()}
    loaded, rise, band = (nums[k] for k in ("full_load_eq", "per_block", "rest_band"))
    tolerance = band + _ELEVATOR_STEP
    # The catalogue says paper can push with a run-up. Like the T-board, do not check the run-up here.
    kinds = [k for k in coarse.KIND_INDEX if catalog.by_ball[k]["verdict"] == "pushes"]
    tops = {k: nums["all_off_top_by_ball"][k] for k in kinds}
    blocks = len(catalog.bodies) - 1  # all placed bodies except the floor
    low = min(loaded, *(loaded + rise - nums["ball_sink"][k] for k in kinds)) - tolerance
    high = max(tops.values()) + tolerance
    hulls = {}
    for h in info.get("hulls", ()):
        hulls.setdefault(h.get("placement"), []).append(h)
    decks = []
    for p in placed:
        name = p["name"]
        if name in without:
            continue
        o, frame = np.asarray(p["position"], float), _placement_frame(p)
        bodies = {}
        for h in hulls.get(name, ()):
            bodies.setdefault(h["body"], []).append((np.asarray(h["vertices"]) - o) @ frame.T)
        floors = bodies.pop(type_id + "_Floor", ())
        gates = bodies.get(type_id + "_Gate", ())
        deck = _pendulum_shape(np.vstack(floors)) if floors else None
        d = dict(name=name, type="elevator", edges=[], rides={}, pos=p["position"], origin=o, frame=frame,
                 deck=deck, node=len(g.pos) + len(decks), bit=1 << (first_bit + len(decks)))
        decks.append(d)
        if deck is None or len(gates) < 3:
            continue
        gate = np.vstack(gates)
        centre = (gate.max(0) + gate.min(0)) / 2
        outward = centre[[0, 2]] / np.linalg.norm(centre[[0, 2]])
        tangent = np.array([-outward[1], outward[0]])
        posts = sorted((part[:, [0, 2]] @ tangent for part in gates if part[:, 1].min() < centre[1]),
                       key=lambda a: a.mean())
        if len(posts) != 2:
            continue
        aperture = (posts[0].max() + BALL_R, posts[1].min() - BALL_R)
        gate_plane = max(part[:, [0, 2]].dot(outward).max() for part in gates)

        def doorway(q, outward=outward, tangent=tangent, plane=gate_plane, aperture=aperture):
            r = float(q[[0, 2]] @ outward)
            if r < plane:
                return False
            z = (q[[0, 2]] @ tangent) * plane / r
            return aperture[0] <= z <= aperture[1]

        d["doorway"] = doorway
        d["walls"] = {n: [_pendulum_shape(v) for v in parts] for n, parts in bodies.items()}
        P = (g.pos - o) @ frame.T
        d["sites"] = list(_elevator_sites(g, P, o, frame, region, deck, low, high, mesh))
        d["entries"] = [u for u, q, gap in d["sites"] if doorway(q) and gap <= BALL_R + _ELEVATOR_LIP
                        and abs(q[1] - loaded) <= tolerance]
    points = [d["origin"] + (loaded + BALL_R) * d["frame"][1] for d in decks]
    for d in decks:
        if "entries" not in d:
            continue
        o, frame = d["origin"], d["frame"]
        normals, offsets = d["deck"]
        entries = list(d["entries"])
        exits = [(u, q, gap, 0, None) for u, q, gap in d["sites"] if u not in d["entries"]]
        for other in decks:
            if other is d or "doorway" not in other:
                continue
            delta = other["origin"] - o
            q = delta @ frame.T
            t = -delta @ other["frame"].T
            distance = np.linalg.norm(q[[0, 2]])
            if distance < 1e-6 or not other["doorway"](t):
                continue
            dn, do = other["deck"]
            step = dn @ np.array([t[0], 0., t[2]])
            f = (-do[step > 1e-9] / step[step > 1e-9]).min()
            q = (delta + f * (t * [1, 0, 1]) @ other["frame"] + loaded * other["frame"][1]) @ frame.T
            step = normals @ np.array([q[0], 0., q[2]])
            gap = np.linalg.norm(q[[0, 2]]) * (1 - (-offsets[step > 1e-9] / step[step > 1e-9]).min())
            if -_ELEVATOR_LIP <= gap <= _ELEVATOR_REACH and low <= q[1] <= high:
                exits.append((other["node"], q, max(0., float(gap)), other["bit"], other["name"]))
        # Include the platform as an input only where an earlier ride can land; add those edges below.
        d["exits"] = exits
        d["entries"] = entries
    for d in decks:
        if "exits" not in d:
            continue
        o, frame = d["origin"], d["frame"]
        for v, q, gap, bit, target in d["exits"]:
            ray = np.array([q[0], 0., q[2]])
            blockers = []
            for name, parts in d["walls"].items():
                if name == type_id + "_Gate" and d["doorway"](q):
                    continue
                if any(shape is not None and _pendulum_hit(np.zeros(3), ray, (shape[0][:-2], shape[1][:-2]))
                       is not None for shape in parts):
                    blockers.append(name)
            for kind in kinds:
                k = coarse.KIND_INDEX[kind]
                for n in range(max(1, len(blockers)), blocks + 1):
                    top = tops[kind] if n == blocks else loaded + rise * n - nums["ball_sink"][kind]
                    if abs(top - q[1]) > tolerance:
                        continue
                    side = coarse.elevator_side(k, n, bit)
                    for u in (*d["entries"], d["node"]):
                        if u == v:
                            continue
                        a = points[u - len(g.pos)] if u >= len(g.pos) else g.pos[u]
                        b = points[v - len(g.pos)] if v >= len(g.pos) else g.pos[v]
                        c = np.linalg.norm(a - o) + np.linalg.norm(b - o)  # ASSUMED planning cost, not travel time
                        d["edges"].append((coarse.ELEVATOR, u, v, c, side, "ride_elevator"))
                        d["rides"][u, v, side] = dict(blocks=n, walls=sorted(blockers), platform_top=float(o[1] + top),
                                                      exit_top=float(o[1] + q[1]), gap=gap, flight=gap > BALL_R,
                                                      with_speed=bool(q[1] > top + band),
                                                      exit_at=(o + q @ frame).tolist(), target_elevator=target)
    # Remove only virtual inputs that cannot be landed on by any ride. No fabricated relaxed gaps to these places.
    incoming = {v for d in decks for _, _, v, _, _, _ in d["edges"] if v >= len(g.pos)}
    for d in decks:
        if d["node"] not in incoming:
            d["edges"] = [e for e in d["edges"] if e[1] != d["node"]]
            d["rides"] = {key: ride for key, ride in d["rides"].items() if key[0] != d["node"]}
    return decks, points


# ASSUMED: allowances for coarse floor/rail samples, not timing or game thresholds.
_SWING_LIP = 1.5
_TIMED_SUPPORT_MARGIN = 0.35
_TIMED_SUPPORT_STEP = 0.5
_TIMED_ROLL = ("roll", "rail", "rail end")


def _timed_fixed_support(mesh, point):
    if mesh is None:
        return False
    contacts = [q for q in mesh.contacts(point, radius=BALL_R, margin=_TIMED_SUPPORT_MARGIN)
                if not mesh.entities[q.entity].get("placed") or mesh.entities[q.entity].get("type") in coarse.FIXED_TYPES]
    return not any(q.gap < -_TIMED_SUPPORT_MARGIN for q in contacts) and coarse._held(mesh, contacts)


def _swing_banks(g, P, region, parts, numbers):
    """Project each open end to the catalogue reach; match the nearby fixed-road samples."""
    reach = max(numbers["end_reach_record"]["v"])
    half = numbers["bucket_half_length"]["v"]
    hinge = numbers["hinge_above_origin"]["v"]
    angle = np.radians(max(numbers["amplitude_record"]["v"]))
    # The two broad bottom pieces extend to the open ends; the narrow side pieces are not entrances.
    deck = [V for V in parts if V[:, 2].min() < 0 < V[:, 2].max()]
    if not deck:
        return []
    lo, hi = np.vstack(deck).min(0), np.vstack(deck).max(0)
    y = hinge + (hi[1] - hinge) * np.cos(angle) + half * np.sin(angle)
    bank = (region & (abs(P[:, 0]) >= reach - _SWING_LIP) & (abs(P[:, 0]) <= reach + _SWING_LIP)
            & (P[:, 2] >= lo[2] - _MECH_ALIGN) & (P[:, 2] <= hi[2] + _MECH_ALIGN)
            & (abs(P[:, 1] - y - BALL_R) <= 2 * _SWING_LIP))
    left, right = (np.flatnonzero(bank & (sign * P[:, 0] > 0)) for sign in (-1, 1))
    pairs = set()
    for starts, ends in ((left, right), (right, left)):
        for u in starts:
            vs = ends[abs(P[ends, 2] - P[u, 2]) <= _MECH_ALIGN]
            if len(vs):
                v = min(vs, key=lambda v: (abs(P[v, 2] - P[u, 2]), abs(P[v, 0])))
                pairs.add(tuple(sorted((int(u), int(v)))))
    return [(a, b, float(np.linalg.norm(g.pos[a] - g.pos[b])))
            for u, v in sorted(pairs) for a, b in ((u, v), (v, u))]


def _sandbag_banks(g, rel, P, region, V, mesh, guards, skipped):
    """Find genuinely supported crossings transverse to the local Z drive, not across a gap."""
    shape = _pendulum_shape(V)
    if shape is None:
        return [], set(), False
    lo, hi = V.min(0) - BALL_R, V.max(0) + BALL_R
    shape = (shape[0], shape[1] - BALL_R)

    def hits(u, v):
        return _pendulum_hit(P[u], P[v], shape) is not None

    bank = (region & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]) & (P[:, 2] >= lo[2]) & (P[:, 2] <= hi[2]))
    left = np.flatnonzero(bank & (P[:, 0] <= lo[0]) & (P[:, 0] >= lo[0] - _MECH_BANK))
    right = np.flatnonzero(bank & (P[:, 0] >= hi[0]) & (P[:, 0] <= hi[0] + _MECH_BANK))
    def rail_supported(u, v):
        # Analytic rails already have sampled support, including sloping sections where a
        # straight chord does not touch both rails. Require a continuous, near-aligned run.
        lane = g.lane[u]
        if lane < 0 or g.lane[v] != lane:
            return False
        a, b = sorted((P[u, 0], P[v, 0]))
        seen, todo = {u}, [u]
        while todo:
            w = todo.pop()
            if w == v:
                return True
            for q, _, kind in rel.fwd[w]:
                if (kind == "rail" and q not in seen and region[q] and g.lane[q] == lane
                        and a <= P[q, 0] <= b and abs(P[q, 2] - P[u, 2]) <= _MECH_ALIGN
                        and (w, q, kind) not in guards and (w, q, kind) not in skipped and (w, q) not in skipped):
                    seen.add(q)
                    todo.append(q)
        return False

    supported = {}
    pairs = set()
    for starts, ends in ((left, right), (right, left)):
        for u in starts:
            vs = ends[abs(P[ends, 2] - P[u, 2]) <= _MECH_ALIGN]
            # Match like to like (sampled floor or analytic rail), taking the nearest supported opposite site.
            vs = vs[g.kind[vs] == g.kind[u]]
            for v in sorted(vs, key=lambda v: np.linalg.norm(P[v] - P[u])):
                if not hits(u, v):
                    continue
                pair = tuple(sorted((int(u), int(v))))
                if pair not in supported:
                    c = float(np.linalg.norm(g.pos[u] - g.pos[v]))
                    way = np.linspace(g.pos[u], g.pos[v], int(np.ceil(c / _TIMED_SUPPORT_STEP)) + 1)
                    supported[pair] = c if (rail_supported(u, v) and rail_supported(v, u)
                                            or all(_timed_fixed_support(mesh, q) for q in way)) else None
                if supported[pair] is not None:
                    pairs.add(pair)
                    break
    if not pairs:
        return [], set(), False

    # A nearby cached rolling detour around the *stationary* bag proves this is not a road cut.
    # Do not use jumps, conditionals or any path that intersects the inflated sack.
    nearby = (region & (P[:, 0] >= lo[0] - _MECH_BANK) & (P[:, 0] <= hi[0] + _MECH_BANK)
              & (P[:, 2] >= lo[2] - _MECH_BANK) & (P[:, 2] <= hi[2] + _MECH_BANK)
              & (P[:, 1] >= lo[1]) & (P[:, 1] <= hi[1]))
    screen, local = set(), {}
    mid_x = (lo[0] + hi[0]) / 2
    for u in np.flatnonzero(nearby):
        for v, _, kind in rel.fwd[u]:
            key = (int(u), int(v), kind)
            if kind not in _TIMED_ROLL or not region[v] or key in guards or key in skipped or key[:2] in skipped:
                continue
            if hits(u, v):
                # Cut the cached road at one cross-section. Do not isolate its intermediate
                # samples (keep other attachments); the window edge still spans the whole sack.
                if (P[u, 0] < mid_x) != (P[v, 0] < mid_x):
                    screen.add(key)
            elif nearby[v]:
                local.setdefault(u, []).append(v)

    # Require a detour for each supported pair on its own road layer. A parallel track
    # or higher road is not a bypass for a blocked lane.
    def bypass(u, v):
        low, high = sorted((P[u, 1], P[v, 1]))
        seen, todo = {u}, [u]
        while todo:
            w = todo.pop()
            if w == v:
                return True
            for q in local.get(w, ()):
                if (q not in seen and g.kind[q] == g.kind[u]
                        and low - _MECH_HEIGHT <= P[q, 1] <= high + _MECH_HEIGHT):
                    seen.add(q)
                    todo.append(q)
        return False

    if all(bypass(u, v) and bypass(v, u) for u, v in pairs):
        return [], set(), True
    return [(a, b, supported[u, v]) for u, v in sorted(pairs) for a, b in ((u, v), (v, u))], screen, False


def _timed_edges(g, rel, info, sector, region, mesh, without, guards, skipped):
    from .things import load

    out, restored, screened = [], [], set()
    skipped = set(skipped)
    for p in info["placements"]:
        typ, name = p["type"], p["name"]
        if typ not in ("P_Modul_08", "P_Modul_26") or p.get("sector") != sector:
            continue
        swing = typ == "P_Modul_08"
        if swing and name in without:
            continue
        A, o = _placement_frame(p), np.asarray(p["position"], float)
        P = (g.pos - o) @ A.T
        body = "P_Modul_08_Schaukel" if swing else "P_Modul_26_Sack"
        parts = [(np.asarray(h["vertices"], float) - o) @ A.T for h in info.get("hulls", ())
                 if h.get("placement") == name and h.get("body") == body and len(h["vertices"])]
        numbers = load()[typ].numbers
        edges, screen, bypass, safety = [], set(), False, False
        if swing:
            if parts:
                edges = _swing_banks(g, P, region, parts, numbers)
            depth = np.mean(numbers["l8_safety_floor_depth"]["v"])
            safety = _timed_fixed_support(mesh, o + (BALL_R - depth) * A[1])
        elif parts:
            edges, screen, bypass = _sandbag_banks(g, rel, P, region, np.vstack(parts), mesh, guards, skipped)
        if name in without:
            # Let the original roll/rail edges through and restore any missing supported span.
            restored.extend((u, v, c, "roll") for u, v, c in edges
                            if not any(w == v and k in _TIMED_ROLL for w, _, k in rel.fwd[u]))
        else:
            screened.update(screen)
            out.append(dict(name=name, type="swing" if swing else "sandbag", pos=p["position"], edges=edges,
                            bypass=bypass, safety=safety))
    return out, restored, screened


def state_graph_for(g, info: dict, sector: int, world, operate_cost: float, cork: bool = False, without=()):
    """The coarse level of a sector over the ball's kind and its mechanisms' states (`coarse.StateGraph`), and how to
    read those states from the kernel's scene: (graph, reader(kind name, scene) -> state index). Only the places
    of the sector's region take part: its reset point, its goal and what is placed in it, 60 wider. cork: with the
    region-only cork bridges, falling rocks, T boards, two-way flaps, box float rows, pendulums, elevators, swings and sandbags; the solver that rolls leaves them out. without: names of placements taken as not there (a what-if): a transformer has no range, a fan no lifts, a
    cork bridge, push board or seesaw no way over it; a gate is no obstacle (the ways it clears are always there).
    A removed rock has no upper bridge and never seals the lower route; a removed T board leaves only the
    supported crossings of its old foot. No change to the cached places or edges."""
    without = {str(w) for w in without}
    stacks = list(_fan_box_stacks(info, sector)) if cork else []
    absent = {nm for _, stack in stacks if without.intersection(stack) for nm in stack}
    without.update(absent)
    from . import trafo
    from .kernel import _hinge_angle
    from .launch import FOLLOW_SPEED
    rel = getattr(g, "_relaxed", None)
    if rel is None:
        rel = g._relaxed = coarse.Relaxed(g)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    pts = [np.array(info["reset_points"][sector - 1], float)] + ([np.array(anchors[sector], float)] if anchors[sector] is not None else [])
    pts += [np.array(p["position"], float) for p in info["placements"] if p.get("sector") == sector]
    P = np.array(pts)
    region = np.all((g.pos >= P.min(0) - 60.0) & (g.pos <= P.max(0) + 60.0), axis=1)
    R = trafo.rules()
    cost = (R.body_after + 1) / 66.0 * FOLLOW_SPEED
    trafos = [t for t in trafo.of(info, sector) if t["name"] not in without]
    tk = np.full(len(g.pos), -1, np.int64)
    for t in trafos:
        tk[np.linalg.norm(g.pos - t["pos"], axis=1) < R.radius] = coarse.KIND_INDEX[t["kind"]]

    # a lift that ends in a transformer's range on the way (its run stopped there) ends as that kind, whatever place it is kept for
    gone_fans = [np.array(p["position"], float) for p in info["placements"] if p["name"] in without and p["type"] == "P_Modul_18"]
    lifts = [(u, v, e["cost"], coarse.KIND_INDEX[e["ends_in"]] if e.get("ends_in") and any(t["kind"] == e["ends_in"] for t in trafos) else -1)
             for (u, v), e in g.info.get("lifts", {}).items() if e.get("sector") == sector
             and not any(np.linalg.norm(np.asarray(e["fan"], float) - f) < 1.0 for f in gone_fans)]
    # a run-up rolled as another kind (it starts in a transformer's range) is that kind's only where the range is this
    # sector's: elsewhere the ball there is not that kind
    skip = [(w, v) for (w, v), e in g.info.get("launches", {}).items() if e.get("ball", "wood") != "wood" and tk[w] != coarse.KIND_INDEX[e["ball"]]]
    mechs, readers = [], []
    props = world.props
    for name, d in sorted(g.info.get("gates", {}).items()):
        if d["sector"] != sector or not d["edges"]:
            continue
        if name in without:                                # no gate: the ways it clears are plain ways
            rel = rel.with_edges([(u, v, c, k) for u, v, c, k in d["edges"]])
            continue
        edges = [(coarse.GATE_THROUGH, u, v, c, 0, k) for u, v, c, k in d["edges"]]
        op = d["operate"]
        if cork:
            # the region analysis: the place in front of the target plate may be on a rail (the lower road is a rail
            # at some gates), and a wood ball may push it open too (coarse.GATE_WOOD)
            op = g.nearest(np.asarray(d["face"], float) - (BALL_R + 0.3) * np.asarray(d["axis"], float), 3.0, floor_only=False)
        if op >= 0:
            edges.append((coarse.GATE_OPERATE, op, op, 0.0, coarse.GATE_WOOD if cork else 0, "operate"))
        k = next((i for i, p in enumerate(props) if p["name"].startswith(name + "/P_Modul_01_Pusher") and p.get("leader") == i), None)
        if k is None:
            continue
        axis, rest = np.array(d["axis"], float), np.array(props[k]["rest"], float)
        mechs.append(dict(name=name, type="gate", edges=edges, axis=axis, operate=op, pieces=d.get("pieces")))
        readers.append(lambda sc, k=k, axis=axis, rest=rest: int((sc[k, :3] - rest) @ axis >= GATE_FULL))
    for name, d in sorted(g.info.get("boards", {}).items()):
        if d["sector"] != sector or not d["edges"] or name.split("/")[0] in without:
            continue
        edges = [(coarse.BOARD_PUSH, u, v, c, 0, k) for u, v, c, k in d["edges"]]
        edges += [(coarse.BOARD_OVER, u, v, c, 0, k) for u, v, c, k in d["edges"] + d["back"]]
        k = next((i for i, p in enumerate(props) if p["name"].startswith(name + "/") and p.get("hinge") and p.get("leader") == i), None)
        if k is None:
            continue
        mechs.append(dict(name=name, type="board", edges=edges, pieces=d.get("pieces")))
        readers.append(lambda sc, k=k: int(np.degrees(_hinge_angle(world.prop_consts, k, sc[k, 0], sc[k, 1], sc[k, 2])) >= BOARD_DOWN))
    for name, d in sorted(g.info.get("seesaws", {}).items()):
        if d["sector"] != sector or not (d["edges"] or d["other"]) or name.split("/")[0] in without:
            continue
        edges = [(coarse.SEESAW, u, v, c, 0, k) for u, v, c, k in d["edges"]] + [(coarse.SEESAW, u, v, c, 1, k) for u, v, c, k in d["other"]]
        k = next((i for i, p in enumerate(props) if p["name"].startswith(name + "/") and p.get("hinge") and p.get("leader") == i), None)
        if k is None:
            continue
        mechs.append(dict(name=name, type="seesaw", edges=edges))
        # put at 10 degrees, it falls to its first low end (about +9.6 from where it was put); tipped over, -29.6
        readers.append(lambda sc, k=k: int(np.degrees(_hinge_angle(world.prop_consts, k, sc[k, 0], sc[k, 1], sc[k, 2])) < -10.0))
    guards = {}
    if cork:
        for mc, rules, removed in road_state_edges(g, rel, info, sector, region, getattr(world, "mesh", None)):
            if mc["name"] in without:
                # No falling block or box, or no T-board wall and no crossbar in the gap.
                if removed:
                    rel = rel.with_edges([(u, v, c, "roll") for u, v, c in removed])
                continue
            bit = 1 << len(mechs)
            for edge, (on, off) in rules.items():
                a, b = guards.get(edge, (0, 0))
                guards[edge] = (a | (bit if on else 0), b | (bit if off else 0))
            mechs.append(mc)
            readers.append(lambda _scene: 0)  # the region-only bits start at rest; never read from physics
    if cork:
        for mc, rules in flap_state_edges(g, rel, info, sector, region, lifts):
            if mc["name"] in without:
                skip.extend(mc["deck_edges"])
                continue
            bit = 1 << len(mechs)
            for edge, (on, off) in rules.items():
                a, b = guards.get(edge, (0, 0))
                guards[edge] = (a | (bit if on else 0), b | (bit if off else 0))
            mechs.append(mc)
            readers.append(lambda _scene: 0)
    floats = float_rows(g, info, sector) if cork else []
    region, screened = _float_screen(g, rel, region, lifts, floats)
    skip.extend(screened)
    if screened:
        lifts = [e for e in lifts if (*e[:2], "lift") not in screened]
    pendulums = []
    if cork:
        pendulums, restored, screened = pendulum_edges(g, rel, info, sector, region, getattr(world, "mesh", None),
                                                       without, lifts, guards, skip)
        skip.extend(screened)
        if screened:
            lifts = [e for e in lifts if (*e[:2], "lift") not in screened]
        if restored:
            rel = rel.with_edges(restored)
        _pendulum_mechanisms(g, sector, without, pendulums, stacks, mechs, readers, guards, lifts)
    active_floats = [r for r in floats if not without.intersection([r["name"], *r["placements"]])]
    if active_floats:
        _float_departures(g, region, active_floats)
    bridges = [b for b in cork_edges(g, info, sector) if b[3] not in without] if cork else ()
    timed = []
    if cork:
        timed, restored, screened = _timed_edges(g, rel, info, sector, region, getattr(world, "mesh", None), without, guards, skip)
        skip.extend(screened)
        if restored:
            rel = rel.with_edges(restored)
    links = _link_rules(rel, g, region, tk, trafos, R.radius, mechs)
    if cork:
        elevators, points = _elevator_edges(g, info, sector, region, getattr(world, "mesh", None), without, len(mechs))
        if points:
            # A local view, never stored in the cached graph or seen by the rolling solver.
            n = len(points)
            g = coarse.Graph(np.vstack((g.pos, points)), np.r_[g.kind, np.zeros(n, dtype=g.kind.dtype)],
                             np.r_[g.lane, np.full(n, -1)], g.adj + [[] for _ in points], g.lanes, g.info)
            rel = rel.with_edges(())
            rel.g = g
            rel.fwd.extend([] for _ in points)
            rel.back.extend([] for _ in points)
            targets = {v for d in elevators for _, _, v, _, _, _ in d["edges"]}
            region = np.r_[region, [d["node"] in targets for d in elevators]]
            tk = np.r_[tk, np.full(n, -1)]
            mechs.extend(elevators)
            readers.extend((lambda _scene: 0) for _ in elevators)
    sg = coarse.StateGraph(rel, g, region, tk, cost, lifts, mechs, operate_cost, guards=guards,
                           links=links, skip=skip, bridges=bridges,
                           floats=active_floats, pendulums=pendulums, timed=timed)
    sg.float_row_names = {r["name"] for r in floats}
    M = len(mechs)

    def state_of(kind: str, scene) -> int:
        bits = 0
        if scene is not None and len(scene) == len(props):
            for m, rd in enumerate(readers):
                bits |= rd(scene) << m
        return (coarse.KIND_INDEX[kind] << M) | bits

    return sg, state_of


SEESAW_COST = 5.0         # going over a seesaw, on top of the way: ASSUMED, only to rank ways


def seesaw_edges(g, info: dict, parts: list) -> dict:
    """The seesaws (P_Modul_30) on the coarse level. Woken, a seesaw falls to the end its mass centre leans to; a
    ball from that low end goes up it and on: a wood or paper one flies off the high end onto the far side, a stone
    one tips it over and goes down with it (MEASURED on the rebuilt engine). From the high end every ball falls into
    the gap. So the places on the road beyond its low end (7.5 to 12 from the axle, within 2 of its middle) get an
    edge across to the places beyond its high end (8.5 to 16), one way. Once a stone ball has tipped it over it is the
    other way round: the coarse level does not keep that. Per seesaw: dict(sector, edges, low (the way to its low end, level))."""
    out = {}
    for b in parts:
        if not b.get("hinge") or not b["kind"].startswith("P_Modul_30") or b["body"] in out:
            continue
        hg = b["hinge"]
        a, o, com = np.array(hg["axis"], float), np.array(hg["anchor"], float), np.array(hg["mass_centre"], float)
        along = np.cross(a, [0.0, 1.0, 0.0])
        along /= np.linalg.norm(along)
        if (com - o) @ along < 0:
            along = -along                                 # along: towards the low end
        side = np.cross([0.0, 1.0, 0.0], along)
        rel = g.pos - o
        x, y, z = rel @ along, rel[:, 1], rel @ side
        floor = g.kind == 0
        def across(sign):
            """From the road beyond the end at `sign` (1: the end low as it falls first) to beyond the other end."""
            near = np.flatnonzero(floor & (sign * x > 7.5) & (sign * x < 12.0) & (np.abs(z) <= 2.0) & (y > -3.0) & (y < 2.0))
            far = np.flatnonzero(floor & (sign * x < -8.5) & (sign * x > -16.0) & (np.abs(z) <= 2.5) & (y > -3.0) & (y < 6.0))
            out_ = []
            for u in near:
                if not len(far):
                    break
                v = int(far[np.argmin(np.abs(z[far] - z[u]) + 0.1 * np.abs(sign * x[far] + 11.0))])
                out_.append((int(u), v, float(np.linalg.norm(g.pos[v] - g.pos[u])) + SEESAW_COST, "roll"))
            return out_
        # kept apart from the graph's own edges: which are there depends on which end is low (`coarse.StateGraph`)
        out[b["body"]] = dict(sector=b.get("sector"), edges=across(1.0), other=across(-1.0), low=along)
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
                 gate_rule="a closed gate blocks the edges into it, the target plate only past the face it is pushed on; the edges it clears when pushed open %.2f are kept per gate" % GATE_OPEN,
                 board_rule="a push board is crossed where it is pushed: road places in front of it lead across to the far side, cost %.1f more" % BOARD_COST,
                 seesaw_rule="a seesaw is crossed from its low end to beyond its high end, cost %.1f more; kept apart, by state" % SEESAW_COST,
                 state_rule="board, seesaw and gate edges and the transformations' costs are in the state graph, not the graph's own edges",
                 launch_rule="a run-up inside a transformer's range is that kind's",
                 lift_rule="fan lifts for a paper ball, ridden by their legs, with the kind they end in (exits 4 to 32 out, -1.5 to 25 up, always those by a transformer; two fans in a row within %.0f, braking %s): waits %s, %d entries and %d exits per fan, unsure x%.1f" % (LIFT_CHAIN, LIFT_CHAIN_BRAKES, LIFT_WAITS, LIFT_ENTRIES, LIFT_EXITS, LIFT_UNSURE))
    if cache is not None and cache.exists():
        g = pickle.load(open(cache, "rb"))
        n = g.info.get("mesh_triangles")                  # graphs cached before this was recorded have none
        if (n == len(mesh.A) or (n is None and not mesh.meta.get("bodiless_dropped"))) and all(g.info.get(k) == v for k, v in marks.items()):
            return g
        log("the cached coarse graph %s was made by other rules (fans, markers, domes or transformers were obstacles in it): building it again" % cache.name)
    from .launch import add_launches
    g = coarse.build(mesh, avoid=coarse.keep_out(info["placements"]), log=log)
    coarse.add_steps(g, mesh, log=log)
    R = trafo.rules()
    by_kind = dict(wood=ph.WOOD, stone=ph.STONE, paper=ph.PAPER)
    ball_at = {}
    for t in trafo.of(info):
        for k in np.flatnonzero(np.linalg.norm(g.pos - t["pos"], axis=1) < R.radius):
            ball_at[int(k)] = by_kind[t["kind"]]
    add_launches(g, world_for(mesh, info, "strict"), log=log, ball_at=ball_at)
    # (a transformer is gone through at the cost of the transformation: by the ball's kind, in `coarse.StateGraph`)
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
    g.info["boards"] = board_edges(g, info, parts)
    g.info["seesaws"] = seesaw_edges(g, info, parts)
    for name, d in g.info["seesaws"].items():
        log("seesaw %s (sector %s): %d edges across it" % (name, d["sector"], len(d["edges"])))
    g.info["lifts"] = fan_lifts(g, mesh, info, log=log)
    for name, d in g.info["boards"].items():
        log("push board %s (sector %s): %d edges across it" % (name, d["sector"], len(d["edges"])))
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
