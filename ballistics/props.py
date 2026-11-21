"""Loose balls (P_Ball_Wood, P_Ball_Stone, P_Ball_Paper) and boxes (P_Box): free rigid bodies the player ball knocks
and pushes about.

They have no script. Each is a sphere of radius 2 with its own mass, friction, elasticity and damping, read from
the catalogue (`ballistics.things`, the entry P_Ball); the kernel moves them together with the player ball
(`kernel._multi_tick`). The paper one is a convex hull in the game; here it is a sphere of its mass, with the
hull's inertia: good for whether it is knocked away and what speed is left, not for how it rolls and wobbles.

The rules, with their grounds as the catalogue marks them:

* Only the current sector's loose balls are there (READ). The sector's are made where they were put, awake; most
  drop a little and settle, then freeze (READ, MEASURED on the rebuilt engine). Here they start settled and
  frozen: `of` lets each one fall from where it was put until it lies still.
* A frozen one is not a wall: struck, it takes its share of the impact in that very tick and is awake (MEASURED on
  the original engine's records: a wood ball at 4.835 gives a frozen stone one 0.888 in the tick; here 0.863).
* Two balls striking: along the line of centres by their masses, with a restitution of the root of the product of
  the two elasticities, and no lower bound (READ; MEASURED on the rebuilt engine to within 3% in the tick of the
  impact). The friction between them acts in the impact too, which lifts a rolling striker and presses the struck
  ball onto the floor; the floor drags on it from the next tick on (MEASURED: struck balls are down to 0.45 to
  0.56 of the rule's speed 10 to 20 ticks later; here 0.45 to 0.51 for wood and stone ones).
* They trigger nothing: no checkpoint, transformer, pickup or fan (READ).
"""
from __future__ import annotations

import numpy as np

from . import things

KINDS = {"P_Ball_Wood": "wood", "P_Ball_Stone": "stone", "P_Ball_Paper": "paper", "P_Box": "box"}
RADIUS = 2.0
LOCKED = 1e30            # the inertia of a body that does not turn
INERTIA = {"wood": 0.4, "stone": 0.4, "paper": 0.277}      # of mass x radius^2: solid spheres; the paper hull's (catalogue rule rigid_body_mass)
SETTLE_TICKS = 1320                                        # at most this long to come to rest where they were put (10 game s)
# What going through a loose ball costs the coarse level, as way at the follower's pace: ASSUMED, only to rank ways
# (from what a wood ball keeps of its speed: 0.9 against a paper one, 0.4 against a wood one, next to nothing
# against a stone one, which it can only shove along slowly). Whether and how it gets through is the kernel's to say.
COARSE_COST = {"paper": 2.0, "wood": 5.0, "stone": 30.0, "box": 15.0, "P_Modul_34_Kiste": 40.0, "P_Modul_34_Schiebestein": 80.0}

# Mechanisms made of boxes. Each part is a box of its own in the scene, at the place and of the size its shape has in
# the exported mesh (the template's, put where the map puts it); `slider`: it only slides along its own y axis.
#   P_Modul_34, the falling rock: a crate standing on the lower floor carries a block that hangs on an upright rail.
#   Push the crate out from under it and the block comes down (READ: the script's rail; MEASURED on the rebuilt engine:
#   who can push the crate out, see docs/mechanisms.md).
PARTS = {"P_Modul_34": {"P_Modul_34_Kiste": dict(slider=False), "P_Modul_34_Schiebestein": dict(slider=True)}}


def bodies() -> dict:
    """Per kind: the numbers of the loose ball's rigid body, from the catalogue."""
    out = {}
    cat = things.load()
    for b in cat["P_Ball"].bodies:
        kind = next(k for t, k in KINDS.items() if t in b["name"])
        mass = float(b["mass"]["v"])
        out[kind] = dict(kind=kind, shape="sphere", mass=mass, friction=float(b["friction"]["v"]), elasticity=float(b["elasticity"]["v"]),
                         lin_damp=float(b["lin_damp"]["v"]), rot_damp=float(b["rot_damp"]["v"]), inertia=INERTIA[kind] * mass * RADIUS ** 2)
    # the box (P_Box): a cube that slides and does not turn. MEASURED on the rebuilt engine: pushed, a box slides,
    # it does not tip (at most 5.6 degrees in 48 runs). So here its orientation is fixed: it cannot tip over an
    # edge either, it goes down once its middle is past the edge
    b = cat["P_Box"].bodies[0]
    half = float(cat["P_Box"].numbers["box_size"]["v"]) / 2.0
    out["box"] = dict(kind="box", shape="box", mass=float(b["mass"]["v"]), friction=float(b["friction"]["v"]), elasticity=float(b["elasticity"]["v"]),
                      lin_damp=float(b["lin_damp"]["v"]), rot_damp=float(b["rot_damp"]["v"]), inertia=LOCKED, half=(half, half, half))
    return out


def part_boxes(info: dict, shapes, sector=None) -> list:
    """The box-shaped parts of the mechanisms in PARTS, read from a mesh that still has the placed shapes:
    [dict(kind (the part's name), name, sector, placed (centre), half, axes, slider)], the sector's or all."""
    out = []
    if shapes is None:
        return out
    for p in info.get("placements", ()):
        parts = PARTS.get(p["type"])
        if not parts or (sector is not None and p.get("sector") != sector):
            continue
        axes = np.array([np.array(p[k], float) / np.linalg.norm(p[k]) for k in ("x_axis", "y_axis", "z_axis")])
        for i, e in enumerate(shapes.entities):
            if e.get("placement") != p["name"] or e.get("name") not in parts:
                continue
            sel = shapes.owner == i
            if not sel.any():
                continue
            V = np.vstack([shapes.A[sel], shapes.B[sel], shapes.C[sel]]) @ axes.T        # in the mechanism's own axes
            lo, hi = V.min(0), V.max(0)
            out.append(dict(kind=e["name"], name="%s/%s" % (p["name"], e["name"]), sector=p.get("sector"), placed=[float(v) for v in ((lo + hi) / 2) @ axes],
                            half=tuple(float(v) for v in (hi - lo) / 2), axes=axes.tolist(), slider=bool(parts[e["name"]]["slider"])))
    return out


def of(info: dict, sector, mesh=None, shapes=None) -> list:
    """The loose bodies that are there while `sector` is the current one: the loose balls, the boxes and the box-shaped
    parts of mechanisms: [dict(name, kind, mass, ..., placed, rest)]. `rest` is where the body lies once it has
    settled: with a mesh (the fixed geometry, without the bodies' own shapes) each is let fall from where it was
    put; without one it is where it was put. shapes: a mesh that still has the placed shapes, for the mechanisms'
    parts. sector None: none."""
    if sector is None:
        return []
    table = bodies()
    out = []
    cat = things.load()
    for b in part_boxes(info, shapes, sector):
        c = next(x for x in cat[b["kind"][:10]].bodies if x["name"] == b["kind"])           # "P_Modul_34"
        out.append(dict(b, shape="box", mass=float(c["mass"]["v"]), friction=float(c["friction"]["v"]), elasticity=float(c["elasticity"]["v"]),
                        lin_damp=float(c["lin_damp"]["v"]), rot_damp=float(c["rot_damp"]["v"]), inertia=LOCKED, rest=list(b["placed"])))
    for p in info.get("placements", ()):
        kind = KINDS.get(p["type"])
        if kind and p.get("sector") == sector:
            body = dict(table[kind], name=p["name"], placed=[float(v) for v in p["position"]], rest=[float(v) for v in p["position"]])
            if body["shape"] == "box":                       # the box's own axes, as it was put
                unit = dict(x_axis=(1.0, 0.0, 0.0), y_axis=(0.0, 1.0, 0.0), z_axis=(0.0, 0.0, 1.0))
                body["axes"] = [[float(v) for v in np.array(p.get(k, unit[k]), float) / np.linalg.norm(p.get(k, unit[k]))] for k in unit]
            out.append(body)
    if mesh is not None and out:
        from .kernel import World
        w = World.of(mesh, stop_on_placed=False, props=out)
        scene = w.scene()
        scene[:, 9] = 1.0                                  # awake where they were put
        w.idle(scene, SETTLE_TICKS)
        for k, b in enumerate(out):
            b["rest"] = [float(v) for v in scene[k, :3]]
            b["settled"] = bool(scene[k, 9] == 0.0)        # False: still moving after SETTLE_TICKS (a slope), or fell out
            b["gone"] = bool(scene[k, 9] == 2.0)
    return out
