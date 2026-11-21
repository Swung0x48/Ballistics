"""Loose balls (P_Ball_Wood, P_Ball_Stone, P_Ball_Paper): free rigid spheres the player ball knocks about.

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

KINDS = {"P_Ball_Wood": "wood", "P_Ball_Stone": "stone", "P_Ball_Paper": "paper"}
RADIUS = 2.0
INERTIA = {"wood": 0.4, "stone": 0.4, "paper": 0.277}      # of mass x radius^2: solid spheres; the paper hull's (catalogue rule rigid_body_mass)
SETTLE_TICKS = 1320                                        # at most this long to come to rest where they were put (10 game s)
# What going through a loose ball costs the coarse level, as way at the follower's pace: ASSUMED, only to rank ways
# (from what a wood ball keeps of its speed: 0.9 against a paper one, 0.4 against a wood one, next to nothing
# against a stone one, which it can only shove along slowly). Whether and how it gets through is the kernel's to say.
COARSE_COST = {"paper": 2.0, "wood": 5.0, "stone": 30.0}


def bodies() -> dict:
    """Per kind: the numbers of the loose ball's rigid body, from the catalogue."""
    out = {}
    for b in things.load()["P_Ball"].bodies:
        kind = next(k for t, k in KINDS.items() if t in b["name"])
        mass = float(b["mass"]["v"])
        out[kind] = dict(kind=kind, mass=mass, friction=float(b["friction"]["v"]), elasticity=float(b["elasticity"]["v"]),
                         lin_damp=float(b["lin_damp"]["v"]), rot_damp=float(b["rot_damp"]["v"]), inertia=INERTIA[kind] * mass * RADIUS ** 2)
    return out


def of(info: dict, sector, mesh=None) -> list:
    """The loose balls that are there while `sector` is the current one: [dict(name, kind, mass, ..., placed, rest)].
    `rest` is where the ball lies once it has settled: with a mesh (the fixed geometry, without the loose balls'
    own shapes) each is let fall from where it was put; without one it is where it was put. sector None: none."""
    if sector is None:
        return []
    table = bodies()
    out = []
    for p in info.get("placements", ()):
        kind = KINDS.get(p["type"])
        if kind and p.get("sector") == sector:
            out.append(dict(table[kind], name=p["name"], placed=[float(v) for v in p["position"]], rest=[float(v) for v in p["position"]]))
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
