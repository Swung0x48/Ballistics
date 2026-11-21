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

# Mechanisms made of bodies that slide and do not turn. Each part is a body of its own in the scene.
#   `slider`: it only slides along its own y axis.
#   `hulls`: its shape is its collision hulls (the map's `hulls`, `ballistics.export.placed_hulls`): a piece per
#     hull, all of one body; without it, the box its shape has in the exported mesh (the template's, put where the
#     map puts it).
#   `role`: 0 an ordinary body; 1 one that does not touch the level's floors and rails (the game's group Floor):
#     it touches balls, boxes and its mechanism's fixed parts; 2 a fixed part that touches only its mechanism's
#     parts of role 1 (the ball goes through it).
#   P_Modul_34, the falling rock: a crate standing on the lower floor carries a block that hangs on an upright rail.
#   Push the crate out from under it and the block comes down (READ: the script's rail; MEASURED on the rebuilt engine:
#   who can push the crate out, see docs/mechanisms.md).
#   P_Modul_01, the sliding gate: the gate (target plate, bar and fence, one body of three hulls) lies in a groove
#   (a floor plate and two walls) that only it touches; pushed along the groove it slides out of the upper road. The
#   filler plate under it is road (`coarse.FIXED_PARTS`). READ: the groups, the hulls; the rest is contacts.
PARTS = {"P_Modul_34": {"P_Modul_34_Kiste": dict(slider=False), "P_Modul_34_Schiebestein": dict(slider=True)},
         "P_Modul_01": {"P_Modul_01_Pusher": dict(hulls=True, role=1), "P_Modul_01_Rinne": dict(hulls=True, role=2)}}
MAX_TRIS = 12            # a hull piece that is not a box: at most this many triangles (a hexahedron)


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


def _piece(kind, name, body, p, axes, V, spec, mech, faces=None):
    """A part's piece: the box around the vertices V (world) in the mechanism's axes, and, when V is a hull that is
    not that box, its triangles about the box's centre."""
    L = V @ axes.T                                   # in the mechanism's own axes
    lo, hi = L.min(0), L.max(0)
    centre = ((lo + hi) / 2) @ axes
    tris = None
    if faces is not None:
        corner = np.all(np.minimum(np.abs(L - lo), np.abs(L - hi)) < 1e-4, axis=1)
        if not corner.all():                         # not a box: keep its faces (a trapezoid prism: the gate's fence)
            F = np.asarray(faces, int)
            if len(F) > MAX_TRIS:
                raise ValueError("%s: a hull of %d triangles (at most %d)" % (name, len(F), MAX_TRIS))
            tris = [[float(v) for v in np.concatenate([V[f[0]] - centre, V[f[1]] - centre, V[f[2]] - centre])] for f in F]
    return dict(kind=kind, name=name, body=body, sector=p.get("sector"), placed=[float(v) for v in centre], half=tuple(float(v) for v in (hi - lo) / 2),
                axes=axes.tolist(), slider=bool(spec.get("slider", False)), role=int(spec.get("role", 0)), mech=mech, tris=tris)


def part_boxes(info: dict, shapes, sector=None) -> list:
    """The parts of the mechanisms in PARTS, as pieces that slide and do not turn: [dict(kind (the part's name),
    name, body (the pieces of one body share it), sector, placed (centre), half, axes, slider, role, mech (which
    mechanism, from 1), tris)], the sector's or all. A part without hulls is read from a mesh that still has the
    placed shapes; one with hulls from the map's hulls."""
    out = []
    hulls = info.get("hulls", ())
    for m, p in enumerate(info.get("placements", ())):
        parts = PARTS.get(p["type"])
        if not parts or (sector is not None and p.get("sector") != sector):
            continue
        axes = np.array([np.array(p[k], float) / np.linalg.norm(p[k]) for k in ("x_axis", "y_axis", "z_axis")])
        for part, spec in parts.items():
            body = "%s/%s" % (p["name"], part)
            if spec.get("hulls"):
                for h in hulls:
                    if h["placement"] == p["name"] and h["body"] == part:
                        out.append(_piece(part, "%s/%s" % (body, h["mesh"]), body, p, axes, np.array(h["vertices"], float), spec, m + 1, h["faces"]))
                continue
            if shapes is None:
                continue
            for i, e in enumerate(shapes.entities):
                if e.get("placement") != p["name"] or e.get("name") != part:
                    continue
                sel = shapes.owner == i
                if sel.any():
                    out.append(_piece(part, body, body, p, axes, np.vstack([shapes.A[sel], shapes.B[sel], shapes.C[sel]]), spec, m + 1))
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
    first: dict = {}
    for b in part_boxes(info, shapes, sector):
        c = next(x for x in cat[b["kind"][:10]].bodies if x["name"] == b["kind"])           # "P_Modul_34"
        fixed = b["role"] == 2
        # every piece of a body carries the body's mass: they move as one (`kernel._multi_tick`)
        out.append(dict(b, shape="box", mass=LOCKED if fixed else float(c["mass"]["v"]), friction=float(c["friction"]["v"]), elasticity=float(c["elasticity"]["v"]),
                        lin_damp=float(c["lin_damp"]["v"]), rot_damp=float(c["rot_damp"]["v"]), inertia=LOCKED, rest=list(b["placed"]),
                        leader=-1 if fixed else first.setdefault(b["body"], len(out))))
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
        scene[:, 9] = [0.0 if b.get("role") == 2 else 1.0 for b in out]       # awake where they were put (a fixed part never is)
        w.idle(scene, SETTLE_TICKS)
        for k, b in enumerate(out):
            b["rest"] = [float(v) for v in scene[k, :3]]
            b["settled"] = bool(scene[k, 9] == 0.0)        # False: still moving after SETTLE_TICKS (a slope), or fell out
            b["gone"] = bool(scene[k, 9] == 2.0)
    return out
