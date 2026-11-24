"""Loose balls (P_Ball_Wood, P_Ball_Stone, P_Ball_Paper) and boxes (P_Box): free rigid bodies the player ball knocks
and pushes about.

They have no script. Each is a sphere of radius 2 with its own mass, friction, elasticity and damping, read from
the catalogue (`ballistics.things`, the entry P_Ball); the kernel moves them together with the player ball
(`kernel._multi_tick`). The paper one is a convex hull in the game; here it is a sphere of its mass, with the
hull's inertia: good for whether it is knocked away and what speed is left, not for how it rolls and wobbles.

The rules, with their grounds as the catalogue marks them:

* Only the current sector's loose balls are there (READ). The sector's are made where they were put, awake; most
  drop a little and settle, then freeze (READ, MEASURED on the rebuilt engine). Here they start settled and
  frozen when supported: `of` finds their first support by a vertical geometric sweep.
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
BALL_REST_SLOPE = np.tan(np.radians(0.1))   # ASSUMED flatness tolerance, not a friction threshold
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
#   `hinge`: it turns about a fixed hinge (the catalogue's motion: axis and point in the template's frame), with
#     the mass centre the script gives it (in its own frame) and the measured inertia about that.
#   P_Modul_25, the push board: a board on a hinge at the end of a road. Woken, it leans back onto the road (its
#     wedge); pushed over its top from the road it falls forward across the gap onto a stopper and is a bridge.
#   P_Modul_30, the seesaw: a board on a hinge with its mass centre above the axle, in the group Floor (it goes
#     through the floors; the stoppers under its two ends hold it). Woken, it falls to one end; a ball that gets past
#     the axle may tip it over to the other.
#   P_Modul_01, the sliding gate: the gate (target plate, bar and fence, one body of three hulls) lies in a groove
#   (a floor plate and two walls) that only it touches; pushed along the groove it slides out of the upper road. The
#   filler plate under it is road (`coarse.FIXED_PARTS`). READ: the groups, the hulls; the rest is contacts.
PARTS = {"P_Modul_34": {"P_Modul_34_Kiste": dict(slider=False), "P_Modul_34_Schiebestein": dict(slider=True)},
         "P_Modul_01": {"P_Modul_01_Pusher": dict(hulls=True, role=1), "P_Modul_01_Rinne": dict(hulls=True, role=2)},
         "P_Modul_25": {"P_Modul_25_Bridge": dict(hulls=True, role=0, hinge=True)},
         "P_Modul_30": {"P_Modul_30_Wippe": dict(hulls=True, role=1, hinge=True)}}
STOPPER = "Phys_FloorStopper"       # the catalogue's body for the stoppers (in collision_rules)


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


def _piece(kind, name, body, p, axes, V, spec, mech, hull=False):
    """A part's piece: the box around the vertices V (world) in the mechanism's axes, and, when V is a hull, the hull
    of those vertices (the engine's shape, whatever faces the mesh has): its triangles about the box's centre where
    it is not that box, and its corners where the piece turns or is a stopper."""
    from .geometry import convex_hull
    L = V @ axes.T                                   # in the mechanism's own axes
    lo, hi = L.min(0), L.max(0)
    centre = ((lo + hi) / 2) @ axes
    tris = verts = None
    if hull:
        H, F = convex_hull(V)
        corner = np.all(np.minimum(np.abs(H @ axes.T - lo), np.abs(H @ axes.T - hi)) < 1e-4, axis=1)
        if not corner.all():                         # not a box: keep its faces (a trapezoid prism: the gate's fence)
            tris = [[float(v) for v in np.concatenate([H[f[0]] - centre, H[f[1]] - centre, H[f[2]] - centre])] for f in F]
        if spec.get("hinge") or spec.get("role") == 4:
            verts = [[float(v) for v in h - centre] for h in H]
            if spec.get("hinge") and len(H) <= 8:
                # a hinged piece touches by its corners; where its edge lies on the edge of what holds it (a seesaw's end
                # as wide as its stopper) a corner falls on the side, so each corner also has a point a little inside it
                Hc = H.mean(0)
                verts += [[float(v) for v in (h + 0.03 * (Hc - h)) - centre] for h in H]
    return dict(kind=kind, name=name, body=body, sector=p.get("sector"), placed=[float(v) for v in centre], half=tuple(float(v) for v in (hi - lo) / 2),
                axes=axes.tolist(), slider=bool(spec.get("slider", False)), role=int(spec.get("role", 0)), mech=mech, tris=tris, verts=verts)


def _hinge(cat_body, p, hulls, mass):
    """A hinged body's hinge in world, from the catalogue (axis and point in the template's frame, the script's mass
    centre in the body's own frame, the inertia about that along the body's axes) and the body's frame in the map."""
    PA = np.array([np.array(p[k], float) / np.linalg.norm(p[k]) for k in ("x_axis", "y_axis", "z_axis")])
    h = cat_body["hinge"]["v"]                         # (the catalogue's motion.anchor is in the template's frame for one entry, the body's for another)
    if h.get("frame") != "template":
        raise ValueError("%s: the hinge is given in the frame %r, not the template's" % (cat_body["name"], h.get("frame")))
    axis = np.array(h["axis"], float) @ PA
    axis /= np.linalg.norm(axis)
    anchor = np.array(h["anchor"], float) @ PA + np.array(p["position"], float)
    F = np.array(hulls[0]["frame"], float)
    R, T = F[:3, :3], F[3, :3]
    com = np.array(cat_body["mass_center"]["v"], float) @ R + T
    core = np.array(cat_body["inertia"]["v"], float)
    r = com - anchor
    r_perp = r - axis * (r @ axis)
    inertia = float(sum(core[i] * ((R[i] / np.linalg.norm(R[i])) @ axis) ** 2 for i in range(3))) + mass * float(r_perp @ r_perp)
    return dict(anchor=[float(v) for v in anchor], axis=[float(v) for v in axis], inertia=inertia, mass_centre=[float(v) for v in com])


def stoppers(info: dict, sector=None) -> list:
    """The stoppers (the map's `stoppers`): fixed blocks that stop mechanisms and props, not the player ball (READ).
    Each is a piece with its hull's corners and triangles."""
    out = []
    cat = things.load()
    c = next(b for b in cat["collision_rules"].bodies if b["name"].startswith(STOPPER))
    for i, s in enumerate(info.get("stoppers", ())):
        if sector is not None and s.get("sector") not in (None, sector):
            continue
        V = np.array(s["vertices"], float)
        if len(np.unique(np.round(V, 6), axis=0)) > 16:
            continue                                     # (a big stopper block, of level 13: not needed yet)
        b = _piece(STOPPER, s["name"], s["name"], dict(sector=s.get("sector")), np.eye(3), V, dict(role=4), 0, hull=True)
        out.append(dict(b, shape="box", mass=LOCKED, friction=float(c["friction"]["v"]), elasticity=float(c["elasticity"]["v"]),
                        lin_damp=0.0, rot_damp=0.0, inertia=LOCKED, rest=list(b["placed"]), leader=-1))
    return out


def part_boxes(info: dict, shapes, sector=None, *, include_t_board=False) -> list:
    """The placed pieces of the mechanisms in PARTS (optionally including the T-board): [dict(kind (the part's name),
    name, body (the pieces of one body share it), sector, placed (centre), half, axes, slider, role, mech (which
    mechanism, from 1), tris)], the sector's or all. A part without hulls is read from a mesh that still has the
    placed shapes; one with hulls from the map's hulls."""
    out = []
    hulls = info.get("hulls", ())
    for m, p in enumerate(info.get("placements", ())):
        parts = PARTS.get(p["type"])
        if include_t_board and p["type"] == "P_Modul_37":
            # Region-only state change; in --roll this was fixed geometry and still is.
            parts = {"P_Modul_37_Bridge": dict(hulls=True, role=3)}
        if not parts or (sector is not None and p.get("sector") != sector):
            continue
        axes = np.array([np.array(p[k], float) / np.linalg.norm(p[k]) for k in ("x_axis", "y_axis", "z_axis")])
        for part, spec in parts.items():
            body = "%s/%s" % (p["name"], part)
            if spec.get("hulls"):
                hs = [h for h in hulls if h["placement"] == p["name"] and h["body"] == part
                      and (spec.get("role") != 3 or h.get("frame") is not None)]
                pieces = [_piece(part, "%s/%s" % (body, h["mesh"]), body, p, axes, np.array(h["vertices"], float), spec, m + 1, hull=True) for h in hs]
                if spec.get("hinge") and hs:
                    cb = next(x for x in things.load()[p["type"]].bodies if x["name"] == part)
                    hg = _hinge(cb, p, hs, float(cb["mass"]["v"]))
                    for b in pieces:
                        b["hinge"] = hg
                    # the piece furthest from the hinge first: the body's angle is read from where it is
                    a, o = np.array(hg["axis"]), np.array(hg["anchor"])
                    far = lambda b: float(np.linalg.norm((np.array(b["placed"]) - o) - a * ((np.array(b["placed"]) - o) @ a)))
                    pieces.sort(key=lambda b: -far(b))
                out += pieces
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



def _rotation(axis, angle):
    """Rodrigues rotation, acting on column vectors."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return c * np.eye(3) + (1. - c) * np.outer(axis, axis) + s * np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _opening_pose(b, entry, frame):
    """Centre, turn, and catalogue state for one piece (axes/tris stay in the placed frame for a live hinge)."""
    kind = entry.id
    placed = np.asarray(b["placed"])
    if kind == "P_Modul_01":           # at_rest: the slider settles onto the groove, without opening
        if b["kind"] == "P_Modul_01_Pusher":
            placed = placed - np.asarray(b["axes"][1]) * entry.numbers["slider_settle"]["v"]
        return placed, np.eye(3), "at_rest"
    if kind == "P_Modul_34":           # standing: both settle at wake, without the block dropping
        if b["kind"] == "P_Modul_34_Kiste":
            placed = placed - (0., entry.numbers["crate_settle"]["v"], 0.)
        else:
            placed = placed - np.asarray(b["axes"][1]) * entry.numbers["block_settle"]["v"]
        return placed, np.eye(3), "standing"
    if kind == "P_Modul_37":           # standing: foot leans toward the road (+local X)
        axis = np.asarray(b["axes"])[2]
        pivot = np.asarray(frame)[3, :3]
        angle = -np.radians(entry.numbers["standing_lean_deg"]["v"])
        state = "standing"
    else:
        axis, pivot = (np.asarray(b["hinge"][k]) for k in ("axis", "anchor"))
        if kind == "P_Modul_25":       # leaning: wedge on the approach road
            angle = np.radians(entry.numbers["lean_rest_deg"]["v"])
            state = "leaning"
        else:                          # low_home: the local -X end is low
            bx = np.asarray(frame)[0, :3]    # the board's (not the placement's) already tilted X axis
            horizontal = np.cross((0., 1., 0.), axis)
            if horizontal @ b["axes"][0] < 0.:
                horizontal = -horizontal
            old = np.arctan2(bx @ np.cross(axis, horizontal), bx @ horizontal)
            target = np.radians(entry.numbers["rest_angle"]["v"])
            target *= np.sign(np.cross(axis, horizontal)[1])
            angle = target - old
            state = "low_home"
    turn = _rotation(axis, angle)
    return pivot + turn @ (placed - pivot), turn, state


# Twelve outward triangles of an oriented box, indexed by the binary signs of the corners.
_BOX_FACES = ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
              (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3))
_BOX_SIGNS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def _support_tris(b, turn):
    """World-space surfaces at the opening pose (including a hinged piece's rotation)."""
    if b.get("tris"):
        local = np.asarray(b["tris"]).reshape(-1, 3, 3)
    else:
        corners = (_BOX_SIGNS * b["half"]) @ b["axes"]
        local = corners[np.asarray(_BOX_FACES)]
    return local @ turn.T + b["rest"]


def _drop_loose(mesh, loose, supports, floor_friction):
    from .geometry import vertical_candidates, vertical_support

    # Lower boxes before upper ones (by the placed bottom), then balls. A sliding/falling body is not a support.
    def bottom(b):
        return b["placed"][1] - np.asarray(b["half"]) @ np.abs(np.asarray(b["axes"])[:, 1])

    boxes = sorted((b for b in loose if b["shape"] == "box"), key=bottom)
    balls = [b for b in loose if b["shape"] == "sphere"]
    for b in (*boxes, *balls):
        centre = np.array(b["placed"])
        if b["shape"] == "box":
            half, axes = np.asarray(b["half"]), np.asarray(b["axes"])
            shape = dict(axes=axes, half=half)
            reach = half @ np.abs(axes[:, [0, 2]])
        else:
            shape, reach = dict(radius=RADIUS), RADIUS
        ix = vertical_candidates(mesh, centre, reach)
        tris = [np.stack((mesh.A[ix], mesh.B[ix], mesh.C[ix]), axis=1)]
        fr = [[float(mesh.entities[mesh.owner[i]].get("friction", floor_friction)) if mesh.entities else floor_friction for i in ix]]
        for t, mu in supports:
            tris.append(t)
            fr.append(np.full(len(t), mu))
        hit = vertical_support(np.concatenate(tris), np.concatenate(fr), centre, **shape)
        b["gone"] = hit is None
        b["settled"] = False
        if hit is not None:
            b["rest"][1], ny, mu = hit
            # Coulomb friction can balance downslope force on a box. For a sphere it supplies rolling torque,
            # not static equilibrium on a slope; only the flatness tolerance above is allowed.
            limit = b["friction"] * mu if b["shape"] == "box" else BALL_REST_SLOPE
            b["settled"] = bool(np.sqrt(max(0., 1. - ny**2)) <= limit * ny + 1e-12)
            if b["settled"] and b["shape"] == "box":
                supports.append((_support_tris(b, np.eye(3)), b["friction"]))


def of(info: dict, sector, mesh=None, shapes=None) -> list:
    """Opening poses of the sector's mechanisms and loose props; no time integration.

    `mesh` is the fixed geometry, `shapes` still includes placed entities. Without a mesh only the mechanisms
    have a catalogue opening pose; the loose balls and boxes retain their placement.
    """
    if sector is None:
        return []
    table, cat = bodies(), things.load()
    out, loose, supports = [], [], []
    first = {}
    frames = {(h["placement"], h["body"]): h["frame"] for h in info.get("hulls", ()) if h.get("frame") is not None}
    for b in part_boxes(info, shapes, sector, include_t_board=True):
        entry = cat[b["kind"][:10]]
        c = next(x for x in entry.bodies if x["name"] == b["kind"])
        fixed = b["role"] in (2, 3)
        rest, turn, state = _opening_pose(b, entry, frames.get(tuple(b["body"].rsplit("/", 1))))
        p = dict(b, shape="box", mass=LOCKED if fixed else float(c["mass"]["v"]), friction=float(c["friction"]["v"]),
                 elasticity=float(c["elasticity"]["v"]), lin_damp=float(c["lin_damp"]["v"]), rot_damp=float(c["rot_damp"]["v"]),
                 inertia=LOCKED, rest=rest.tolist(), settled=True, gone=False, opening_state=state,
                 leader=-1 if fixed else first.setdefault(b["body"], len(out)))
        if b["role"] != 2:                       # the gate groove only collides with the gate
            supports.append((_support_tris(p, turn), p["friction"]))
        if b["role"] == 3:                       # T-board only changes pose in the region state graph
            p["axes"] = (np.asarray(p["axes"]) @ turn.T).tolist()
            if p.get("tris"):
                p["tris"] = (np.asarray(p["tris"]).reshape(-1, 3, 3) @ turn.T).reshape(-1, 9).tolist()
        out.append(p)
    for p in info.get("placements", ()):
        kind = KINDS.get(p["type"])
        if kind and p.get("sector") == sector:
            body = dict(table[kind], name=p["name"], placed=[float(v) for v in p["position"]], rest=[float(v) for v in p["position"]])
            if body["shape"] == "box":
                axes = np.array([p.get(k + "_axis", e) for k, e in zip("xyz", np.eye(3))], float)
                body["axes"] = (axes / np.linalg.norm(axes, axis=1)[:, None]).tolist()
            loose.append(body)
    out += loose
    if out:
        for stopper in stoppers(info, sector):
            out.append(dict(stopper, settled=True, gone=False))
            supports.append((_support_tris(stopper, np.eye(3)), stopper["friction"]))
    if mesh is not None and loose:
        # The catalogue gives standard floor x box friction 0.49 = 0.7 x 0.7; use the actual surface's material
        # where exported (including on other supports), with the catalogue floor as the missing-material fallback.
        floor_fr = cat["P_Box"].numbers["floor_friction"]["v"] / table["box"]["friction"]
        _drop_loose(mesh, loose, supports, floor_fr)
    return out
