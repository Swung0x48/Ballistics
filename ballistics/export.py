"""From a map file to the files the analyzer works on.

Input: a Ballance map (.nmo). Output, per map, in the output folder:

* `<name>-mesh.npz`: the static collision triangles, and the tube axes fitted to the rails (see
  `ballistics.geometry.Mesh`, `ballistics.rails`).
* `<name>-map.json`: spawn, reset points, checkpoints in order, airship, sectors and what is placed in them, and
  the death boxes.

What collides with the ball follows the game's own rule: the objects in the groups Phys_Floors and
Phys_FloorRails become fixed bodies with their mesh as the collision shape (objects in Phys_FloorStopper only stop
mechanisms and props, not the player ball, and are left out). The game also keeps one body of its own in every
level, a fixed 4-unit cube at the origin; it is added so that the result is the complete set of fixed bodies.

Checked on level 1: the triangles produced here from Level_01.NMO are the same set as the triangles the running
game holds for its fixed bodies (21,775, including the cube)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .geometry import Mesh
from .nmo import NmoLevel, read_nmo
from .rails import fit_tubes

FLOOR_GROUP, RAIL_GROUP, STOPPER_GROUP = "Phys_Floors", "Phys_FloorRails", "Phys_FloorStopper"
# friction and elasticity the game gives these bodies when it builds a level (read from the running game on level
# 1; they come from the game's level-building script, not from the map file)
SURFACE = {FLOOR_GROUP: (0.7, 0.3), RAIL_GROUP: (0.7, 0.3)}
FIXTURE = dict(name="FixCube", half=2.0, friction=0.7, elasticity=0.4)   # the game's own fixed cube at the origin
START_GROUP, END_GROUP, CHECKPOINT_GROUP, RESET_GROUP, DEATH_GROUP = "PS_Levelstart", "PE_Levelende", "PC_Checkpoints", "PR_Resetpoints", "DepthTestCubes"
SECTOR = re.compile(r"^Sector_(\d+)$")


def _cube(h: float):
    """The 12 triangles of an axis-aligned cube of half-size h at the origin."""
    c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    A, B, C = [], [], []
    for a, b, d, e in quads:
        A += [c[a], c[a]]; B += [c[b], c[d]]; C += [c[d], c[e]]
    return np.array(A), np.array(B), np.array(C)


# Placed types whose parts are not put into the mesh. Pickups do not hinder; a transformer matters by its trigger
# distance (see the map file), not its body; a fan (P_Modul_18) has no rigid body at all, neither its rotor nor its
# box (READ in its scripts; in the original engine's records the paper ball flies inside the box for thousands of
# ticks): it acts by its push (`ballistics.fans`).
SKIP_TEMPLATES = ("P_Extra_", "P_Trafo_", "P_Modul_18")
SKIP_PARTS = ("Shadow",)


def placed_parts(level: NmoLevel, templates, bmap=None) -> list:
    """The objects placed on the map, each part at its rest pose: [(placement name, type, sector, part name, world
    vertices, faces)]. A map only holds a marker for each placed object; its parts come from the game's template
    file of that type (`<templates>/<type>.nmo`) and sit where the marker's world matrix puts them."""
    templates = Path(templates)
    sector_of = {}
    for g, names in level.groups.items():
        m = SECTOR.match(g)
        if m:
            for n in names:
                sector_of[n] = int(m.group(1))
    cache: dict = {}
    out = []
    for o in level.objects.values():
        kinds = [g for g in o.groups if g.startswith("P_")]
        if not kinds or kinds[0].startswith(SKIP_TEMPLATES):
            continue
        kind = kinds[0]
        if kind not in cache:
            f = templates / (kind + ".nmo")
            cache[kind] = read_nmo(f, bmap) if f.is_file() else None
        if cache[kind] is None:
            continue
        m = np.asarray(o.matrix, np.float64)
        for part in cache[kind].objects.values():
            if part.vertices is None or part.name.endswith(SKIP_PARTS):
                continue
            V = part.world_vertices() @ m[:3, :3] + m[3, :3]
            out.append((o.name, kind, sector_of.get(o.name), part.name, V, part.faces))
    return out


def placed_hulls(level: NmoLevel, templates, bmap=None) -> list:
    """The collision hulls of the placed objects' bodies, where the catalogue names them (`hull.kind` convex_hulls):
    [dict(placement, type, sector, body, mesh, vertices (world), faces)]. A body's hull is not the mesh it shows: the
    template's script gives the body a few convex meshes that no object shows (P_Modul_01's gate: three, its groove:
    three). Each is in its body's frame (DERIVED from the geometry: the groove's floor plate is the groove's own
    shown mesh, and the gate's bar sits on the plate as the shown gate does), put where the map puts the object."""
    from . import things
    templates = Path(templates)
    cat = things.load()
    sector_of = {}
    for g, names in level.groups.items():
        m = SECTOR.match(g)
        if m:
            for n in names:
                sector_of[n] = int(m.group(1))
    cache: dict = {}
    out = []
    for o in level.objects.values():
        kinds = [g for g in o.groups if g.startswith("P_")]
        if not kinds or kinds[0].startswith(SKIP_TEMPLATES):
            continue
        kind = kinds[0]
        th = cat.of_type(kind)
        if th is None:
            continue
        wanted = {b["name"]: list((b.get("hull") or {}).get("meshes") or ()) for b in th.bodies if (b.get("hull") or {}).get("kind") == "convex_hulls"}
        if not any(wanted.values()):
            continue
        if kind not in cache:
            f = templates / (kind + ".nmo")
            cache[kind] = read_nmo(f, bmap) if f.is_file() else None
        t = cache[kind]
        if t is None:
            continue
        m = np.asarray(o.matrix, np.float64)
        for body, names in wanted.items():
            part = t.objects.get(body)
            if part is None:
                continue
            pm = np.asarray(part.matrix, np.float64)
            for name in names:
                if name not in t.meshes:
                    continue
                V, F = t.meshes[name]
                W = (V @ pm[:3, :3] + pm[3, :3]) @ m[:3, :3] + m[3, :3]
                out.append(dict(placement=o.name, type=kind, sector=sector_of.get(o.name), body=body, mesh=name,
                                vertices=np.round(W, 6).tolist(), faces=F.tolist()))
    return out


def build_mesh(level: NmoLevel, fixture: bool = True, templates=None, bmap=None) -> Mesh:
    """The fixed bodies the player ball collides with, as triangles in world coordinates. With `templates` (the
    game's folder of placed-object templates), the placed objects' parts are added at their rest poses and marked
    `placed`: they are what stands on the map before anything is touched, not fixed geometry."""
    A, B, C, owner, rail, ents = [], [], [], [], [], []
    seen = set()
    for group in (FLOOR_GROUP, RAIL_GROUP):
        for o in level.members(group):
            if o.name in seen or o.vertices is None:
                continue
            if any(g.startswith("P_") for g in o.groups):
                continue          # a placed object's marker: the game puts the template there, the marker's own shape is no body
            seen.add(o.name)
            V, F = o.world_vertices(), o.faces
            idx = len(ents)
            ents.append(dict(name=o.name, groups=sorted(o.groups), friction=SURFACE[group][0], elasticity=SURFACE[group][1]))
            A.append(V[F[:, 0]]); B.append(V[F[:, 1]]); C.append(V[F[:, 2]])
            owner.append(np.full(len(F), idx)); rail.append(np.full(len(F), group == RAIL_GROUP))
    if fixture:
        a, b, c = _cube(FIXTURE["half"])
        idx = len(ents)
        ents.append(dict(name=FIXTURE["name"], groups=[], friction=FIXTURE["friction"], elasticity=FIXTURE["elasticity"], fixture=True))
        A.append(a); B.append(b); C.append(c)
        owner.append(np.full(len(a), idx)); rail.append(np.zeros(len(a), bool))
    n_placed = 0
    if templates is not None:
        for place, kind, sector, part, V, F in placed_parts(level, templates, bmap):
            idx = len(ents)
            ents.append(dict(name=part, groups=[], placed=True, type=kind, placement=place, sector=sector))
            A.append(V[F[:, 0]]); B.append(V[F[:, 1]]); C.append(V[F[:, 2]])
            owner.append(np.full(len(F), idx)); rail.append(np.zeros(len(F), bool))
            n_placed += 1
    if not A:
        z = np.zeros((0, 3))
        return Mesh(z, z, z, np.zeros(0, np.int64), np.zeros(0, bool), np.zeros((0, 7)), ents, {})
    A, B, C = np.concatenate(A), np.concatenate(B), np.concatenate(C)
    owner, rail = np.concatenate(owner), np.concatenate(rail)
    tubes, fit = fit_tubes(A[rail], B[rail], C[rail]) if rail.any() else (np.zeros((0, 7)), {})
    meta = dict(source="map file: objects in %s and %s, plus the game's fixed cube at the origin" % (FLOOR_GROUP, RAIL_GROUP),
                map=Path(level.path).name, triangles=int(len(A)), rail_triangles=int(rail.sum()),
                stoppers_left_out=len(level.groups.get(STOPPER_GROUP, [])), tubes=int(len(tubes)), tube_radii=fit.get("radii", []),
                placed_parts=n_placed)
    return Mesh(A, B, C, owner, rail, tubes, ents, meta)


def _number(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def build_map(level: NmoLevel) -> dict:
    """Anchors, sectors, placements and death boxes, read from the groups."""
    pos = lambda o: [float(v) for v in o.position]
    resets = sorted(level.members(RESET_GROUP), key=lambda o: _number(o.name))
    checks = sorted(level.members(CHECKPOINT_GROUP), key=lambda o: _number(o.name))
    start, end = level.members(START_GROUP), level.members(END_GROUP)
    sector_of = {}
    for g, names in level.groups.items():
        m = SECTOR.match(g)
        if m:
            for n in names:
                sector_of[n] = int(m.group(1))
    placements = []
    for o in level.objects.values():
        kinds = [g for g in o.groups if g.startswith("P_")]       # the placeholder's type is the P_ group it is in
        if not kinds:
            continue
        m = np.asarray(o.matrix, np.float64)
        placements.append(dict(name=o.name, type=kinds[0], sector=sector_of.get(o.name), position=pos(o),
                               x_axis=[float(v) for v in m[0, :3]], y_axis=[float(v) for v in m[1, :3]], z_axis=[float(v) for v in m[2, :3]]))
    placements.sort(key=lambda p: (p["sector"] or 0, p["type"], p["name"]))
    death = []
    for o in level.members(DEATH_GROUP):
        if o.vertices is not None:
            V = o.world_vertices()
            death.append(dict(name=o.name, lo=[float(v) for v in V.min(0)], hi=[float(v) for v in V.max(0)]))
    return dict(map=Path(level.path).name,
                start=pos(resets[0]) if resets else (pos(start[0]) if start else None),     # the ball appears at the first reset point
                start_marker=pos(start[0]) if start else None,
                reset_points=[pos(o) for o in resets],
                checkpoints=[pos(o) for o in checks],
                airship=pos(end[0]) if end else None,
                sectors=max(sector_of.values(), default=0),
                placements=placements, death_boxes=death,
                start_ball=None,     # not in the map file: the game decides it by the level slot the map is loaded into
                source="read from the map file's groups by python -m ballistics export")


def export_map(nmo_path, out_dir, name: str | None = None, bmap=None, templates=None) -> dict:
    """Read a map file and write `<name>-mesh.npz` and `<name>-map.json` into out_dir. Returns a summary."""
    level = read_nmo(nmo_path, bmap)
    name = name or Path(nmo_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_mesh(level, templates=templates, bmap=bmap)
    mesh.save(out_dir / (name + "-mesh.npz"))
    info = build_map(level)
    if templates is not None:
        info["hulls"] = placed_hulls(level, templates, bmap)       # the bodies' own collision shapes, where they are not the shown mesh
    (out_dir / (name + "-map.json")).write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf8")
    return dict(name=name, objects=len(level.objects), triangles=len(mesh), rail_triangles=int(mesh.rail.sum()), entities=len(mesh.entities), tubes=len(mesh.tubes),
                placed_parts=int(mesh.meta.get("placed_parts", 0)), placed_triangles=int(mesh.placed.sum()),
                checkpoints=len(info["checkpoints"]), reset_points=len(info["reset_points"]), sectors=info["sectors"],
                placements=len(info["placements"]), airship=info["airship"] is not None)
