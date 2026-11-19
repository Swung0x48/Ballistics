"""From a map file to the files the analyzer works on.

Input: a Ballance map (.nmo). Output, per map, in the output folder:

* `<name>-mesh.npz`: the static collision triangles (see `ballistics.geometry.Mesh`).
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


def build_mesh(level: NmoLevel, fixture: bool = True) -> Mesh:
    """The fixed bodies the player ball collides with, as triangles in world coordinates."""
    A, B, C, owner, rail, ents = [], [], [], [], [], []
    seen = set()
    for group in (FLOOR_GROUP, RAIL_GROUP):
        for o in level.members(group):
            if o.name in seen or o.vertices is None:
                continue
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
    if not A:
        z = np.zeros((0, 3))
        return Mesh(z, z, z, np.zeros(0, np.int64), np.zeros(0, bool), np.zeros((0, 7)), ents, {})
    A, B, C = np.concatenate(A), np.concatenate(B), np.concatenate(C)
    owner, rail = np.concatenate(owner), np.concatenate(rail)
    meta = dict(source="map file: objects in %s and %s, plus the game's fixed cube at the origin" % (FLOOR_GROUP, RAIL_GROUP),
                map=Path(level.path).name, triangles=int(len(A)), rail_triangles=int(rail.sum()),
                stoppers_left_out=len(level.groups.get(STOPPER_GROUP, [])))
    return Mesh(A, B, C, owner, rail, np.zeros((0, 7)), ents, meta)


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


def export_map(nmo_path, out_dir, name: str | None = None, bmap=None) -> dict:
    """Read a map file and write `<name>-mesh.npz` and `<name>-map.json` into out_dir. Returns a summary."""
    level = read_nmo(nmo_path, bmap)
    name = name or Path(nmo_path).stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_mesh(level)
    mesh.save(out_dir / (name + "-mesh.npz"))
    info = build_map(level)
    (out_dir / (name + "-map.json")).write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf8")
    return dict(name=name, objects=len(level.objects), triangles=len(mesh), rail_triangles=int(mesh.rail.sum()), entities=len(mesh.entities),
                checkpoints=len(info["checkpoints"]), reset_points=len(info["reset_points"]), sectors=info["sectors"],
                placements=len(info["placements"]), airship=info["airship"] is not None)
