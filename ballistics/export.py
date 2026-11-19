"""Exporter: read a level's static collision geometry out of the game and write the files the analyzer works on.

The engine loads the level once and its level data is read; the ball is not driven and nothing is simulated beyond
the level's own start-up. This is parsing, not analysis: the game is the only thing that knows how to load its
maps, so it does the loading.

What is extracted, and how it is written, is decided here. Talking to the headless engine is not: that is done
through the Ballrain project's client (`ballrain.client`, `ballrain.headless`), which must be importable when this
module's `export_level` runs (pass its checkout with `--ballrain`, or install it). Nothing else in this package
needs Ballrain.

Files written per level into the output folder:

* `L<nn>-mesh.npz`: the collision triangles (see `ballistics.geometry.Mesh`).
* `L<nn>-map.json`: the level's anchors (spawn, checkpoints in order, airship) and the ball it starts with."""
from __future__ import annotations

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .geometry import Mesh

RAIL_GROUP = "Phys_FloorRails"


def to_world(vertices, matrix) -> np.ndarray:
    """Entity-space vertices to world space (row vectors; matrix is 4 x 4 or 4 x 3, translation in the last row)."""
    m = np.asarray(matrix, np.float64)
    return np.asarray(vertices, np.float64) @ m[:3, :3] + m[3, :3]


def build_mesh(level, fit_tubes=None) -> Mesh:
    """The static collision triangles of a loaded level.

    level: the level data as the engine client holds it. Used: `static_entities()` (the fixed physicalized
    entities the player ball collides with), `meshes_of(id)` (each with vertices, faces and a world matrix),
    `in_group(id, name)`, and the dictionaries `entities`, `phys_by_ent`, `ent_groups`; `static_key()` if present.
    fit_tubes: optional function (A, B, C, owner) -> (segments (S, 7), fallback mask, stats) that fits analytic
    tube segments to the rail triangles; without it the mesh has no tubes."""
    A, B, C, owner, rail, ents = [], [], [], [], [], []
    for ck in level.static_entities():
        is_rail = bool(level.in_group(ck, RAIL_GROUP))
        e = getattr(level, "entities", {}).get(int(ck))
        p = getattr(level, "phys_by_ent", {}).get(int(ck))
        idx = len(ents)
        ents.append(dict(name=getattr(e, "name", None), groups=sorted(getattr(level, "ent_groups", {}).get(int(ck), ())),
                         friction=float(getattr(p, "friction", 0.0) or 0.0), elasticity=float(getattr(p, "elasticity", 0.0) or 0.0),
                         surface=getattr(p, "surface", None)))
        for m in level.meshes_of(ck):
            F = np.asarray(m.faces, np.int64).reshape(-1, 3)
            if not len(F):
                continue
            V = to_world(m.vertices, m.matrix)
            A.append(V[F[:, 0]]); B.append(V[F[:, 1]]); C.append(V[F[:, 2]])
            owner.append(np.full(len(F), idx)); rail.append(np.full(len(F), is_rail))
    if not A:
        z = np.zeros((0, 3))
        return Mesh(z, z, z, np.zeros(0, np.int64), np.zeros(0, bool), np.zeros((0, 7)), ents, {})
    A, B, C = np.concatenate(A), np.concatenate(B), np.concatenate(C)
    owner, rail = np.concatenate(owner), np.concatenate(rail)
    tubes, loose = np.zeros((0, 7)), 0
    if fit_tubes is not None and rail.any():
        tubes, fallback, _ = fit_tubes(A[rail], B[rail], C[rail], owner[rail])
        tubes, loose = np.asarray(tubes, np.float64).reshape(-1, 7), int(np.asarray(fallback).sum())
    meta = dict(triangles=int(len(A)), rail_triangles=int(rail.sum()), tubes=int(len(tubes)), rail_triangles_not_tubes=loose,
                source="engine level data: fixed physicalized entities the player ball collides with")
    key = getattr(level, "static_key", None)
    if callable(key):
        meta["static_key"] = key()
    return Mesh(A, B, C, owner, rail, tubes, ents, meta)


@contextmanager
def _loaded_level(level: int, engine_root: Path, engine_build: Path, instance: str, work_dir: Path):
    """A headless engine with the level loaded and started, through Ballrain's client. Closed (killed) on exit."""
    try:
        from ballrain import tas
        from ballrain.client import DEFAULT_DELTA, AgentSession, SessionConfig
        from ballrain.headless import HeadlessLauncher
        from ballrain.protocol import Ev
    except ImportError as ex:
        raise SystemExit("the exporter talks to the engine through the Ballrain project's client, which is not importable (%s); "
                         "pass its checkout with --ballrain or install it" % ex)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        launcher = HeadlessLauncher(binary=str(engine_build.resolve()), root=str(engine_root.resolve()), socket="tcp", log=False)
        s = AgentSession(SessionConfig(level, Path(tmp), copy=instance, hello_timeout_s=20, obs_timeout_s=20), launcher)
        try:
            s.start()
            tr = s.tracker
            # let the level's own start-up finish (level data, reset points, the ball handed to the player): no keys
            # that move the ball are sent; Q only answers the start-up's "press a key" where a map asks for one
            if not tr.flags.started:
                s.run_until(0, [Ev.StartLevel], limit=3000)
            if not tr.flags.cam_nav and not tr.flags.ball_nav:
                s.run_until(0, [Ev.CamNavActive, Ev.BallNavActive], limit=400)
            if not tr.flags.cam_nav and not tr.flags.ball_nav:
                for _ in range(200):
                    s.act(tas.make_frames([DEFAULT_DELTA] * 2, [tas.KEY_Q, 0]), stop=[Ev.BallNavActive, Ev.CamNavActive])
                    if tr.flags.cam_nav or tr.flags.ball_nav:
                        break
            if not tr.flags.ball_nav:
                s.run_until(0, [Ev.BallNavActive], limit=3000)
            yield s
        finally:
            s.close(kill=True)


def export_level(level: int, out_dir, engine_root, engine_build, instance: str) -> dict:
    """Load `level` once in the headless engine and write its mesh and anchors into out_dir. Returns a summary."""
    from ballrain.fastreach import extract_rails          # tube fitting for rail topology (contacts use the triangles)
    from ballrain.path import level_anchors

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _loaded_level(level, Path(engine_root), Path(engine_build), instance, out_dir) as s:
        tr = s.tracker
        mesh = build_mesh(tr.level, extract_rails)
        anchors = level_anchors(tr)
        ball = tr.ball_type() if tr.flags.ball_nav else None
    mesh.meta["level"] = int(level)
    mesh.save(out_dir / ("L%02d-mesh.npz" % level))
    xyz = lambda v: None if v is None else [float(x) for x in np.asarray(v)[:3]]
    anchor = dict(level=int(level), start=xyz(anchors["start"]), checkpoints=[xyz(c) for c in anchors["checkpoints"]], airship=xyz(anchors["airship"]),
                  start_ball={0: "wood", 1: "paper", 2: "stone"}.get(None if ball is None else int(ball)),
                  source="read from the headless engine's level data by python -m ballistics export")
    (out_dir / ("L%02d-map.json" % level)).write_text(json.dumps(anchor, indent=1), encoding="utf8")
    return dict(level=int(level), triangles=len(mesh), rail_triangles=int(mesh.rail.sum()), tubes=len(mesh.tubes), entities=len(mesh.entities),
                checkpoints=len(anchor["checkpoints"]), airship=anchor["airship"] is not None, start_ball=anchor["start_ball"])


def add_ballrain(path) -> None:
    """Make a Ballrain checkout importable for this process."""
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
