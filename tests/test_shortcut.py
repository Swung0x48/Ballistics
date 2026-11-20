"""Shortcuts: islands the coarse level has no edge onto, and the other routes they give."""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from ballistics import coarse, shortcut, staged
from ballistics.geometry import Mesh
from ballistics.solve import world_for

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1 = [DATA / n for n in ("Level_01-mesh.npz", "Level_01-map.json", "Level_01-coarse-solid.pkl", "Level_01-staged-s1.pkl")]


def quad(x0, x1, z0, z1, y0, y1=None):
    """A floor facing up; y1: its height at z1 (a ramp along z)."""
    y1 = y0 if y1 is None else y1
    p = [np.array(v, float) for v in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z1), (x0, y1, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]


def detour():
    """A road out (y 0), a ramp down beside its end, and a road back (y -3.2) to the goal. Between the two roads lies
    a lower floor (y -2): a ball that rolls off the side of the first road falls 2 onto it, and from it a small
    step down leads onto the road back. The coarse level (no launch edges here) has no edge onto that floor, since
    a fall of 2 is not an edge of positions: it is an island."""
    parts = [quad(0, 40, 0, 8, 0.0), quad(32, 40, 8, 40, 0.0, -3.2), quad(0, 40, 40, 48, -3.2), quad(8, 24, 8, 40, -2.0)]
    A, B, C, owner, ents = [], [], [], [], []
    for tris in parts:
        ents.append(dict(name="part%d" % len(ents), placed=False, type=""))
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)
    info = dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=[4.0, -1.2, 44.0], reset_points=[[4.0, 3.0, 4.0]], placements=[], death_boxes=[])
    return m, info, coarse.build(m)


def test_a_floor_no_edge_leads_onto_is_an_island_and_gives_another_route():
    m, info, g = detour()
    quiet = lambda *a: None
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=quiet)
    assert res["solved"] and [sg["type"] for sg in res["segments"]] == ["follow"]          # the standard route goes round by the ramp
    assert res["segments"][0]["trace"][:, 0].max() > 30.0
    assert np.allclose(res["start"], [4.0, 2.0, 4.0], atol=0.1)                           # where the ball enters, not where it ends

    worlds = {k: world_for(m, info, k) for k in staged.ORDER}
    states = shortcut.route_states(worlds, res)
    assert states[-1]["t"] == res["ticks"] and np.abs(states[-1]["pos"] - res["segments"][-1]["end"][0]).max() < 1e-9
    isl = shortcut.islands(g, states)
    assert len(isl) == 1 and isl[0]["gain"] > 132                                          # the lower floor, worth more than a game second
    P = g.pos[isl[0]["places"]]
    assert np.allclose(P[:, 1], 0.0, atol=1e-6) and P[:, 0].min() >= 8.0 and P[:, 0].max() <= 24.0

    res["alternatives"] = shortcut.alternatives(m, info, g, 1, res, tries=20_000, most=2, log=quiet)
    rs = shortcut.routes(res)
    assert [r[0] for r in rs] == ["R1", "R2"]
    route = rs[1][1]
    sc = route["shortcut"]
    assert route["solved"] and route["replays"] and sc["saved"] > 132 and route["ticks"] == res["ticks"] - sc["saved"]
    assert sc["rejoin"] is not None and sc["rejoin"]["saved"] > 132                        # back on the standard route, ahead of it
    assert set(sc["nudged"]) == set(staged.NUDGES) and all(0 <= v <= 20 for v in sc["nudged"].values())
    kinds = [sg["type"] for sg in route["segments"]]
    assert kinds.count("cross") == 1 and kinds[0] == "follow" and kinds[-1] == "follow"
    tr = np.concatenate([sg["trace"] for sg in route["segments"]])
    assert tr[:, 0].max() < 30.0                                                           # it never went as far as the ramp
    assert np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).max() < 0.45                 # one unbroken trajectory
    t = shortcut.tolerance(route)
    assert t["passages"] == 1 and t["weakest"][1]["no"] == "S1"
    assert shortcut.tolerance(res)["weakest"] is None                                      # the standard route here is all closed loop
    out = json.loads(json.dumps(shortcut.listing("detour", [res])))
    rows = out["sectors"][0]["routes"]
    assert [r["id"] for r in rows] == ["R1", "R2"] and rows[1]["game_seconds"] < rows[0]["game_seconds"]
    assert rows[1]["shortcut"]["game_seconds_saved"] > 1.0 and rows[1]["segments"][0]["keys"]


def test_a_road_without_islands_has_only_the_standard_route():
    A, B, C = zip(*quad(0, 40, 0, 8, 0.0))
    m = Mesh(np.array(A), np.array(B), np.array(C), np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="road", placed=False, type="")])
    info = dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=[34.0, 2.0, 4.0], reset_points=[[4.0, 3.0, 4.0]], placements=[], death_boxes=[])
    g = coarse.build(m)
    res = staged.solve_sector(m, info, g, 1, tries=5_000, log=lambda *a: None)
    res["alternatives"] = shortcut.alternatives(m, info, g, 1, res, tries=5_000, log=lambda *a: None)
    assert res["alternatives"] == [] and [r[0] for r in shortcut.routes(res)] == ["R1"]


@pytest.mark.skipif(not all(p.is_file() for p in L1), reason="level 1 mesh, map, coarse graph or sector 1 result not in the data folder")
def test_level_1_the_west_wall_top_of_the_box_platform_is_the_best_island_of_sector_1():
    # the known shortcut of sector 1 (docs/game/levels.md) runs along the top of this wall; no search here
    m = Mesh.load(L1[0])
    info = json.load(open(L1[1], encoding="utf8"))
    g = pickle.load(open(L1[2], "rb"))
    res = pickle.load(open(L1[3], "rb"))
    states = shortcut.route_states({k: world_for(m, info, k) for k in staged.ORDER}, res)
    assert states[-1]["t"] == res["ticks"]
    isl = shortcut.islands(g, states)
    P = g.pos[isl[0]["places"]]
    on_wall = (P[:, 0] > -167.5) & (P[:, 0] < -159.5) & (P[:, 2] > -167) & (P[:, 2] < -127) & (np.abs(P[:, 1] - 22.9) < 0.5)
    assert on_wall.sum() > 100 and isl[0]["gain"] > 2 * 132
