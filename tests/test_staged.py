"""Coarse first, fine only at the breaks."""
import numpy as np

from ballistics import coarse, staged
from ballistics.geometry import Mesh


def quad(x0, x1, z0, z1, y):
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]           # wound to face up


def level(parts, goal):
    A, B, C, owner, ents = [], [], [], [], []
    for tris in parts:
        ents.append(dict(name="part%d" % len(ents), placed=False, type=""))
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)
    info = dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=list(goal), reset_points=[[4.0, 3.0, 4.0]], placements=[], death_boxes=[])
    g = coarse.build(m)
    coarse.add_steps(g, m)
    coarse.add_flights(g, m)
    return m, info, g


def test_a_plain_road_needs_no_search():
    m, info, g = level([quad(0, 40, 0, 8, 0.0)], (34.0, 2.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
    assert res["solved"] and res["replays"] and not res["breaks"]
    assert [sg["type"] for sg in res["segments"]] == ["follow"]


def test_a_gap_is_one_break_and_only_there_is_searched():
    m, info, g = level([quad(0, 30, 0, 8, 0.0), quad(37, 70, 0, 8, -1.0)], (62.0, 1.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=60_000, log=lambda *a: None)
    crossed = [b for b in res["breaks"] if b["status"] == "crossed"]
    assert res["solved"] and res["replays"] and len(crossed) == 1
    b = crossed[0]
    assert b["mode"] == "strict" and b["flights"] and 26.0 < b["near"][0] < 31.0
    kinds = [sg["type"] for sg in res["segments"]]
    assert kinds.count("cross") == 1 and kinds[0] == "follow" and kinds[-1] == "follow"
    rolled = sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "follow")
    assert rolled > 2 * sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "cross")      # most of the way is plain rolling


def test_a_wall_is_reported_unresolved_and_what_follows_is_conditional():
    riser = [(np.array((30.0, 0.0, 0.0)), np.array((30.0, 9.0, 0.0)), np.array((30.0, 9.0, 8.0))),
             (np.array((30.0, 0.0, 0.0)), np.array((30.0, 9.0, 8.0)), np.array((30.0, 0.0, 8.0)))]
    m, info, g = level([quad(0, 30, 0, 8, 0.0) + riser, quad(30, 60, 0, 8, 9.0)], (52.0, 11.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=3_000, log=lambda *a: None)
    assert not res["solved"] and res["unresolved"] and res["reached"]
    assert any(sg["type"] == "assumed" for sg in res["segments"])
    b = [b for b in res["breaks"] if b["status"] == "unresolved"][0]
    assert b["rise"] > 8.0 and [a["mode"] for a in b["attempts"]] == ["strict", "solid", "nohang", "free"]
