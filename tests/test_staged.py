"""Coarse first, fine only at the breaks."""
import numpy as np

from ballistics import coarse, staged
from ballistics.geometry import Mesh
from ballistics.takeoff import add_launches


def quad(x0, x1, z0, z1, y):
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]           # wound to face up


def level(parts, goal, launches=True):
    A, B, C, owner, ents = [], [], [], [], []
    for tris in parts:
        ents.append(dict(name="part%d" % len(ents), placed=False, type=""))
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)
    info = dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=list(goal), reset_points=[[4.0, 3.0, 4.0]], placements=[], death_boxes=[])
    g = coarse.build(m)
    coarse.add_steps(g, m)
    if launches:
        add_launches(g, m, info)
    return m, info, g


def test_a_plain_road_needs_no_search():
    m, info, g = level([quad(0, 40, 0, 8, 0.0)], (34.0, 2.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
    assert res["solved"] and res["replays"] and not res["breaks"]
    assert [sg["type"] for sg in res["segments"]] == ["follow"]


def test_a_gap_is_one_break_and_only_there_is_searched():
    # without launch edges on the graph, the gap is a break of the plan, and the fine level is asked only there
    m, info, g = level([quad(0, 30, 0, 8, 0.0), quad(37, 70, 0, 8, -1.0)], (62.0, 1.0, 4.0), launches=False)
    res = staged.solve_sector(m, info, g, 1, tries=60_000, log=lambda *a: None)
    crossed = [b for b in res["breaks"] if b["status"] == "crossed"]
    assert res["solved"] and res["replays"] and 1 <= len(crossed) <= 2      # the gap, and perhaps the follower's stop short of it
    b = next(b for b in crossed if b["flights"])                  # the one over the gap
    assert 16.0 < b["near"][0] < 31.0
    kinds = [sg["type"] for sg in res["segments"]]
    assert kinds.count("cross") == len(crossed) and kinds[0] == "follow" and kinds[-1] == "follow"
    # the search kept several ways, took the one worth most and adjusted it: never worse than the first it found
    for x in crossed:
        w = x["ways"]
        assert w["found"] >= 1 and (w["refined"]["arrives"], w["refined"]["arrives3"]) >= (w["chosen"]["arrives"], w["chosen"]["arrives3"]) >= (w["first"]["arrives"], w["first"]["arrives3"])
        assert x["arrives"] == w["refined"]["arrives"]
    rolled = sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "follow")
    assert rolled > sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "cross")          # most of the way is plain rolling


def test_a_fall_on_the_plan_is_taken_with_the_speed_the_ball_has():
    m, info, g = level([quad(0, 30, 0, 8, 0.0), quad(36, 70, 0, 8, -4.0)], (62.0, -2.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
    assert res["solved"] and res["replays"]
    assert [sg["type"] for sg in res["segments"]] == ["follow"]          # one stretch: no search, and no stop before the ledge
    tr = res["segments"][0]["trace"]
    air = np.flatnonzero(tr[:, 3] == 0)
    off = int(air[np.flatnonzero(np.diff(air, prepend=-9) > 1)][np.argmax([np.sum(tr[a:a + 20, 3] == 0) for a in air[np.flatnonzero(np.diff(air, prepend=-9) > 1)]])])
    before = np.hypot(np.diff(tr[off - 66:off, 0]), np.diff(tr[off - 66:off, 2])) * 66.0
    assert before.min() > 3.0 and before[-1] > 8.0                       # it rolled up to the ledge and off it, never slowing to a halt
    f = max(res["flights"], key=lambda f: f["down"])
    assert f["down"] > 3.0 and f["across"] > 5.0 and f["speed"] > 8.0
    # speed is one unbroken record: no jump of position anywhere along the sector
    assert np.linalg.norm(np.diff(tr[:, :3], axis=0), axis=1).max() < 0.45


def test_a_wall_is_reported_unresolved_and_what_follows_is_conditional():
    riser = [(np.array((30.0, 0.0, 0.0)), np.array((30.0, 9.0, 0.0)), np.array((30.0, 9.0, 8.0))),
             (np.array((30.0, 0.0, 0.0)), np.array((30.0, 9.0, 8.0)), np.array((30.0, 0.0, 8.0)))]
    m, info, g = level([quad(0, 30, 0, 8, 0.0) + riser, quad(30, 60, 0, 8, 9.0)], (52.0, 11.0, 4.0))
    res = staged.solve_sector(m, info, g, 1, tries=3_000, log=lambda *a: None)
    assert not res["solved"] and res["unresolved"] and res["reached"]
    assert any(sg["type"] == "assumed" for sg in res["segments"])
    b = [b for b in res["breaks"] if b["status"] == "unresolved"][0]
    assert b["rise"] > 8.0 and [a["mode"] for a in b["attempts"]] == ["strict", "free"] and b["would_pass"] is None


def test_a_loose_ball_across_the_road_is_knocked_aside():
    from ballistics import solve
    def ball(x, z, y=2.0, n=12):
        """A rough sphere of radius 2 as triangles: the shape the export puts into the mesh for a loose ball."""
        tris = []
        for i in range(n):
            for j in range(n // 2):
                def at(a, b):
                    th, ph_ = 2 * np.pi * a / n, np.pi * b / (n // 2)
                    return np.array([x + 2 * np.sin(ph_) * np.cos(th), y + 2 * np.cos(ph_), z + 2 * np.sin(ph_) * np.sin(th)])
                tris += [(at(i, j), at(i + 1, j), at(i + 1, j + 1)), (at(i, j), at(i + 1, j + 1), at(i, j + 1))]
        return tris

    for kind, speed_kept in (("Paper", 0.5), ("Wood", 0.15)):
        A, B, C, owner = [], [], [], []
        ents = [dict(name="road", placed=False, type="")] + [dict(name="ball", placed=True, type="P_Ball_" + kind, sector=1) for _ in range(2)]
        for k, tris in enumerate((quad(0, 60, 0, 8, 0.0), ball(26.0, 2.0), ball(26.0, 6.0))):       # two side by side: as wide as the road
            for t in tris:
                A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(k)
        m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)
        info = dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=[54.0, 2.0, 4.0], reset_points=[[4.0, 3.0, 4.0]], death_boxes=[],
                    placements=[dict(name="b%d" % i, type="P_Ball_" + kind, sector=1, position=[26.0, 2.0, z]) for i, z in enumerate((2.0, 6.0))])
        blocked = coarse.build(m)                                 # as fixed shapes they close the road
        assert blocked.route(blocked.nearest((4, 2, 4)), blocked.nearest((54, 2, 4)))[0] is None
        g = solve.graph_for(m, info, log=lambda *a: None)         # as bodies they only cost
        assert g.route(g.nearest((4, 2, 4)), g.nearest((54, 2, 4)))[0] is not None
        res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
        assert res["solved"] and res["replays"], kind
        end = res["segments"][-1]["end"]
        scene = end[3]
        assert len(scene) == 2 and np.abs(scene[:, :3] - np.array([[26.0, 2.011, 2.0], [26.0, 2.011, 6.0]])).max() > 1.0      # knocked away from where they lay
        tr = np.vstack([sg["trace"] for sg in res["segments"] if sg["type"] != "assumed"])
        assert ((tr[:, 0] > 24.0) & (tr[:, 0] < 28.0)).any()      # the route goes through where they were
