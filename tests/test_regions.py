"""The coarse level as regions (ballistics.regions): which places a ball of each kind rolls to and back, and the way
through the regions with the fewest breaks. Made-up graphs only."""
from types import SimpleNamespace

import numpy as np

from ballistics import coarse, regions

W, S, P = coarse.WOOD, coarse.STONE, coarse.PAPER


def test_a_region_is_what_rolls_both_ways():
    # 0 <-> 1 <-> 2 roll both ways; 2 -> 3 only down; 3 <-> 4
    adj = [[1], [0, 2], [1, 3], [4], [3]]
    lab = regions.components(5, adj)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3]


def road(n, tk):
    """A road of n places rolling both ways, place k in a transformer's range of kind tk[k] (-1: none)."""
    roll = [[v for v in (u - 1, u + 1) if 0 <= v < n] for u in range(n)]
    fwd = [[(v, 1.0, "roll", -1, 0) for v in roll[u]] for u in range(n)]
    return roll, SimpleNamespace(n=n, fwd=fwd, lfwd=[[] for _ in range(n)], cfwd=[[] for _ in range(n)], tk=np.array(tk))


def test_a_transformer_across_the_road_cuts_it_for_the_other_kinds_and_rolling_into_it_transforms():
    roll, sg = road(5, [-1, -1, P, -1, -1])
    labs = regions.kind_labels(5, roll, sg.tk)
    assert labs[W][0] == labs[W][1] != labs[W][3] == labs[W][4] and labs[W][2] == -1          # a wood ball cannot get past it as wood
    assert len(set(labs[P])) == 1 and labs[P][2] >= 0                                          # a paper ball rolls through
    R = regions.Regions(labs, np.zeros((5, 3)))
    links = regions.links_of(sg, R)
    into = [k for k in links if k[1] == R.of(W, 1)]
    assert into == [("plain", R.of(W, 1), R.of(P, 2), "roll", 0, W, P)]                         # rolled into the range: a paper ball from there
    assert regions.describe(into[0]) == "roll, enters paper transformer range, becomes paper"
    # from the wood ball's side before it, the far side is reached only as paper
    breaks, path, _free = regions.connect(links, (R.of(W, 0), W, 0), {R.of(W, 4), R.of(P, 4)})
    assert breaks == 0 and [(R.name(r), k) for r, k, _b, _w, _key in path] == [(R.name(R.of(W, 0)), W), (R.name(R.of(P, 2)), P)]


def key(*k):
    return {k: dict(cost=1.0, count=1, at=(0, 0))}


def test_only_a_stone_ball_opens_a_gate_and_a_break_counts():
    # regions: 0 wood, 1 stone (both at the gate's push place), 2 stone past the gate (the goal); a break from 0 to 2
    links = {}
    links.update(key("mech", 1, 1, coarse.GATE_OPERATE, 0, 0, S, S))
    links.update(key("mech", 1, 2, coarse.GATE_THROUGH, 0, 0, S, S))
    links.update(key("plain", 0, 2, "gap", 0, W, S))
    b, path, free = regions.connect(links, (1, S, 0), {2})
    assert b == 0 and [(r, bits, what) for r, _k, bits, what, _key in path] == [(1, 0, "start"), (1, 1, "push_gate_open"), (2, 1, "through_gate")]
    b, path, free = regions.connect(links, (0, W, 0), {2})
    assert b == 1 and [what for *_x, what, _k in path] == ["start", "gap"]


def test_a_stone_ball_tips_the_seesaw_and_comes_back_over_it_only_from_the_new_low_end():
    # region 0 and 1 at the two ends; side 0: from 0 while bit 0 is 0; side 1: from 1 while bit 0 is 1
    for k in (W, S):
        links = {}
        links.update(key("mech", 0, 1, coarse.SEESAW, 0, 0, k, k))
        links.update(key("mech", 1, 0, coarse.SEESAW, 0, 1, k, k))
        b, path, free = regions.connect(links, (0, k, 0), {1})
        assert b == 0 and path[-1][2] == (1 if k == S else 0)                # tipped by the stone ball only
        _b, _p, free = regions.connect(links, (0, k, 0), {99})             # (nowhere: everything reachable)
        assert ((1, k, 1) in free) == (k == S) and ((1, k, 0) in free) == (k == W)
        # back over it from 1 needs it low at 1: only after a stone ball tipped it
        assert bool(regions._moves(("mech", 1, 0, coarse.SEESAW, 0, 1, k, k), path[-1][2])) == (k == S)


def test_a_paper_ball_cannot_push_a_board_over_so_that_link_is_not_made():
    n = 2
    sg = SimpleNamespace(n=n, fwd=[[], []], lfwd=[[], []], cfwd=[[(1, 1.0, "roll", 0, coarse.BOARD_PUSH, 0)], []], tk=np.array([-1, -1]))
    labs = regions.kind_labels(n, [[], []], sg.tk)
    R = regions.Regions(labs, np.zeros((n, 3)))
    made = {(int(R.kind[k[1]]), k[3]) for k in regions.links_of(sg, R)}
    assert made == {(W, coarse.BOARD_PUSH), (S, coarse.BOARD_PUSH)}


def test_a_cork_bridge_is_crossed_by_a_wood_or_paper_ball_either_way_whichever_way_it_lies():
    from ballistics import solve
    for ax in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8)):          # (nothing about which way it lies on the map)
        a = np.array(ax)
        side = np.cross([0.0, 1.0, 0.0], a)
        pos = [t * a + w * side + [0.0, 2.0, 0.0] for t in list(np.arange(-12, -8)) + list(np.arange(9, 13)) for w in (-1.0, 0.0, 1.0)]
        g = coarse.Graph(np.array(pos), np.zeros(len(pos), int), np.full(len(pos), -1), [[] for _ in pos], [])
        # its planks' hulls, in world as put: 14.8 long along its x, 5 wide, its top at 0.3
        V = [t * a + w * side + [0.0, y, 0.0] for t in (-7.4, 7.4) for w in (-2.5, 2.5) for y in (-0.3, 0.3)]
        info = dict(placements=[dict(name="C", type="P_Modul_29", sector=1, position=[0.0, 0.0, 0.0], x_axis=list(ax))],
                    hulls=[dict(placement="C", type="P_Modul_29", vertices=[list(v) for v in V])])
        e = solve.cork_edges(g, info, 1)
        ends = {(int(np.sign(g.pos[u] @ a)), int(np.sign(g.pos[v] @ a))) for u, v, _c, _n in e}
        assert ends == {(-1, 1), (1, -1)} and all(n == "C" for *_x, n in e)
    key = ("mech", 0, 1, coarse.CORK, 0, 0, W, W)
    assert regions._moves(key, 0) and regions._moves(("mech", 0, 1, coarse.CORK, 0, 0, P, P), 0)
    assert not regions._moves(("mech", 0, 1, coarse.CORK, 0, 0, S, S), 0)        # a stone ball breaks it and falls


def test_step_3_finds_a_gate_needed_and_what_opening_it_is_for():
    labs = np.array([[-1, -1], [0, 1], [-1, -1]])                               # two places, a stone ball's regions 0 and 1
    R = regions.Regions(labs, np.zeros((2, 3)))
    links = {}
    links.update(key("mech", 0, 0, coarse.GATE_OPERATE, 0, 0, S, S))
    links.update(key("mech", 0, 1, coarse.GATE_THROUGH, 0, 0, S, S))
    links.update(key("plain", 0, 1, "gap", 0, S, S))
    start, goals = (0, S, 0), [1]
    b, path, free = regions.connect(links, start, goals)
    res = dict(R=R, places=np.zeros((2, 3)), links=links, start=start, goal_regions=goals, breaks=b, path=path, mechs=[dict(name="G", type="gate")],
               sg=SimpleNamespace(bridges=[]), regions={r: dict(name=R.name(r), kind="stone", places=50) for r in range(2)}, placements=[])
    regions.mechanisms(res)
    (m,) = res["mechanisms"]
    assert m["states"] == ["closed", "open"] and m["initial"] == "closed" and m["need"] == "needed" and m["held_breaks"] == 1
    assert m["lets_through"]["open"] == [{"from": R.name(0), "to": R.name(1), "balls": "stone"}]
    (c,) = res["chain"]
    assert c["step"] == 1 and c["for"]["step"] == 2 and "open" in c["did"]


def test_what_ifs_a_locked_mechanism_never_changes_and_states_are_read_by_name():
    gate_open = ("mech", 0, 0, coarse.GATE_OPERATE, 0, 0, S, S)
    seesaw = ("mech", 0, 1, coarse.SEESAW, 0, 0, S, S)
    board = ("mech", 0, 1, coarse.BOARD_PUSH, 0, 0, W, W)
    assert regions._moves(gate_open, 0) and not regions._moves(gate_open, 0, lock=1)            # a locked gate is not opened
    assert regions._moves(board, 0) and not regions._moves(board, 0, lock=1)                    # nor a locked board pushed over
    assert regions._moves(seesaw, 0)[0][1] == 1 and regions._moves(seesaw, 0, lock=1)[0][1] == 0  # a locked seesaw is gone over, not tipped
    ov = regions.Overrides.parse(["P_Trafo_Paper_01"], ["P_Modul_01_03=开", "B=倒"], ["S"])
    assert ov and ov.without == {"P_Trafo_Paper_01"} and ov.state == {"P_Modul_01_03": "开", "B": "倒"} and ov.lock == {"S"}
    assert regions.Overrides.bit("gate", "开") == 1 and regions.Overrides.bit("board", "立") == 0 and regions.Overrides.bit("seesaw", "1") == 1
    assert regions.Overrides.bit("gate", "OPEN") == 1 and regions.Overrides.bit("t_board", "down") == 1
    assert regions.Overrides.bit("falling_rock", "standing") == 0
    labels = regions._seesaw_states([seesaw], {seesaw: {"at": (0, 1)}}, np.array([[2, 0, -3], [0, 0, 0]]))
    assert labels == ["low end at (2, -3)", "low end at the other end"]
    assert regions.Overrides.bit("seesaw", "low end at (2, -3)", labels) == 0
    assert regions.Overrides.bit("seesaw", "低端在另一头", labels) == 1
    assert regions.Overrides(state={"G": "开"}).plain([dict(name="G", type="gate", initial="open")])["state"] == {"G": "open"}
    assert not regions.Overrides()
    import pytest
    with pytest.raises(ValueError):
        regions.Overrides.bit("gate", "倒")
    with pytest.raises(ValueError):
        regions.Overrides.parse(state=["P_Modul_01_03"])


def _road_state_fixture(typ, xyz, edges, hulls, axis, mesh=None):
    from ballistics import solve
    X = np.array(axis)
    Y = np.array([0., 1., 0.])
    A = np.array([X, Y, np.cross(X, Y)])
    o = np.array([41., 7., -38.])
    pos = np.array(xyz) @ A + o
    adj = [[] for _ in pos]
    for u, v, c, kind in edges:
        adj[u].append((v, c, kind))
    g = coarse.Graph(pos, np.zeros(len(pos), int), np.full(len(pos), -1), adj, [])
    g._relaxed = coarse.Relaxed.__new__(coarse.Relaxed)
    g._relaxed.g, g._relaxed.fwd, g._relaxed.back = g, adj, [[] for _ in pos]
    obj = dict(name="M", type=typ, sector=1, position=o, x_axis=A[0], y_axis=A[1], z_axis=A[2])
    info = dict(placements=[obj], hulls=[dict(placement="M", body=body, vertices=vertices @ A + o, frame=np.r_[np.c_[A, np.zeros(3)], [np.r_[o, 1.]]]) for body, vertices in hulls],
                start=pos[0], airship=pos[-1], checkpoints=[], reset_points=[pos[0]])
    world = SimpleNamespace(props=[], scene=lambda: None, mesh=mesh(A, o) if mesh else None)

    def build(**kw):
        return solve.state_graph_for(g, info, 1, world, operate_cost=3.0, **kw)

    return info, g, world, build


def _hull_box(lo, hi):
    from itertools import product
    return np.array(list(product(*zip(lo, hi))), float)


ROAD_AXES = ((1., 0., 0.), (0., 0., 1.), (0.6, 0., 0.8))


def test_falling_rock_is_pushed_below_by_stone_opens_above_and_closes_below():
    xyz = [(0, -3, -5), (0, -3, 5), (-5, -3, -5), (-5, 2, -5), (-4, 2, 0), (4, 2, 0)]
    hulls = [("Schiebestein", _hull_box((-2.5, -0.22, -2.5), (2.5, 4.78, 2.5))),
             ("Kiste", _hull_box((-2.05, -5, -2.5), (2.05, -0.9, 2.5)))]
    edges = [(0, 1, 10, "roll"), (1, 0, 10, "roll"), (0, 2, 5, "roll"), (2, 3, 5, "step"), (3, 4, 5, "roll")]
    for axis in ROAD_AXES:
        info, g, world, build = _road_state_fixture("P_Modul_34", xyz, edges, hulls, axis)
        sg, read = build(cork=True)
        assert sg.M == 1 and read("stone", None) == (S << 1)
        for k in (W, S, P):
            s = k << 1
            assert (s, 1, 10, "roll") in sg.step(s, 0)
            assert not sg.step(s, 4)
            assert [(t, v, h) for t, v, _, h in sg.step(s | 1, 4)] == [(s | 1, 5, "over_falling_rock")]
            assert not any(v == 1 for _, v, _, _ in sg.step(s | 1, 0))
            assert np.isfinite(sg.togo(1)[s, 0]) and np.isinf(sg.togo(1)[s | 1, 0])
            assert any(h == "drop_falling_rock" for _, _, _, h in sg.step(s, 0)) == (k == S)
            assert np.isfinite(sg.togo(5)[s, 0]) == (k == S)
        s = S << 1
        nodes, kinds, bits = sg.plan(s, 0, sg.togo(5))
        assert nodes == [0, 0, 2, 3, 4, 5] and bits == [s] + [s | 1] * 5
        assert kinds[0] == "drop_falling_rock" and kinds[-1] == "over_falling_rock"
        assert np.isinf(sg.togo(5, lock=1)[s, 0])
        res = regions.analyse(g, info, 1, world, "stone")
        regions.mechanisms(res)
        (rock,) = res["mechanisms"]
        assert rock["need"] == "needed" and rock["lets_through"]["standing"] and rock["lets_through"]["dropped"]
        assert all(row["balls"] == "stone" and row["at"] for row in rock["switched"])
        assert res["chain"][0]["for"] and res["breaks"] == 0
        locked = regions.analyse(g, info, 1, world, "stone", overrides=regions.Overrides(lock=["M"]))
        assert locked["breaks"] is None and len(locked["plan"]) == 1
        dropped = regions.analyse(g, info, 1, world, "paper", overrides=regions.Overrides(state={"M": "已落"}, lock=["M"]))
        assert dropped["breaks"] == 0 and dropped["start"][2] == 1
        for plain in (build(cork=False)[0], build(cork=True, without=["M"])[0]):
            assert plain.M == 0 and np.isfinite(plain.togo(1)[W, 0]) and np.isinf(plain.togo(5)[S, 0])


def test_t_board_operation_is_separate_from_the_crossbar_and_paper_cannot_push():
    angle = np.radians(94.9)
    c, s = np.cos(angle), np.sin(angle)
    turn = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    hulls = [("head", _hull_box((-13., -1., -4.5), (-8., 0., 4.5)) @ turn.T),
             ("foot", _hull_box((-1., 0., -1.), (1., 3., 1.)))]
    xyz = [(3.5, 2, 0), (3.5, 2, -6), (-10, 2, -6), (-10, 2, 6)]
    for axis in ROAD_AXES:
        info, g, world, build = _road_state_fixture("P_Modul_37", xyz, [(0, 1, 6, "roll"), (1, 2, 14, "roll")], hulls, axis)
        sg, _ = build(cork=True)
        for k in (W, S, P):
            s = k << 1
            assert any(v == 0 and t == s | 1 for t, v, _, _ in sg.step(s, 0)) == (k != P)
            assert not sg.step(s, 2)
            assert [(t, v, h) for t, v, _, h in sg.step(s | 1, 2)] == [(s | 1, 3, "over_t_board")]
            assert np.isfinite(sg.togo(3)[s, 0]) == (k != P)
        nodes, kinds, _ = sg.plan(W << 1, 0, sg.togo(3))
        assert nodes == [0, 0, 1, 2, 3] and kinds == ["push_t_board", "roll", "roll", "over_t_board"]
        res = regions.analyse(g, info, 1, world)
        regions.mechanisms(res)
        (board,) = res["mechanisms"]
        assert board["states"] == ["up", "down"] and board["need"] == "needed"
        assert "run-up of about 12" in board["switch_note"] and board["lets_through"]["down"]
        preset = regions.analyse(g, info, 1, world, "paper", overrides=regions.Overrides(state={"M": "倒"}, lock=["M"]))
        assert preset["breaks"] == 0 and preset["start"][2] == 1
        assert not build(cork=True, without=["M"])[0].step(W, 2)


def test_t_board_restores_only_supported_floor_under_its_old_foot():
    from ballistics.geometry import Mesh

    def floor(A, o):
        # A continuous floor across the upright foot, with no fallen board in this lane.
        points = np.array([[0, 0, -8], [0, 0, 9], [5, 0, 9], [5, 0, -8]]) @ A + o
        a, b, c = points[[[0, 1, 2], [0, 2, 3]]].transpose(1, 0, 2)
        return Mesh(a, b, c, np.zeros(2, int), np.zeros(2, bool), np.empty((0, 7)), [{}])

    xyz = [(2.4, 2, -3), (2.4, 2, 3)]
    hulls = [("foot", _hull_box((1.9, 2.1, -1), (3, 4, 1)))]
    for axis in ROAD_AXES:
        *_, build = _road_state_fixture("P_Modul_37", xyz, [], hulls, axis, mesh=floor)
        sg, _ = build(cork=True)
        assert not sg.step(W, 0)
        assert [(t, v, h) for t, v, _, h in sg.step(1, 0)] == [(1, 1, "past_t_board")]
        assert np.isfinite(sg.togo(1)[1, 0])
        assert build(cork=True, without=["M"])[0].step(W, 0)[0][3] == "roll"
        *_, unsupported = _road_state_fixture("P_Modul_37", xyz, [], hulls, axis)
        assert not unsupported(cork=True)[0].step(1, 0)


def test_box_float_rows_allow_only_the_right_balls_both_ways_and_without_any_member():
    for axis in ROAD_AXES:
        for n, allowed in ((2, {P, W}), (5, {P})):
            info, g, world, build, pitch = _float_row_fixture(n, axis)
            length = (n - 1) * pitch
            sg, read = build(cork=True)
            (row,) = sg.floats
            assert row["name"] == "P_Modul_41_01-%02d" % n
            assert sg.M == 0 and sg.loc[1] < 0
            assert sg.step(S, sg.loc[2]) == [(S, sg.loc[3], 2, "roll")]
            assert build(cork=False)[0].step(S, 1) and not build(cork=False)[0].floats
            for name, k in coarse.KIND_INDEX.items():
                assert read(name, None) == k
                for u, v in ((0, 4), (4, 0)):
                    got = sg.step(k, sg.loc[u])
                    remaining = sg.togo(v)[k, u]
                    if k in allowed:
                        ((t, w, cost, how),) = got
                        assert (t, w, how) == (k, sg.loc[v], "float_row")
                        assert np.isclose(cost, length + 8) and np.isclose(remaining, cost)
                    else:
                        assert not got and np.isinf(remaining)
                res = regions.mechanisms(regions.analyse(g, info, 1, world, name))
                assert res["breaks"] == (0 if k in allowed else None)
                (m,) = res["mechanisms"]
                assert m["on_way"] == (k in allowed) and m["without_breaks"] is None and m["boxes"] == n
                assert m["passes"] == {kind: kk in allowed for kind, kk in coarse.KIND_INDEX.items()}
                if k in allowed:
                    assert regions.describe(res["path"][-1][-1]) == "%s crosses float row (%d boxes)" % (regions.KINDS[k], n)
                for member in [*row["placements"], row["name"]]:
                    ov = regions.Overrides.parse(without=[member])
                    removed = regions.mechanisms(regions.analyse(g, info, 1, world, name, overrides=ov))
                    assert removed["breaks"] is None and not removed["mechanisms"]


def _float_row_fixture(n, axis, xyz=None, edges=()):
    from ballistics import things

    pitch = things.load()["P_Modul_41"].numbers["pitch"]["v"]
    if xyz is None:
        length = (n - 1) * pitch
        points = [(-4, 2, 0), (0, 2, 0), (length / 2, -3, -1), (length / 2, -3, 1), (length + 4, 2, 0)]
        edges = [(0, 1, 4, "roll"), (1, 4, length + 4, "step"),
                 (0, 4, length + 8, "bridge"), (4, 0, length + 8, "launch"),
                 (2, 3, 2, "roll"), (3, 2, 2, "roll")]
    else:
        points = xyz(pitch)
    info, g, world, build = _road_state_fixture("P_Modul_41", points, edges, [], axis)
    template = info["placements"].pop()
    for i in reversed(range(n)):
        a = np.array(axis)
        o = template["position"] + i * pitch * a
        name = "P_Modul_41_%02d" % (i + 1)
        # Some hulls face the other way; the centre line and outside ends are unchanged.
        x = (-1)**i * a
        A = np.array([x, [0, 1, 0], np.cross(x, [0, 1, 0])])
        info["placements"].append(dict(template, name=name, position=o, x_axis=x))
        info["hulls"].append(dict(placement=name, vertices=_hull_box((-2.5, -1.5, -5), (2.5, 0, 5)) @ A + o))
    return info, g, world, build, pitch


def test_box_float_row_drops_from_far_end_within_reach_and_never_back_up():
    from ballistics import physics, things
    from ballistics.geometry import BALL_R

    catalogue = things.load()
    nums = catalogue["P_Modul_41"].numbers
    factor = catalogue.rules["units"]["numbers"]["time_factor"]["v"]
    t = np.sqrt(2 * 10 / physics.GRAVITY)
    for axis in ROAD_AXES:
        for reverse in (False, True):
            def xyz(pitch):
                wood = pitch / (factor * nums["wood_seconds_per_box"]["v"]) * t + BALL_R
                paper = pitch / (factor * min(nums["paper_seconds_per_box"]["v"])) * t + BALL_R
                points = [(-4, 2, 0), (2 * pitch + 2.5 + wood - 0.1, -8, 0),
                          (2 * pitch + 2.5 + (wood + paper) / 2, -8, 0), (2 * pitch + 2.5 + paper + 0.1, -8, 0)]
                return [(2 * pitch - x, y, z) for x, y, z in points] if reverse else points

            info, g, world, build, _ = _float_row_fixture(3, axis, xyz)
            sg, _ = build(cork=True)
            for k, targets in ((P, {1, 2}), (W, {1}), (S, set())):
                assert {sg.glob[v] for _, v, _, how in sg.step(k, sg.loc[0]) if how == "float_row"} == targets
                for v in (1, 2, 3):
                    assert not sg.step(k, sg.loc[v])
                    assert np.isfinite(sg.togo(v)[k, sg.loc[0]]) == (v in targets)
                    assert np.isinf(sg.togo(0)[k, sg.loc[v]])
            (m,) = regions.mechanisms(regions.analyse(g, info, 1, world, "wood"))["mechanisms"]
            assert {(e["box"], e["direction"], e["balls"]) for e in m["exits"]} == {(3, "end", "wood"), (3, "end", "paper")}
            assert any("drops from float box 3 (end)" in regions.describe(key)
                       for key in regions.analyse(g, info, 1, world)["links"] if key[0] == "mech")


def test_box_float_row_side_exits_count_from_either_bank_and_prune_by_region():
    def xyz(pitch):
        return [(-4, 2, 0), (pitch, 1, 8), (pitch, 1, 8.2), (4 * pitch + 1, 1, 9.6),
                (5 * pitch + 4, 2, 0), (pitch, 2, -6.5), (pitch, 5, -6.5), (pitch, -8, 0)]

    for axis in ROAD_AXES:
        _, _, _, build, _ = _float_row_fixture(6, axis, xyz, [(1, 2, 0.2, "roll"), (2, 1, 0.2, "roll")])
        sg, _ = build(cork=True)

        def departures(u, k):
            return {sg.glob[v]: (n, direction) for v, _, _, _, _, side in sg.cfwd[sg.loc[u]]
                    if side & (1 << k) for n, direction in [coarse.float_exit(side)] if direction != coarse.FLOAT_THROUGH}

        left_p, left_w, right_w = departures(0, P), departures(0, W), departures(4, W)
        assert set(left_w) == {1, 5} and left_w[1][0] <= 3
        assert set(left_p) == {1, 3, 5} and left_p[3][0] == 5
        assert left_p[3][1] == left_w[1][1] != left_w[5][1]
        assert set(right_w) == {3} and right_w[3][0] == 2
        assert not departures(0, S) and not departures(4, S)


def _flap_fixture(axis, sampled=True):
    from ballistics import things
    from ballistics.geometry import BALL_R

    nums = things.load()["P_Modul_19"].numbers
    a0, a1 = nums["gap_x"]["v"]["lane_A"]
    b0, b1 = nums["gap_x"]["v"]["lane_B"]
    az0, az1 = nums["lane_z"]["v"]["lane_A"]
    bz0, bz1 = nums["lane_z"]["v"]["lane_B"]
    az, bz = (az0 + az1) / 2, (bz0 + bz1) / 2
    h = nums["standing_flap_height"]["v"]
    points = [(a0 - BALL_R - 0.1, BALL_R, az), (a1 + 1, BALL_R, az),
              (b0 - 1, BALL_R, bz), (b1 + BALL_R + 0.1, BALL_R, bz),
              ((a0 + a1) / 2, BALL_R if sampled else -5, az),
              (a0 - BALL_R - 0.1, 0.3, az), (b1 + BALL_R + 0.1, 0.3, bz),
              (a0 - 3, -4, az), (a1 + 2, -4, az),
              (a0 - 3, h + 2, az), (a1 + 2, h + 2, az),
              (b0 - 2, h + 2, bz), (b1 + 3, h + 2, bz),
              (0.1, BALL_R + 0.7, az), (1, BALL_R + 0.7, az)]
    old = [(7, 8, 10, "roll"), (8, 7, 10, "roll"), (9, 10, 10, "launch"), (12, 11, 10, "launch"), (13, 14, 1, "rail")]
    if sampled:
        old += [(0, 4, 5, "roll"), (4, 0, 5, "roll"), (1, 4, 5, "roll"), (4, 1, 5, "roll")]
    hulls = [("plate", _hull_box((a0, -0.25, az0), (a1, 0.25, az1))),
             ("plate", _hull_box((b1 - 0.5, -0.25, bz0), (b1, h, bz1)))]
    info, g, world, build = _road_state_fixture("P_Modul_19", points, old, hulls, axis)
    # Both launch endpoints and the lift endpoints are above the wall. Only the lift's stored path hits it.
    g.info["launches"] = {(u, v): {"way": g.pos[[u, v]]} for u, v in ((9, 10), (12, 11))}
    mid = (g.pos[11] + g.pos[12]) / 2
    mid[1] = g.pos[3, 1]
    g.info["lifts"] = {(11, 12): dict(cost=10, way=[g.pos[11], mid, g.pos[12]], sector=1)}
    g.kind[[5, 6, 13, 14]] = 1  # low push sites, plus independent rail at deck height
    return info, g, world, build


def test_two_way_flap_states_pushes_guards_and_overrides():
    for axis in ROAD_AXES:
        for sampled in (False, True):
            info, g, world, build = _flap_fixture(axis, sampled)
            sg, state = build(cork=True)
            assert sg.M == 1 and state("wood", None) == 0
            plain, _ = build()
            assert plain.M == 0 and not any(plain.cfwd)

            def steps(k, pose, u, lock=0, graph=sg):
                return {(s & ((1 << graph.M) - 1), int(graph.glob[v]), how)
                        for s, v, _, how in graph.step((k << graph.M) | pose, graph.loc[u], lock)}

            for k in (W, S, P):
                a, b = (k << 1), (k << 1) | 1
                assert np.isfinite(sg.togo(1, lock=1)[a, 0]) and np.isfinite(sg.togo(0, lock=1)[a, 1])
                assert np.isinf(sg.togo(1, lock=1)[b, 0]) and np.isinf(sg.togo(0, lock=1)[b, 1])
                assert np.isinf(sg.togo(2, lock=1)[a, 3]) and np.isinf(sg.togo(3, lock=1)[a, 2])
                assert (1, 2, "over_flap") in steps(k, 1, 3) and (1, 3, "over_flap") in steps(k, 1, 2)
                for u, v, pose in ((3, 2, 0), (6, 2, 0), (0, 1, 1), (5, 1, 1)):
                    assert ((1 - pose, v, "push_flap") in steps(k, pose, u)) == (k != P)
                    assert not steps(k, pose, u, lock=1)
                    assert np.isfinite(sg.togo(v)[(k << 1) | pose, u]) == (k != P)
                assert not steps(k, 0, 2) and not steps(k, 1, 1)
                assert (1, 8, "roll") in steps(k, 1, 7)
                for pose in (0, 1):
                    assert (pose, 10, "launch") in steps(k, pose, 9)
                    assert (pose, 11, "launch") in steps(k, pose, 12)
                    assert (pose, 14, "rail") in steps(k, pose, 13)
            assert not steps(P, 0, 11)
            assert (1, 12, "lift") in steps(P, 1, 11)
            assert np.isinf(sg.togo(12)[P << 1, 11])
            assert np.isfinite(sg.togo(12)[(P << 1) | 1, 11])
            gone, _ = build(cork=True, without={"M"})
            for k in (W, S, P):
                assert all(not steps(k, 0, u, graph=gone) for u in range(7))
                assert (0, 12, "lift") in steps(P, 0, 11, graph=gone)
            assert (0, 14, "rail") in steps(P, 0, 13, graph=gone)

            info["airship"] = g.pos[1]
            ov = regions.Overrides.parse(state=["M=B姿态"], lock=["M"])
            r = regions.mechanisms(regions.analyse(g, info, 1, world, overrides=ov))
            (m,) = r["mechanisms"]
            assert r["start"][2] == 1 and m["initial"] == "pose B" and r["breaks"] is None
            removed = regions.mechanisms(regions.analyse(g, info, 1, world, overrides=regions.Overrides.parse(without=["M"])))
            assert removed["breaks"] is None and not removed["mechanisms"]
            assert all({k for row in rows for k in row["balls"].split(", ")} == {"wood", "stone", "paper"}
                       for rows in m["lets_through"].values())
            assert {row["from_state"] for row in m["switched"]} == {"pose A", "pose B"}
            assert all(row["balls"] in ("wood", "stone") and row["at"] for row in m["switched"])
            assert all({k for row in m["switched"] if row["from_state"] == state for k in row["balls"].split(", ")}
                       == {"wood", "stone"} for state in m["states"])
            assert all(regions.Overrides.bit("flap", label) == value for label, value in
                       (("A", 0), ("A 姿态", 0), ("pose A", 0), ("B", 1), ("B姿态", 1), ("pose B", 1)))


def test_two_way_flap_region_route_must_cross_a_before_flipping():
    for axis in ROAD_AXES:
        info, g, world, _ = _flap_fixture(axis, sampled=False)
        g.adj[1].append((3, 20, "roll"))  # A far bank reaches the B push side
        g.adj[2].append((1, 20, "launch"))  # after crossing B, only the wrong side to push A is reachable
        info["airship"] = g.pos[2]
        r = regions.mechanisms(regions.analyse(g, info, 1, world))
        assert r["breaks"] == 0 and r["mechanisms"][0]["held_breaks"] is None
        assert r["mechanisms"][0]["need"] == "needed"
        used = [key for *_, key in r["path"] if key and key[0] == "mech"]
        assert [(key[3], key[5]) for key in used] == [(coarse.FLAP_OVER, 0), (coarse.FLAP_PUSH, 0)]
        assert "lane B as lane A closes" in r["chain"][0]["did"]
        labs = r["R"]
        sg = r["sg"]
        assert regions.connect(r["links"], (labs.of(W, sg.loc[2]), W, 1), [labs.of(W, sg.loc[0])])[0] is None
        locked = regions.analyse(g, info, 1, world, overrides=regions.Overrides.parse(lock=["M"]))
        assert locked["start"][2] == 0 and locked["breaks"] is None


def _pendulum_fixture(axis, *, limited=False, gap=False, cached=False, post=False):
    from ballistics import things
    from ballistics.geometry import Mesh
    nums = things.load()["P_Modul_17"].numbers
    reach = nums["arm_reach"]["v"]
    x = 0.8 * reach

    def floor(A, o):
        def strip(z0, z1, y=0.):
            return [[-reach, y, z0], [-reach, y, z1], [reach, y, z1], [reach, y, z0]]

        strips = [strip(-8, -1.5), strip(1.5, 8)] if gap else [strip(-8, 8)]
        if post:
            strips.append(strip(-1., 1., y=1.6))   # fixed obstruction, not part of the rotating body
        v = (np.array(strips) @ A + o)[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)
        return Mesh(*v.transpose(1, 0, 2), np.zeros(len(v), int), np.zeros(len(v), bool), np.empty((0, 7)), [{}])

    points = [(-x, 2, 3), (-x, 2, -3), (x, 2, 3), (x, 2, -3), (-x, 4, 3), (-x, 4, -3)]
    hulls = [("Modul17_Dreharme", _hull_box((a, 1.4, -0.2), (b, 3., 0.2)))
             for a, b in ((-reach, -0.75 * reach), (0.75 * reach, reach))]
    hulls.append(("Modul17_Dreharme", _hull_box((-0.6 * reach, 1.8, -0.4), (0.6 * reach, 2.4, 0.4))))
    old = [(4, 5, 6, "launch")]
    if cached:
        old += [(0, 1, 6, "rail"), (1, 0, 6, "rail")]
    info, g, world, build = _road_state_fixture("P_Modul_17", points, old, hulls, axis, mesh=floor)
    if cached:
        g.kind[[0, 1]] = 1
        g.pos[[0, 1], 1] -= 0.5  # rail ball centres are below the floor ball centres, still inside the target band
    info["reset_points"][0] = g.pos[0]
    info["airship"] = g.pos[1]
    name = "M"
    if limited:
        # Only the catalogue lookup metadata is shared with the maps; all geometry is synthetic.
        info["map"] = "Level_13.NMO"
        name = next(iter(nums["l13_stopper_deg"]["v"]))
        info["placements"][0]["name"] = name
        for h in info["hulls"]:
            h["placement"] = name
    p = info["placements"][0]
    pos = p["position"] - x * p["x_axis"] + 2 * p["y_axis"] + 2 * p["z_axis"]
    info["placements"].append(dict(name="board", type="P_Modul_25", position=pos, sector=1))
    g.info["launches"] = {(4, 5): dict(way=g.pos[[4, 5]])}
    return info, g, world, build, name


def test_pendulum_pushes_both_ways_for_every_ball_and_reports_candidates(tmp_path):
    for axis in ROAD_AXES:
        info, g, world, build, name = _pendulum_fixture(axis)
        sg, read = build(cork=True)
        assert sg.M == 0 and read("paper", None) == P
        for k in (W, S, P):
            for u, v in ((0, 1), (1, 0), (2, 3), (3, 2)):
                assert any(t == k and w == v and how == "push_pendulum" for t, w, _, how in sg.step(k, u))
                assert np.isfinite(sg.togo(v)[k, u])
        assert not build(cork=False)[0].step(W, 0)
        res = regions.analyse(g, info, 1, world)
        assert res["breaks"] == 0 and "pushes past the pendulum" in regions.describe(res["path"][-1][4])
        regions.mechanisms(res)
        (rod,) = res["mechanisms"]
        assert rod["name"] == name and rod["states"] == [] and rod["on_way"] and rod["without_breaks"] == 0
        assert {e["balls"] for e in rod["crossings"]} == {"wood", "stone", "paper"}
        assert [q["name"] for q in rod["candidates"]] == ["board"]
        if axis == ROAD_AXES[0]:
            picture = tmp_path / "pendulum.png"
            regions.draw_mechanisms(world.mesh, res, picture)
            assert picture.is_file()


def test_pendulum_limit_is_directional_and_removing_rod_leaves_supported_road():
    for axis in ROAD_AXES:
        for cached in (False, True):
            info, g, world, build, name = _pendulum_fixture(axis, limited=True, cached=cached)
            sg, _ = build(cork=True)
            plain, _ = build(cork=True, without=[name])
            assert sg.M == plain.M == 0 and not plain.pendulums
            for k in (W, S, P):
                assert not sg.step(k, 0)  # -X end, from +Z: negative torque about the placement's Y
                assert not sg.step(k, 3)  # +X end, from -Z: also negative torque
                assert any(v == 0 and how == "push_pendulum" for _, v, _, how in sg.step(k, 1))
                assert any(v == 3 and how == "push_pendulum" for _, v, _, how in sg.step(k, 2))
                assert any(v == 5 and how == "launch" for _, v, _, how in sg.step(k, 4))
                assert np.isinf(sg.togo(1)[k, 0]) and np.isfinite(plain.togo(1)[k, 0])
                assert any(v == 1 and how == ("rail" if cached else "roll") for _, v, _, how in plain.step(k, 0))
            res = regions.analyse(g, info, 1, world)
            regions.mechanisms(res)
            (rod,) = res["mechanisms"]
            assert not rod["on_way"] and rod["without_breaks"] == 0 and rod["stopper_deg"] < 0 and rod["blocked"]


def test_pendulum_does_not_bridge_empty_or_blocked_fixed_floor():
    for axis in ROAD_AXES:
        for gap, post in ((True, False), (False, True)):
            _, _, _, build, name = _pendulum_fixture(axis, gap=gap, post=post)
            for without in ([], [name]):
                sg, _ = build(cork=True, without=without)
                assert not sg.step(W, 0)


def test_a_wood_ball_opens_a_gate_only_on_a_gate_wood_edge_and_the_note_reads_the_catalogue():
    for side, wood in ((0, False), (coarse.GATE_WOOD, True)):
        assert bool(regions._moves(("mech", 0, 0, coarse.GATE_OPERATE, 0, side, W, W), 0)) == wood
        assert bool(regions._moves(("mech", 0, 0, coarse.GATE_OPERATE, 0, side, S, S), 0))
        assert not regions._moves(("mech", 0, 0, coarse.GATE_OPERATE, 0, side, 2, 2), 0)
        assert (W in regions._needs(("mech", 0, 0, coarse.GATE_OPERATE, 0, side, W, W), W, 0)[0][1]) == wood
    note = regions._gate_note()
    assert "10.40" in note and "2.5" in note and "[ASSUMED]" in note



def test_report_vocabulary_is_ascii_and_figures_keep_chinese():
    plain = [("plain", 0, 1, kind, 0, coarse.WOOD, coarse.WOOD)
             for kind in ("launch", "drop", "step", *regions.ROLL, *regions.BREAK)]
    plain += [("plain", 0, 1, "roll", 2, coarse.WOOD, coarse.PAPER, 1),
              ("plain", 0, 1, "rail", 2, coarse.WOOD, coarse.WOOD, 5, 1)]
    lifts = [("lift", 0, 1, coarse.PAPER), ("lift", 0, 1, coarse.STONE, 2, 1)]
    mech = [("mech", 0, 1, typ, 0, side, k, coarse.PAPER)
            for typ in regions.MECH if typ != coarse.FLOAT_ROW
            for side in (0, 1) for k in (coarse.WOOD, coarse.STONE)]
    floats = [("mech", 0, 1, coarse.FLOAT_ROW, 0, coarse.float_side(3, direction, 5), coarse.WOOD, coarse.WOOD)
              for direction in (coarse.FLOAT_THROUGH, coarse.FLOAT_MINUS, coarse.FLOAT_PLUS, coarse.FLOAT_END)]
    for key in plain + lifts + mech + floats:
        text = regions.describe(key)
        assert text and text.isascii(), (key, text)
    assert all(state.isascii() for states in regions.STATES.values() for state in states)
    assert regions.NO_STATE_BIT == "no state bit"
    assert set(regions.NEED.values()) == {"needed", "used, can be bypassed", "not used by this way", "goal not reached"}
    assert all(note.isascii() for note in regions.SWITCH_NOTES.values())
    assert (regions.FLOAT_NOTE[0] % (8, 4)).isascii()
    assert regions._gate_note().isascii()
    assert regions.describe(plain[-2], "zh") == "滚，要机关位 2，要落石未落（机关位 1）进纸变的范围，变纸球"
    assert regions.describe(lifts[0], "zh") == "纸球乘风扇上升"
    assert regions.describe(floats[1], "zh") == "木球从箱式浮板第 3 个（侧面−）落下"
    gate_push = ("mech", 0, 1, coarse.GATE_OPERATE, 0, 0, coarse.WOOD, coarse.WOOD)
    assert regions.describe(gate_push, "zh") == "木球推开栅栏（要助跑【假】）（机关 0）"
    assert [regions._chinese(state) for state in regions.STATES["flap"]] == ["A 姿态", "B 姿态"]
    assert regions._chinese(regions.NEED["bypassable"]) == "用到，可以绕开"
    assert regions._chinese(regions.SWITCH_NOTES["board"]) == "木球、石球推倒；不会再立起来"
    assert regions._gate_note("zh") == (
        "石球推开；木球撞前 10.40 以上实测推开 2.94–3.01，过了路口打开要的 2.5，也算推得开（粗层不验助跑【假】）；纸球推不动；不会再关")


def _pendulum_target(info, name, typ, pos, hull=None):
    rod = info["placements"][0]
    frame = np.asarray([rod[a + "_axis"] for a in "xyz"])
    o = rod["position"]
    placement = dict(rod, name=name, type=typ, position=np.asarray(pos) @ frame + o)
    info["placements"].append(placement)
    if hull is not None:
        info["hulls"].append(dict(placement=name, body=name, vertices=hull @ frame + o))
    return placement


def test_pendulum_remote_board_uses_opposite_end_turn_and_lock():
    import json

    for axis in ROAD_AXES:
        info, g, world, build, rod = _pendulum_fixture(axis)
        info["placements"].pop()  # replace the candidate-only board with collision geometry
        b = "P_Modul_25_01"
        board = _pendulum_target(info, b, "P_Modul_25", (-11, 2, 2),
                                 _hull_box((-12, 0, 1.5), (-10, 3, 2.5)))
        # A target without a cached crossing still has a state when remotely operated.
        g.info["boards"] = {b: dict(sector=1, edges=[], back=[], fall=board["z_axis"])}
        sg, state_of = build(cork=True)
        assert sg.M == 1 and build(cork=False)[0].M == 0
        for k, kind in enumerate(regions.KINDS):
            s = state_of(kind, None)
            for gu in range(4):
                moves = [(after, v) for after, v, _, how in sg.step(s, sg.loc[gu]) if how == "pendulum_board"]
                # Local +X hit, motion toward -Z gives a positive Y torque.
                assert moves == ([(s | 1, sg.loc[gu])] if gu == 2 and k != coarse.PAPER else [])
                assert not any(how == "pendulum_board" for _, _, _, how in sg.step(s, sg.loc[gu], lock=1))
        assert not build(cork=True, without=(rod,))[0].pendulums
        assert build(cork=True, without=(b,))[0].M == 0
        res = regions.analyse(g, info, 1, world)
        keys = [k for k in res["links"] if k[0] == "mech" and k[3] == coarse.PENDULUM_BOARD]
        assert {k[6] for k in keys} == {coarse.WOOD, coarse.STONE}
        for key in keys:
            assert regions.describe(key, lang="en").isascii()
            assert "另一端推倒推板" in regions.describe(key, lang="zh")
            assert not regions._moves(key, 0, lock=1)
        report = regions.summary_mechanisms(regions.mechanisms(res))
        assert json.dumps(report, ensure_ascii=False).isascii()
        pend = next(m for m in report["mechanisms"] if m["name"] == rod)
        target = next(m for m in report["mechanisms"] if m["name"] == b)
        assert pend["linked"] and target["switched"] == pend["linked"]
        assert {r["target"] for r in pend["linked"]} == {b}
        assert all(line.isascii() for line in regions._pendulum_notes(pend, lang="en"))
        assert any("远程推倒" in line for line in regions._pendulum_notes(pend, lang="zh"))
        # A target wholly inside the swept disk but not on its rim remains candidate-only.
        h = info["hulls"][-1]
        h["vertices"] += board["x_axis"] * 5
        assert build(cork=True)[0].M == 0


def test_pendulum_remote_board_respects_tangent_and_stopper():
    for axis in ROAD_AXES:
        for limited, fall in ((False, -1), (False, 1), (True, -1)):
            info, g, _, build, _ = _pendulum_fixture(axis, limited=limited, cached=True)
            info["placements"].pop()
            board = _pendulum_target(info, "B", "P_Modul_25", (-11, 2, -0.6),
                                     _hull_box((-12, 0, -1), (-10, 3, -0.3)))
            g.info["boards"] = {"B": dict(sector=1, edges=[], fall=fall * board["z_axis"])}
            sg, _ = build(cork=True)
            # Negative turn would push this target. Its contact is before the stopper,
            # but the mover's negative crossing does not have room for a ball to pass.
            assert bool(sg.M) == (fall < 0 and not limited)


def test_pendulum_clears_fan_stack_and_all_members_alias_overrides():
    for axis in ROAD_AXES:
        info, g, world, build, rod = _pendulum_fixture(axis)
        fan = _pendulum_target(info, "F", "P_Modul_18", (-11, -4, 2))
        _pendulum_target(info, "B1", "P_Box", (-11, 1, 2), _hull_box((-12, -1, 1), (-10, 3, 3)))
        _pendulum_target(info, "B2", "P_Box", (-11, 5, 2), _hull_box((-12, 3, 1), (-10, 7, 3)))
        g.info["lifts"] = {(1, 5): dict(sector=1, cost=2., fan=fan["position"], way=g.pos[[1, 5]])}
        sg, state_of = build(cork=True)
        assert sg.M == 1 and sg.mechs[0]["placements"] == ("B1", "B2")
        u, v = sg.loc[[1, 5]]
        p = state_of("paper", None)
        assert sg.lift_guards[u, v] == (1, 0)
        assert not any(how == "lift" for _, _, _, how in sg.step(p, u))
        assert any(w == v and how == "lift" for _, w, _, how in sg.step(p | 1, u))
        assert not np.isfinite(sg.togo(5)[p, 1]) and np.isfinite(sg.togo(5)[p | 1, 1])
        k = state_of("stone", None)
        assert any(after == k | 1 and how == "pendulum_box" for after, _, _, how in sg.step(k, sg.loc[2]))
        assert not any(how == "pendulum_box" for _, _, _, how in sg.step(p, sg.loc[2]))
        for name in ("B1", "B2"):
            other, st = build(cork=True, without=(name,))
            assert not other.lift_guards and not other.M
            assert all(c["name"] not in ("B1", "B2") for c in other.pendulums[0]["candidates"])
            assert any(how == "lift" for _, _, _, how in other.step(st("paper", None), other.loc[1]))
            locked = regions.scan(g, info, 1, world, overrides=regions.Overrides(lock=[name]))
            assert locked["lock"] == 1
            assert not any(how == "pendulum_box" for _, _, _, how in sg.step(k, sg.loc[2], lock=locked["lock"]))
        other, _ = build(cork=True, without=(rod,))
        assert not other.pendulums and other.lift_guards == sg.lift_guards
        plain, _ = build(cork=False)
        assert not plain.M and not plain.lift_guards
        res = regions.analyse(g, info, 1, world)
        key = next(k for k in res["links"] if k[0] == "mech" and k[3] == coarse.PENDULUM_BOX)
        lift = next(k for k in res["links"] if k[0] == "lift")
        assert not regions._moves(lift, 0) and not regions._moves(key, 0, lock=1)
        assert regions._moves(lift, regions._moves(key, 0)[0][1])
        assert regions.describe(key, lang="en").isascii() and "推开" in regions.describe(key, lang="zh")
        report = regions.mechanisms(res)["mechanisms"]
        box = next(m for m in report if m["type"] == "box")
        assert box["initial"] == "present" and box["lets_through"]["cleared"]
        assert any(r["pendulum"] == rod for r in box["switched"])
        assert regions._stack_note(box, lang="en").isascii() and "箱子" in regions._stack_note(box, lang="zh")
        assert box["switch_note"].isascii()


def _elevator_fixture(axis, *, chain=False):
    from ballistics import things
    cat = things.load()["P_Modul_03"]
    loaded = cat.numbers["full_load_eq"]["v"]
    rise = cat.numbers["per_block"]["v"]
    sink = cat.numbers["ball_sink"]["v"]
    tops = cat.numbers["all_off_top_by_ball"]["v"]
    # A straight approach, a same-height bank at the wrong wall, and several isolated exits.
    xyz = [(11, loaded + 2, 0), (0, loaded + 2, -11),
           (-11, loaded + rise - sink["wood"] + 2, 0),
           (0, loaded + 1.5 * rise - sink["wood"] + 2, 11),
           (0, max(tops.values()) + 2 + 3, -11),
           (-16, tops["wood"] + 2, 0)]
    hulls = [("Floor", _hull_box((-10, -9, -10), (10, -8, 10))),
             ("Gate", _hull_box((8, -8, -4), (9, -2, -2.5))),
             ("Gate", _hull_box((8, -8, 2.5), (9, -2, 4))),
             ("Gate", _hull_box((8, -2, -4), (9, -1, 4))),
             ("Wall01", _hull_box((-9, -8, -3), (-8, -1, 3))),
             ("Wall02", _hull_box((-3, -8, 8), (3, -1, 9))),
             ("Wall03", _hull_box((-3, -8, 6), (3, -1, 7))),
             ("Wall04", _hull_box((-3, -8, -9), (3, -1, -8)))]
    if chain:
        delta = np.array([-20.5, tops["wood"] - loaded, 0.])
        xyz = [xyz[0], np.array([-16, tops["wood"] + 2, 0]) + delta]
    info, g, world, build = _road_state_fixture("P_Modul_03", xyz, [],
                                                [("P_Modul_03_" + name, v) for name, v in hulls], axis)
    if chain:
        p = info["placements"][0]
        d = delta @ np.array([p[a + "_axis"] for a in "xyz"])
        info["placements"].append(dict(p, name="M2", position=p["position"] + d))
        info["hulls"] += [dict(h, placement="M2", vertices=h["vertices"] + d) for h in info["hulls"]]
    return info, g, world, build


def test_elevator_doorway_balls_wall_count_and_height_in_three_orientations():
    import json

    for axis in ROAD_AXES:
        info, g, world, build = _elevator_fixture(axis)
        sg, read = build(cork=True)
        assert sg.M == 1 and build(cork=False)[0].M == 0
        assert len(g.pos) == len(g._relaxed.fwd) == 6
        d = sg.mechs[0]
        assert d["entries"] == [0]
        assert not any(e[1] == 1 or e[2] == 4 for e in d["edges"])
        for k in (W, S, P):
            s = read(regions.KINDS[k], None)
            moves = sg.step(s, sg.loc[0])
            goal = 5 if k == P else 2  # paper is lighter, so its height matches a different exit
            assert any(v == sg.loc[goal] for _, v, _, _ in moves)
            assert np.isfinite(sg.togo(goal)[s, 0])
            assert all(st == s | 1 and how == "ride_elevator" for st, _, _, how in moves)
            assert not sg.step(s | 1, sg.loc[0]) and not sg.step(s, sg.loc[0], lock=1)
        assert {ride["blocks"] for (_, v, side), ride in d["rides"].items() if v == 3 and side & (1 << W)} == {2}
        assert all(len(ride["walls"]) == 2 for (_, v, _), ride in d["rides"].items() if v == 3)
        extra_wall = next(h for h in info["hulls"] if h["body"].endswith("Wall03"))
        info["hulls"].remove(extra_wall)
        one_wall = build(cork=True)[0].mechs[0]
        assert {ride["blocks"] for (_, v, side), ride in one_wall["rides"].items() if v == 3 and side & (1 << W)} == {1, 2}
        info["hulls"].append(extra_wall)
        assert np.isfinite(sg.togo(2)[read("wood", None), 0])
        assert not np.isfinite(sg.togo(2, lock=1)[read("wood", None), 0])
        assert build(cork=True, without=("M",))[0].M == 0
        res = regions.mechanisms(regions.analyse(g, info, 1, world))
        m = res["mechanisms"][0]
        assert res["breaks"] == 0 and m["on_way"] and m["without_breaks"] is None
        for key, link in res["links"].items():
            assert regions.describe(key, lang="en", ride=link.get("elevator")).isascii()
            assert "乘升降机" in regions.describe(key, lang="zh", ride=link.get("elevator"))
            if key[6] == P:
                assert all(tag in regions.describe(key) for tag in ("[WIKI]", "[OWNER]", "[ASSUMED]"))
                assert "助跑" in regions.describe(key, lang="zh")
        assert json.dumps(regions.summary_mechanisms(res), ensure_ascii=False).isascii()
        assert json.dumps(regions.summary_links(res), ensure_ascii=False).isascii()
        assert all(row.isascii() for row in regions._elevator_notes(m))
        assert any(r.startswith("P") for e in m["entries"] for r in e["regions"])
        assert any("[OWNER]" in row for row in regions._elevator_notes(m) if row.startswith("P") and " -> " in row)
        paper = regions.analyse(g, info, 1, world, entry_kind="paper")
        assert paper["breaks"] == 0 and paper["path"][-1][3] == "ride_elevator"
        assert any("要飞过" in row for row in regions._elevator_notes(regions._chinese(m), lang="zh"))
        for ov in (regions.Overrides(lock={"M"}), regions.Overrides(state={"M": "used"}), regions.Overrides(without={"M"})):
            assert regions.analyse(g, info, 1, world, overrides=ov)["breaks"] is None


def test_elevator_chain_must_land_on_a_loaded_platform():
    for axis in ROAD_AXES:
        info, g, world, build = _elevator_fixture(axis, chain=True)
        sg, read = build(cork=True)
        s = read("wood", None)
        assert np.isfinite(sg.togo(1)[s, 0])
        assert not np.isfinite(sg.togo(1)[s | 2, 0])
        assert not sg.step(s | 2, sg.loc[0])
        res = regions.analyse(g, info, 1, world)
        assert [p[3] for p in res["path"]] == ["start", "ride_elevator", "ride_elevator"]
        assert res["path"][-1][2] == 3
        for ov in (regions.Overrides(state={"M2": "used"}), regions.Overrides(without={"M2"})):
            assert regions.analyse(g, info, 1, world, overrides=ov)["breaks"] is None


def test_swing_joins_two_banks_for_all_balls_and_without_removes_only_the_ride():
    from ballistics.things import load

    nums = load()["P_Modul_08"].numbers
    x = max(nums["end_reach_record"]["v"])
    xyz = [(-x, 3.5, 0), (x, 3.5, 0)]
    hulls = [("P_Modul_08_Schaukel", _hull_box((-3, -0.4, -1.8), (3, 0.1, 1.8)))]
    for axis in ROAD_AXES:
        info, g, world, build = _road_state_fixture("P_Modul_08", xyz, [(0, 1, 18, "gap")], hulls, axis)
        sg, _ = build(cork=True)
        assert sg.M == 0
        for k in (W, S, P):
            for u, v in ((0, 1), (1, 0)):
                assert any((t, w, how) == (k, v, "ride_swing") for t, w, _, how in sg.step(k, u))
                assert np.isfinite(sg.togo(v)[k, u])
        res = regions.mechanisms(regions.analyse(g, info, 1, world))
        m, = res["mechanisms"]
        assert res["breaks"] == 0 and m["without_breaks"] == 1 and m["on_way"]
        assert m["states"] == [] and len(m["crossings"]) == 6
        assert regions._chinese(m["need"]) == "用到"
        assert m["timing"]["bank_window_steps_132"] == nums["bank_window_steps_132"]["v"]
        key = res["path"][-1][4]
        assert "keys held for wood and paper" in regions.describe(key, lang="en")
        assert regions.describe(key, lang="en").isascii() and "秋千" in regions.describe(key, lang="zh")
        assert all(row.isascii() for row in regions._timed_notes(m))
        assert not build(cork=True, without=["M"])[0].timed
        assert not build(cork=False)[0].timed


def test_sandbag_windows_require_fixed_support_and_removal_restores_normal_roll():
    from ballistics.geometry import Mesh

    def floor(gap):
        def mesh(A, o):
            strips = [(-6, -1), (1, 6)] if gap else [(-6, 6)]
            pts = np.array([[(a, 0, -2), (a, 0, 2), (b, 0, 2), (b, 0, -2)] for a, b in strips]) @ A + o
            v = pts[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3, 3)
            return Mesh(*v.transpose(1, 0, 2), np.zeros(len(v), int), np.zeros(len(v), bool), np.empty((0, 7)), [{}])
        return mesh

    xyz = [(-3.5, 2, 0), (3.5, 2, 0)]
    hulls = [("P_Modul_26_Sack", _hull_box((-1.4, -0.5, -1.9), (1.4, 3.6, 1.9)))]
    for axis in ROAD_AXES:
        info, g, world, build = _road_state_fixture("P_Modul_26", xyz, [], hulls, axis, mesh=floor(False))
        sg, _ = build(cork=True)
        removed, _ = build(cork=True, without=["M"])
        assert sg.M == removed.M == 0
        for k in (W, S, P):
            for u, v in ((0, 1), (1, 0)):
                assert [(t, w, h) for t, w, _, h in sg.step(k, u)] == [(k, v, "pass_sandbag")]
                assert np.isfinite(sg.togo(v)[k, u])
                assert [(t, w, h) for t, w, _, h in removed.step(k, u)] == [(k, v, "roll")]
        res = regions.mechanisms(regions.analyse(g, info, 1, world))
        m, = res["mechanisms"]
        assert m["states"] == [] and m["without_breaks"] == res["breaks"] == 0 and len(m["crossings"]) == 6
        assert all(row.isascii() for row in regions._timed_notes(m))
        key = res["path"][-1][4]
        assert regions.describe(key, lang="en").isascii() and "沙袋" in regions.describe(key, lang="zh")
        assert not build(cork=False)[0].step(W, 0)
        *_, unsupported = _road_state_fixture("P_Modul_26", xyz, [], hulls, axis, mesh=floor(True))
        assert not unsupported(cork=True)[0].step(W, 0)
        assert not unsupported(cork=True, without=["M"])[0].step(W, 0)


def test_sandbag_cuts_cached_rail_only_in_regions_and_only_at_the_cross_section():
    xyz = [(-3.5, 2, 0), (-1, 2, 0), (1, 2, 0), (3.5, 2, 0)]
    hulls = [("P_Modul_26_Sack", _hull_box((-1.4, -0.5, -1.9), (1.4, 3.6, 1.9)))]
    rails = [(u, v, abs(xyz[u][0] - xyz[v][0]), "rail") for i in range(3) for u, v in ((i, i + 1), (i + 1, i))]
    for axis in ROAD_AXES:
        _, g, _, build = _road_state_fixture("P_Modul_26", xyz, rails, hulls, axis)
        g.kind[:] = 1
        g.lane[:] = 0
        active, _ = build(cork=True)
        assert (W, 1, 2.5, "rail") in active.step(W, 0)
        assert (W, 2, 2, "rail") not in active.step(W, 1)
        assert any((t, v, how) == (W, 3, "pass_sandbag") and np.isclose(c, 7) for t, v, c, how in active.step(W, 0))
        for plain in (build(cork=True, without=["M"])[0], build(cork=False)[0]):
            assert (W, 2, 2, "rail") in plain.step(W, 1)
            assert not plain.timed
