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
    assert regions.describe(into[0]) == "滚进纸变的范围，变纸球"
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
    assert m["states"] == ["关", "开"] and m["initial"] == "关" and m["need"] == "必需" and m["held_breaks"] == 1
    assert m["lets_through"]["开"] == [{"from": R.name(0), "to": R.name(1), "balls": "石"}]
    (c,) = res["chain"]
    assert c["step"] == 1 and c["for"]["step"] == 2 and "开" in c["did"]


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
        assert rock["need"] == "必需" and rock["lets_through"]["未落"] and rock["lets_through"]["已落"]
        assert all(row["balls"] == "石" and row["at"] for row in rock["switched"])
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
        assert board["states"] == ["立", "倒"] and board["need"] == "必需"
        assert "助跑约 12" in board["switch_note"] and board["lets_through"]["倒"]
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
            assert row["name"] == "P_Modul_41_01–%02d" % n
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
                    assert regions.describe(res["path"][-1][-1]) == "%s球过箱式浮板（%d 个一列）" % (regions.BALL[k], n)
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
            assert {(e["box"], e["direction"], e["balls"]) for e in m["exits"]} == {(3, "末端", "木"), (3, "末端", "纸")}
            assert any("从箱式浮板第 3 个（末端）落下" in regions.describe(key)
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
