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
