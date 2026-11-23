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
