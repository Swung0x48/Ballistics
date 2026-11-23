"""The coarse level as regions (ballistics.regions): which places roll to each other, and the way through the regions
with the fewest breaks. Made-up graphs only."""
from ballistics import coarse, regions

W, S, P = coarse.WOOD, coarse.STONE, coarse.PAPER


def test_a_region_is_what_rolls_both_ways():
    # 0 <-> 1 <-> 2 roll both ways; 2 -> 3 only down; 3 <-> 4
    adj = [[1], [0, 2], [1, 3], [4], [3]]
    lab = regions.components(5, adj)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3]


def test_the_way_with_the_fewest_breaks_goes_for_the_stone_ball_and_opens_the_gate():
    # region 0: the start, with a stone transformer's range; the gate's push place is in it; through the open gate to 1
    # (the goal). A link (a break) also leads from 0 to 1
    gate = ("mech", 0, 1, coarse.GATE_THROUGH, 0, 0)
    push = ("mech", 0, 0, coarse.GATE_OPERATE, 0, 0)
    gap = ("plain", 0, 1, "gap", 0, -1, -1)
    links = {k: dict(cost=1.0, count=1, at=(0, 0)) for k in (gate, push, gap)}
    breaks, path, free = regions.connect(links, {0: {S}}, 1, (0, W, 0), 1)
    assert breaks == 0
    assert [(r, k, b, what) for r, k, b, what, _key in path] == [(0, W, 0, "start"), (0, S, 0, "transform"), (0, S, 1, "push_gate_open"), (1, S, 1, "through_gate")]
    assert (1, S, 1) in free
    # without the transformer only the break leads there: a wood ball cannot open the gate
    breaks, path, free = regions.connect(links, {}, 1, (0, W, 0), 1)
    assert breaks == 1 and [what for *_x, what, _k in path] == ["start", "gap"] and not any(n[0] == 1 for n in free)


def test_a_lift_is_a_paper_balls_and_a_board_is_not_pushed_by_one():
    lift = ("lift", 0, 1, -1, -1)
    board = ("mech", 0, 2, coarse.BOARD_PUSH, 0, 0)
    links = {k: dict(cost=1.0, count=1, at=(0, 0)) for k in (lift, board)}
    assert regions.connect(links, {}, 1, (0, W, 0), 1)[0] is None             # wood: no lift
    assert regions.connect(links, {}, 1, (0, P, 0), 1)[0] == 0
    assert regions.connect(links, {}, 1, (0, P, 0), 2)[0] is None             # paper: no push
    b, path, _f = regions.connect(links, {}, 1, (0, S, 0), 2)
    assert b == 0 and path[-1][:3] == (2, S, 1)                               # stone: pushed over, the board's bit set


def test_a_stone_ball_tips_the_seesaw_and_comes_back_over_it_only_from_the_new_low_end():
    # region 0 and 1 at the two ends; side 0: from 0 while bit 0 is 0; side 1: from 1 while bit 0 is 1
    there = ("mech", 0, 1, coarse.SEESAW, 0, 0)
    back = ("mech", 1, 0, coarse.SEESAW, 0, 1)
    links = {k: dict(cost=1.0, count=1, at=(0, 0)) for k in (there, back)}
    # a wood ball goes over and cannot come back (the low end is still at 0)
    assert regions.connect(links, {}, 1, (0, W, 0), 1)[0] == 0
    assert regions.connect(links, {}, 1, (1, W, 0), 0)[0] is None
    # a stone ball tips it on the way: then it can come back
    _b, path, free = regions.connect(links, {}, 1, (0, S, 0), 1)
    assert path[-1][:3] == (1, S, 1) and (0, S, 0) in free
