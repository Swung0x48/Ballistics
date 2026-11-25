"""The coarse level: rest places, how they connect, and that it only proposes."""
import numpy as np

from ballistics import coarse
from ballistics.geometry import Mesh


def quad(x0, x1, z0, z1, y):
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z1 * 0 + z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]           # wound to face up


def box(x0, x1, y0, y1, z0, z1):
    c = [np.array((x, y, z), float) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    q = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    return [t for a, b, d, e in q for t in ((c[a], c[b], c[d]), (c[a], c[d], c[e]))]


def mesh(parts) -> Mesh:
    """parts: [(triangles, placed)]"""
    A, B, C, owner, ents = [], [], [], [], []
    for tris, placed in parts:
        ents.append(dict(name="part%d" % len(ents), placed=placed, type="P_Box" if placed else ""))
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    return Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)


def test_a_floor_is_one_connected_set_of_rest_places():
    g = coarse.build(mesh([(quad(0, 20, 0, 8, 0.0), False)]))
    a, b = g.nearest((2, 2, 4)), g.nearest((18, 2, 4))
    path, cost = g.route(a, b)
    assert path is not None and 15.5 <= cost <= 18.0 and all(g.edge_kind(u, v) == "roll" for u, v in zip(path, path[1:]))
    assert np.allclose(g.pos[path][:, 1], 2.0, atol=1e-6)            # ball centres, resting


def test_a_small_step_down_is_a_one_way_edge_and_a_real_fall_is_not_an_edge_of_positions():
    g = coarse.build(mesh([(quad(0, 10, 0, 8, 1.0), False), (quad(10, 20, 0, 8, 0.0), False)]))
    up, down = g.nearest((3, 3, 4)), g.nearest((17, 2, 4))
    path, _ = g.route(up, down)
    assert path is not None and "drop" in [g.edge_kind(u, v) for u, v in zip(path, path[1:])]
    assert g.route(down, up)[0] is None
    # three units down: where the ball lands depends on how fast it leaves, so positions alone give no edge
    g = coarse.build(mesh([(quad(0, 10, 0, 8, 3.0), False), (quad(10, 20, 0, 8, 0.0), False)]))
    assert g.route(g.nearest((3, 5, 4)), g.nearest((17, 2, 4)))[0] is None


def test_a_gap_is_crossed_by_a_closed_form_launch():
    from ballistics.takeoff import add_launches
    m = mesh([(quad(0, 12, 0, 8, 0.0), False), (quad(18, 30, 0, 8, -4.0), False)])
    g = coarse.build(m)
    a, b = g.nearest((2, 2, 4)), g.nearest((27, -2, 4))
    assert g.route(a, b)[0] is None                                  # no edge of positions gets across: a fall is not one
    assert add_launches(g, m) > 0
    path, _ = g.route(a, b)
    kinds = [g.edge_kind(u, v) for u, v in zip(path, path[1:])]
    assert kinds.count("launch") == 1
    u, v = next((u, v) for u, v in zip(path, path[1:]) if g.edge_kind(u, v) == "launch")
    e = g.info["launches"][(u, v)]
    assert e["runup"] > 0 or e["rim_ticks"] > 0
    assert e["way"][0, 0] <= 12 and e["landing_position"][0] >= 18 - 1e-8
    assert 2.0 < e["down"] < 5.0
    assert e["flight_ticks"] <= e["ticks"] and e["time"] == e["ticks"] / 132
    np.testing.assert_allclose(e["way"][-1, [0, 2]], g.pos[v, [0, 2]])


def test_a_placed_object_blocks_and_a_trigger_is_kept_out_of():
    road = quad(0, 30, 0, 5, 0.0)
    blocked = coarse.build(mesh([(road, False), (box(14, 16, 0, 5, 0, 5), True)]))          # a placed box right across the road
    assert blocked.route(blocked.nearest((2, 2, 2.5)), blocked.nearest((28, 2, 2.5)))[0] is None
    free = coarse.build(mesh([(road, False)]))
    assert free.route(free.nearest((2, 2, 2.5)), free.nearest((28, 2, 2.5)))[0] is not None
    # no transformer is kept out of: the way leads through it (`coarse.StateGraph` makes it cost the transformation,
    # and the ball that kind after it)
    for kind in ("P_Trafo_Stone", "P_Trafo_Paper"):
        assert coarse.keep_out([dict(type=kind, position=[15.0, 0.0, 2.5])]) == []


def test_the_coarse_level_goes_on_as_the_kind_a_transformer_makes():
    """Through a paper transformer the plan goes on as a paper ball; a lift is there for a paper ball only."""
    road = quad(0, 40, 0, 5, 0.0)
    g = coarse.build(mesh([(road, False)]))
    rel = coarse.Relaxed(g)
    N = len(g.pos)
    a, b, t = g.nearest((2, 2, 2.5)), g.nearest((38, 2, 2.5)), g.nearest((20, 2, 2.5))
    tk = np.full(N, -1)
    tk[np.linalg.norm(g.pos - g.pos[t], axis=1) < 4.3] = coarse.PAPER
    sg = coarse.StateGraph(rel, g, np.ones(N, bool), tk, 7.0, [(b, a, 1.0)], [], 0.0)
    togo = sg.togo(a)
    nodes, kinds, states = sg.plan(coarse.WOOD, b, togo)          # from b to a, as a wood ball: through the transformer
    assert nodes[-1] == a and states[0] == coarse.WOOD and states[-1] == coarse.PAPER
    first_paper = states.index(coarse.PAPER)
    assert all(i + 1 > first_paper for i, k in enumerate(kinds) if k == "lift")   # a lift only once it is a paper ball
    nodes, kinds, states = sg.plan(coarse.PAPER, b, togo)        # as a paper ball: the lift goes straight there
    assert kinds == ["lift"] and nodes[-1] == a
    assert togo[coarse.PAPER, b] == 1.0 and togo[coarse.WOOD, b] > 7.0


def test_the_coarse_plan_alone_lists_its_links_and_draws_with_its_keys_outside_the_map(tmp_path):
    from PIL import Image

    from ballistics import solve
    from ballistics.render import SURFACE, TopView, draw_coarse

    def info(goal):
        return dict(start=[4.0, 2.0, 4.0], checkpoints=[], airship=list(goal), reset_points=[[4.0, 3.0, 4.0]], placements=[], death_boxes=[])

    m = mesh([(quad(0, 40, 0, 8, 0.0), False)])
    plans = solve.coarse_plans(m, info((34.0, 2.0, 4.0)), coarse.build(m))
    assert len(plans) == 1 and plans[0]["ok"] and not plans[0]["links"] and set(plans[0]["kinds"]) == {"roll"} and 28.0 < plans[0]["length"] < 36.0
    # a gap, and no launch edges on the graph: the plan has to be joined by a link, and says where and how wide
    m = mesh([(quad(0, 30, 0, 8, 0.0), False), (quad(37, 70, 0, 8, -1.0), False)])
    g = coarse.build(m)
    plan = solve.coarse_plans(m, info((62.0, 1.0, 4.0)), g)[0]
    assert plan["ok"] and len(plan["links"]) == 1 and plan["links"][0]["across"] > 4.0 and plan["links"][0]["rise"] < 0
    assert 26.0 < plan["links"][0]["near"][0] < 31.0 and plan["links"][0]["placed_near"] == []
    # a goal nowhere near any floor: no plan, with the reason
    none = solve.coarse_plans(m, info((500.0, 1.0, 4.0)), g)[0]
    assert not none["ok"] and "goal" in none["reason"]
    # a second road 60 away from the first: links do not reach that far, so the plan cannot begin at the reset point
    far = mesh([(quad(0, 30, 0, 8, 0.0), False), (quad(90, 130, 0, 8, 0.0), False)])
    cut = solve.coarse_plans(far, info((122.0, 2.0, 4.0)), coarse.build(far))[0]
    assert cut["ok"] and cut["starts_off"] > 50.0 and not cut["links"]
    assert "starts_off" not in plan
    # the picture: header and footer are bands of their own, nothing of the map in them
    path = tmp_path / "plan.png"
    draw_coarse(m, g, [plan], path, "plan", width=900)
    img = Image.open(path)
    v = TopView(m, width=900, margin=8.0, height=True)
    assert img.size == v.img.size and v.foot >= 90
    left = [img.getpixel((4, y)) for y in range(img.size[1] - v.foot, img.size[1])]
    assert all(c == SURFACE for c in left)                       # the footer's left end is empty; the scale is at its right
    scale = {img.getpixel((x, v.foot_y() + 8)) for x in range(img.size[0] - 470, img.size[0] - 70)}
    assert len(scale) > 5 and all(r == gr == b for r, gr, b in scale)          # the height scale: greys


def test_a_low_step_is_proposed_upwards_only_and_a_high_one_is_not():
    def stair(h):
        riser = [(np.array((10.0, 0.0, 0.0)), np.array((10.0, h, 0.0)), np.array((10.0, h, 8.0))), (np.array((10.0, 0.0, 0.0)), np.array((10.0, h, 8.0)), np.array((10.0, 0.0, 8.0)))]
        return mesh([(quad(0, 10, 0, 8, 0.0) + riser, False), (quad(10, 20, 0, 8, h), False)])
    m = stair(1.2)
    g = coarse.build(m)
    low, high = g.nearest((3, 2, 4)), g.nearest((17, 3.2, 4))
    assert g.route(low, high)[0] is None                       # rolling does not get up it
    assert coarse.add_steps(g, m) > 0
    path, _ = g.route(low, high)
    assert [g.edge_kind(u, v) for u, v in zip(path, path[1:])].count("step") == 1
    back, _ = g.route(high, low)
    assert back is not None and "step" not in [g.edge_kind(u, v) for u, v in zip(back, back[1:])]      # down is a drop
    m = stair(coarse.STEP_UP + 0.4)
    g = coarse.build(m)
    coarse.add_steps(g, m)
    assert g.route(g.nearest((3, 2, 4)), g.nearest((17, coarse.STEP_UP + 2.4, 4)))[0] is None


class _Rel:
    """A relaxed graph of nothing but these edges."""
    def __init__(self, n, edges):
        self.fwd = [[] for _ in range(n)]
        for u, v, c in edges:
            self.fwd[u].append((v, c, "roll"))
            self.fwd[v].append((u, c, "roll"))


def _line(n):
    return coarse.Graph(np.array([[float(i), 0.0, 0.0] for i in range(n)]), np.zeros(n, int), np.full(n, -1), [[] for _ in range(n)], [])


def test_the_coarse_level_keeps_a_push_board_down():
    """0 is a wood transformer's range, 2 is in front of a push board, 3 beyond it, 4 the goal. A paper ball at 1
    has to become a wood ball to push it over; once it lies, a paper ball goes across."""
    g = _line(5)
    rel = _Rel(5, [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)])
    tk = np.array([coarse.WOOD, -1, -1, -1, -1])
    board = dict(name="B", type="board", edges=[(coarse.BOARD_PUSH, 2, 3, 2.0, 0, "roll"), (coarse.BOARD_OVER, 2, 3, 2.0, 0, "roll"), (coarse.BOARD_OVER, 3, 2, 2.0, 0, "roll")])
    sg = coarse.StateGraph(rel, g, np.ones(5, bool), tk, 10.0, [], [board], 0.0)
    togo = sg.togo(4)
    paper_up, paper_down = coarse.PAPER * 2 + 0, coarse.PAPER * 2 + 1
    nodes, kinds, states = sg.plan(paper_up, 1, togo)
    assert nodes == [1, 0, 1, 2, 3, 4]                           # back to become wood, then over
    assert states[-1] >> 1 == coarse.WOOD and states[-1] & 1 == 1 # it lies now
    assert togo[paper_down, 2] == 2.0 + 1.0                       # lying, a paper ball goes straight across
    tk2 = np.full(5, -1)
    sg2 = coarse.StateGraph(rel, g, np.ones(5, bool), tk2, 10.0, [], [board], 0.0)
    assert not np.isfinite(sg2.togo(4)[paper_up, 1])             # no way to become a ball that can push it


def test_the_coarse_level_turns_a_seesaw_over():
    """A seesaw between 1 and 2, low at 1. Any ball goes up from 1 to 2; a stone ball tips it, and then it is low
    at 2: from there back to 1."""
    g = _line(4)
    rel = _Rel(4, [(0, 1, 1.0), (2, 3, 1.0)])
    seesaw = dict(name="W", type="seesaw", edges=[(coarse.SEESAW, 1, 2, 3.0, 0, "roll"), (coarse.SEESAW, 2, 1, 3.0, 1, "roll")])
    sg = coarse.StateGraph(rel, g, np.ones(4, bool), np.full(4, -1), 10.0, [], [seesaw], 0.0)
    to3, to0 = sg.togo(3), sg.togo(0)
    wood_low1, stone_low1 = coarse.WOOD * 2 + 0, coarse.STONE * 2 + 0
    assert np.isfinite(to3[wood_low1, 0]) and np.isfinite(to3[stone_low1, 0])
    nodes, kinds, states = sg.plan(stone_low1, 0, to3)
    assert nodes == [0, 1, 2, 3] and states[-1] & 1 == 1          # tipped: low at 2 now
    assert np.isfinite(to0[coarse.STONE * 2 + 1, 3])               # and back the other way
    assert not np.isfinite(to0[wood_low1, 3])                      # not while it is low at 1


def test_a_link_through_a_transformer_or_a_standing_board_counts():
    """0 - 1 - 2 by road, 2 - 3 a link (a break) that passes through a paper transformer's range on the way and runs
    into a push board while it stands; 3 is the goal. A wood ball across the link arrives as a paper ball, and only
    once the board is down."""
    g = _line(4)
    rel = _Rel(4, [(0, 1, 1.0), (1, 2, 1.0)])
    rel.fwd[2].append((3, 50.0, "bridge"))
    tk = np.full(4, -1)
    board = dict(name="B", type="board", edges=[(coarse.BOARD_PUSH, 0, 1, 1.0, 0, "roll")])
    links = {(2, 3): (coarse.PAPER, 1)}
    sg = coarse.StateGraph(rel, g, np.ones(4, bool), tk, 10.0, [], [board], 0.0, links=links)
    togo = sg.togo(3)
    up, down = coarse.WOOD * 2, coarse.WOOD * 2 + 1
    assert togo[down, 2] == 50.0 + 10.0                           # down: across, and a paper ball on the far side
    assert togo[coarse.PAPER * 2 + 1, 2] == 50.0                  # (a paper ball already: no transformation)
    assert togo[up, 2] == 2.0 + 2.0 + 50.0 + 10.0                 # standing: back to push it down first
    nodes, kinds, states = sg.plan(up, 0, togo)
    assert nodes == [0, 1, 2, 3] and states[-1] == coarse.PAPER * 2 + 1
    assert togo[up, 0] == 1.0 + 1.0 + 50.0 + 10.0                 # the plan's cost is what togo says
    rel2 = _Rel(4, [(0, 1, 1.0), (1, 2, 1.0)])
    rel2.fwd[2].append((3, 50.0, "bridge"))
    sg2 = coarse.StateGraph(rel2, g, np.ones(4, bool), tk, 10.0, [], [], 0.0, links={(2, 3): (-1, 1)})
    assert not np.isfinite(sg2.togo(3)[0, 2])                     # a board that cannot be pushed down: no way


def test_a_lift_that_lands_in_a_transformers_range_lands_as_its_kind():
    """A paper ball's lift from 0 up to 1, inside a stone transformer's range; the goal 2 is on by road. The lift lands
    a stone ball, at the transformation's cost."""
    g = _line(3)
    rel = _Rel(3, [(1, 2, 1.0)])
    tk = np.array([-1, coarse.STONE, -1])
    sg = coarse.StateGraph(rel, g, np.ones(3, bool), tk, 10.0, [(0, 1, 5.0)], [], 0.0)
    togo = sg.togo(2)
    assert togo[coarse.PAPER, 0] == 5.0 + 10.0 + 1.0
    nodes, kinds, states = sg.plan(coarse.PAPER, 0, togo)
    assert nodes == [0, 1, 2] and kinds[0] == "lift" and states[1] == coarse.STONE


def test_a_ball_of_another_kind_inside_a_range_pays_the_transformation_both_ways():
    """1 and 2 are inside a stone transformer's range; a wood ball at 1 (put there by a mechanism, say) is
    transformed: forward and backward agree that going on costs the transformation."""
    g = _line(3)
    rel = _Rel(3, [(0, 1, 1.0), (1, 2, 1.0)])
    tk = np.array([-1, coarse.STONE, coarse.STONE])
    sg = coarse.StateGraph(rel, g, np.ones(3, bool), tk, 10.0, [], [], 0.0)
    togo = sg.togo(2)
    assert togo[coarse.WOOD, 1] == 1.0 + 10.0 and togo[coarse.STONE, 1] == 1.0
    nodes, kinds, states = sg.plan(coarse.WOOD, 1, togo)
    assert nodes == [1, 2] and states == [coarse.WOOD, coarse.STONE]
    assert min(e[2] for e in sg.step(coarse.WOOD, 1) if e[1] == 2) == 11.0


def test_dropping_an_edge_takes_only_that_kind_and_a_mechanisms_too():
    g = _line(3)
    rel = _Rel(3, [(1, 2, 10.0)])
    rel.fwd[0] += [(1, 1.0, "step"), (1, 2.0, "launch")]
    gate = dict(name="G", type="gate", edges=[(coarse.GATE_THROUGH, 0, 2, 1.0, 0, "step")])
    sg = coarse.StateGraph(rel, g, np.ones(3, bool), np.full(3, -1), 10.0, [], [gate], 0.0)
    open_ = coarse.WOOD * 2 + 1
    assert sg.togo(2)[open_, 0] == 1.0
    sg.drop(0, 1, "step")
    sg.drop(0, 2, "step")
    togo = sg.togo(2)
    assert togo[open_, 0] == 2.0 + 10.0                          # the launch is still there; the gate's step edge is not
    assert min(e[2] + togo[e[0], e[1]] for e in sg.step(open_, 0)) == togo[open_, 0]


def test_a_lift_that_ends_in_a_range_on_the_way_ends_as_its_kind():
    g = _line(3)
    rel = _Rel(3, [(1, 2, 1.0)])
    sg = coarse.StateGraph(rel, g, np.ones(3, bool), np.full(3, -1), 10.0, [(0, 1, 5.0, coarse.STONE)], [], 0.0)
    togo = sg.togo(2)
    assert togo[coarse.PAPER, 0] == 5.0 + 10.0 + 1.0
    nodes, kinds, states = sg.plan(coarse.PAPER, 0, togo)
    assert nodes == [0, 1, 2] and states[1] == coarse.STONE


def test_a_wood_ball_pushes_a_gate_open_only_where_the_region_analysis_lets_it():
    # place 0 is where the gate is pushed from, 0 -> 2 only once it is open; side GATE_WOOD lets a wood ball push it too
    g = _line(3)
    for side, wood in ((0, False), (coarse.GATE_WOOD, True)):
        gate = dict(name="G", type="gate", edges=[(coarse.GATE_THROUGH, 0, 2, 1.0, 0, "roll"), (coarse.GATE_OPERATE, 0, 0, 0.0, side, "operate")])
        sg = coarse.StateGraph(_Rel(3, []), g, np.ones(3, bool), np.full(3, -1), 10.0, [], [gate], 0.0)
        togo = sg.togo(2)
        closed = {k: k * 2 for k in (coarse.WOOD, coarse.STONE, coarse.PAPER)}
        assert np.isfinite(togo[closed[coarse.STONE], 0])                             # a stone ball always
        assert np.isfinite(togo[closed[coarse.WOOD], 0]) == wood                      # a wood ball only with GATE_WOOD
        assert not np.isfinite(togo[closed[coarse.PAPER], 0])                         # a paper ball never
        assert any(e[0] == closed[coarse.WOOD] + 1 for e in sg.step(closed[coarse.WOOD], 0)) == wood
