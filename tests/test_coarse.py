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


def test_a_gap_is_crossed_by_a_launch_made_with_the_kernel():
    from ballistics.kernel import World
    from ballistics.launch import add_launches
    m = mesh([(quad(0, 12, 0, 8, 0.0), False), (quad(18, 30, 0, 8, -4.0), False)])
    g = coarse.build(m)
    a, b = g.nearest((2, 2, 4)), g.nearest((27, -2, 4))
    assert g.route(a, b)[0] is None                                  # no edge of positions gets across: a fall is not one
    assert add_launches(g, World.of(m)) > 0
    path, _ = g.route(a, b)
    kinds = [g.edge_kind(u, v) for u, v in zip(path, path[1:])]
    assert kinds.count("launch") == 1
    u, v = next((u, v) for u, v in zip(path, path[1:]) if g.edge_kind(u, v) == "launch")
    e = g.info["launches"][(u, v)]
    # the edge is a run of the kernel from rest: it knows its keys, how fast the ball left and how fast it landed
    assert e["runup"] > 0 and e["launch_speed"] > 6.0 and e["across"] > 5.0 and 3.0 < e["down"] < 5.0
    assert sum(n for _, n in e["tape"]) == e["ticks"] and g.pos[v][0] > 18.0


def test_a_placed_object_blocks_and_a_trigger_is_kept_out_of():
    road = quad(0, 30, 0, 5, 0.0)
    blocked = coarse.build(mesh([(road, False), (box(14, 16, 0, 5, 0, 5), True)]))          # a placed box right across the road
    assert blocked.route(blocked.nearest((2, 2, 2.5)), blocked.nearest((28, 2, 2.5)))[0] is None
    free = coarse.build(mesh([(road, False)]))
    assert free.route(free.nearest((2, 2, 2.5)), free.nearest((28, 2, 2.5)))[0] is not None
    # a paper transformer in the middle of a road as wide as its range: no way past (the paper ball is not solved)
    kept = coarse.build(mesh([(road, False)]), avoid=coarse.keep_out([dict(type="P_Trafo_Paper", position=[15.0, 0.0, 2.5])]))
    assert kept.route(kept.nearest((2, 2, 2.5)), kept.nearest((28, 2, 2.5)))[0] is None
    # a stone transformer is not kept out of: the way leads through it (`solve.graph_for` makes it cost the transformation)
    assert coarse.keep_out([dict(type="P_Trafo_Stone", position=[15.0, 0.0, 2.5])]) == []


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
