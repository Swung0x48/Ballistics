"""Transformers (ballistics.trafo, the kernel's trigger, the follower): the rule on made-up maps. The comparison
with the original engine's records needs game data and is not in the repository."""
import numpy as np

from ballistics import physics as ph, staged, things, trafo
from ballistics.geometry import Mesh
from ballistics.kernel import OK, TRIGGER, World
from ballistics.solve import world_for


def place(name, kind, x, sector=1, y_axis=(0.0, 1.0, 0.0)):
    return dict(name=name, type="P_Trafo_" + kind, sector=sector, position=[x, 0.0, 0.0], x_axis=[1.0, 0.0, 0.0], y_axis=list(y_axis), z_axis=[0.0, 0.0, 1.0])


INFO = dict(placements=[place("S1", "Stone", 0.0), place("W1", "Wood", 6.0), place("P1", "Paper", 60.0), place("S2", "Stone", 100.0, sector=2)], death_boxes=[])


def floor(y=-2.0, size=400.0):
    A = np.array([[-size, y, -size], [size, y, size]])
    B = np.array([[size, y, -size], [-size, y, size]])
    C = np.array([[-size, y, size], [size, y, -size]])
    return Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="floor", groups=[], friction=0.7, elasticity=0.3)], {})


def test_numbers_are_the_catalogues():
    R = trafo.Rules.load()
    n = things.load().rules["transform_timing"]["numbers"]
    assert R.radius == n["radius"]["v"] == 4.3 and R.hover == n["hover_height"]["v"] == 3
    assert (R.gone_after, R.body_after, R.control_after) == (2, 332, 333)


def test_which_are_in_place():
    assert [t["name"] for t in trafo.of(INFO)] == ["S1", "W1", "P1", "S2"]
    assert [t["name"] for t in trafo.of(INFO, 2)] == ["S2"]
    assert [t["kind"] for t in trafo.of(INFO, 1)] == ["stone", "wood", "paper"]


def test_the_nearest_one_of_another_kind_within_the_distance():
    items = trafo.of(INFO, 1)
    assert trafo.triggered([-4.29, 0, 0], "stone", items) is None and trafo.triggered([0, 4.29, 0], "wood", items)["name"] == "S1"
    assert trafo.triggered([0, 4.3, 0], "wood", items) is None                     # strictly nearer
    assert trafo.triggered([2.9, 0, 0], "wood", items)["name"] == "S1"             # S1 at 2.9, W1 at 3.1
    assert trafo.triggered([3.1, 0, 0], "wood", items) is None                     # the nearest is of its own kind: it shields S1, 3.1 away
    assert trafo.triggered([3.1, 0, 0], "stone", items)["name"] == "W1"
    assert trafo.triggered([30, 0, 0], "wood", items) is None and trafo.triggered([0, 0, 0], "wood", []) is None


def test_the_new_ball_is_at_rest_along_the_transformers_own_axis():
    tilted = trafo.of(dict(placements=[place("T", "Stone", 0.0, y_axis=(0.6, 0.8, 0.0))]))[0]
    pos, vel, omega = trafo.new_state(tilted)
    assert np.allclose(pos, [1.8, 2.4, 0.0]) and not vel.any() and not omega.any()


def test_the_kernel_ends_a_rollout_by_the_same_rule():
    m, P = floor(), ph.Params()
    w = World.of(m, trafos=trafo.of(INFO, 1))
    roll = lambda ball, x0, vx, n=200: w.rollout(ball, P, [x0, 0.011, 0.0], [vx, 0.0, 0.0], [0.0, 0.0, -vx / 2.0], np.zeros((1, 3)), [n])
    st, done, pos, vel, om, stats, _ = roll(ph.WOOD, -12.0, 6.6)                    # towards S1 from the far side of W1
    assert st == TRIGGER and w.trafos[int(stats[9])]["name"] == "S1" and np.linalg.norm(pos) < 4.3
    before = pos - vel * ph.DT
    assert np.linalg.norm(before) >= 4.3 - 0.05                                    # it ended at the first tick inside
    st, done, pos, *_ = roll(ph.STONE, -12.0, 6.6, 400)                            # a stone ball rolls through S1 and is taken by W1
    assert st == TRIGGER and abs(pos[0] - 6.0) < 4.3 and pos[0] > 1.0
    st, done, pos, vel, om, stats, _ = w.rollout(ph.WOOD, P, [14.0, 0.011, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.array([[-1.0, 0.0, 0.0]]), [600])   # from W1's side, key held: W1 shields S1 until S1 is the nearer
    assert st == TRIGGER and w.trafos[int(stats[9])]["name"] == "S1" and pos[0] < 3.0 + 0.2
    legacy = World.of(m, keep_out=[(0.0, 0.0, 0.0, 4.3)])                          # a plain keep-out sphere ends any ball
    assert legacy.rollout(ph.STONE, P, [-12.0, 0.011, 0.0], [6.6, 0.0, 0.0], [0.0, 0.0, -3.3], np.zeros((1, 3)), [200])[0] == TRIGGER
    assert len(world_for(m, INFO, "strict", 2).trafos) == 1 and len(world_for(m, INFO, "strict").trafos) == 4


def test_the_follower_stops_at_a_transformer_and_says_which():
    m = floor()
    w = World.of(m, trafos=trafo.of(INFO, 1))
    way = np.array([[x, 0.011, 0.0] for x in np.arange(-20.0, 30.0, 1.0)])
    r = staged.follow(w, w, (way[0], np.zeros(3), np.zeros(3)), way)
    assert r["status"] == "transform" and w.trafos[r["which"]]["name"] == "S1"
    assert np.linalg.norm(r["end"][0]) < 4.3 and sum(n for _a, n in r["tape"]) == len(r["trace"])
    r = staged.follow(w, w, (way[0], np.zeros(3), np.zeros(3)), way, ball=ph.STONE)   # as stone it rolls through S1 and W1 takes it
    assert r["status"] == "transform" and w.trafos[r["which"]]["name"] == "W1"


def test_a_sector_goes_on_as_the_new_ball(tmp_path):
    from test_staged import level, quad
    from ballistics import route, shortcut
    m, info, g = level([quad(0, 80, 0, 8, 0.0)], (74.0, 2.0, 4.0))                  # a plain road; the graph does not keep out of the transformer
    info["placements"] = [dict(name="S", type="P_Trafo_Stone", sector=1, position=[30.0, 0.5, 4.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0]),
                          dict(name="W", type="P_Trafo_Wood", sector=1, position=[55.0, 0.5, 4.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0])]
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
    assert res["solved"] and res["replays"] and (res["ball"], res["end_ball"]) == ("wood", "wood")
    assert [(t["at"], t["was"], t["to"]) for t in res["transforms"]] == [("S", "wood", "stone"), ("W", "stone", "wood")]
    kinds = [(sg["type"], sg["ball"]) for sg in res["segments"]]
    assert kinds[:5] == [("follow", "wood"), ("transform", "stone"), ("follow", "stone"), ("transform", "wood"), ("follow", "wood")]
    assert all(k == ("follow", "wood") or k == ("cross", "wood") for k in kinds[5:])       # what comes after is the wood ball's
    R = trafo.rules()
    t1 = res["segments"][1]
    assert len(t1["trace"]) == R.body_after + 1 and np.allclose(t1["trace"][0, :3], [30.0, 3.5, 4.0])     # the wait, then one tick of falling
    assert np.linalg.norm(res["segments"][0]["end"][0] - np.array([30.0, 0.5, 4.0])) < R.radius
    assert res["ticks"] == sum(len(sg["trace"]) for sg in res["segments"])
    assert shortcut.alternatives(m, info, g, 1, res, log=lambda *a: None) == []          # a route that changes its ball: no other routes
    rows = shortcut.listing("made up", [res])["sectors"][0]["routes"][0]["segments"]
    assert [r["type"] for r in rows][:5] == ["follower", "transformation", "follower", "transformation", "follower"] and rows[2]["ball"] == "stone"
    P, sec, t_ms, meta = route.reference_line([res], info, 1)
    assert meta["complete"] and abs(meta["game_ms"] / 1000 - res["ticks"] / 132) < 0.5
    said = []
    staged.summary(res, log=said.append)
    assert sum("transformed" in line for line in said) == 2
    from ballistics.render import draw_staged
    draw_staged(m, res, tmp_path / "two-transformers.png")
    assert (tmp_path / "two-transformers.png").stat().st_size > 0
    stone = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None, ball="stone")          # entered as stone: S does nothing, W transforms
    assert stone["solved"] and [(t["at"], t["to"]) for t in stone["transforms"]] == [("W", "wood")]


def test_the_balls_are_the_catalogues():
    cat = things.load()
    for name, ball in (("ball_wood", ph.WOOD), ("ball_stone", ph.STONE), ("ball_paper", ph.PAPER)):
        b = cat[name].bodies[0]
        assert (ball.mass, ball.friction, ball.elasticity, ball.lin_damp, ball.rot_damp) == tuple(
            float(b[k]["v"]) for k in ("mass", "friction", "elasticity", "lin_damp", "rot_damp")), name
    assert ph.WOOD.force == cat["ball_wood"].numbers["push_impulse"]["v"]
    assert abs(ph.STONE.force / ph.STONE.mass - cat["ball_stone"].numbers["push_dv_per_step"]["v"]) < 1e-9
    assert abs(ph.PAPER.force / ph.PAPER.mass - cat["ball_paper"].numbers["push_dv_per_step"]["v"]) < 1e-9


def test_the_search_goes_through_a_transformer_as_the_new_ball():
    from ballistics import fine
    m, P = floor(), ph.Params()
    info = dict(placements=[place("S1", "Stone", 0.0)], death_boxes=[])
    w = World.of(m, trafos=trafo.of(info))
    targets = np.array([[x, 0.011, z] for x in (9.0, 10.0, 11.0) for z in (-1.0, 0.0, 1.0)])
    start = ph.State(np.array([-9.0, 0.011, 0.0]), np.zeros(3), np.zeros(3))
    # a wall of range between the start and the targets is made by taking the far targets only reachable through it:
    # the straight way leads through S1; with transformations on, the search arrives as a stone ball
    s = fine.Search(w, targets, seed=1, transforms=True)
    s.add_start(start)
    c = s.run(max_expansions=40_000, max_seconds=120.0)
    assert c.found
    kinds = [(p["type"], p["ball"]) for p in c.parts]
    if len(c.parts) > 1:                                   # it went through (it may also have found a way around on the open floor)
        assert kinds == [("cross", "wood"), ("transform", "stone"), ("cross", "stone")] and c.end_ball == "stone"
        assert c.ticks == sum(len(p["trace"]) for p in c.parts) == len(c.trace) and c.tape == []
        assert np.linalg.norm(c.parts[0]["end"][0]) < 4.3 and np.allclose(c.parts[1]["trace"][0, :3], [0.0, 3.0, 0.0])
        assert staged._robust(w, s, (start.pos, start.vel, start.omega), c.tape, parts=c.parts) >= 0
    # a corridor as narrow as the range leaves no way around: found with transformations on, not without
    A = np.array([[-40.0, -2.0, -3.0], [40.0, -2.0, 3.0]]); B = np.array([[40.0, -2.0, -3.0], [-40.0, -2.0, 3.0]]); C = np.array([[-40.0, -2.0, 3.0], [40.0, -2.0, -3.0]])
    road = Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="road", groups=[], friction=0.7, elasticity=0.3)], {})
    w2 = World.of(road, trafos=trafo.of(info))
    on = fine.Search(w2, targets[[1, 4, 7]], seed=1, transforms=True)
    on.add_start(start)
    c2 = on.run(max_expansions=40_000, max_seconds=120.0)
    assert c2.found and [(p["type"], p["ball"]) for p in c2.parts] == [("cross", "wood"), ("transform", "stone"), ("cross", "stone")]
    off = fine.Search(w2, targets[[1, 4, 7]], seed=1)
    off.add_start(start)
    assert not off.run(max_expansions=20_000, max_seconds=120.0).found


def test_the_coarse_level_goes_through_a_stone_transformer_at_a_cost():
    from test_staged import level, quad
    from ballistics import solve
    from ballistics.launch import FOLLOW_SPEED
    m, info, _g = level([quad(0, 80, 0, 8, 0.0)], (74.0, 2.0, 4.0), launches=False)
    info["placements"] = [dict(name="S", type="P_Trafo_Stone", sector=1, position=[30.0, 0.5, 4.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0]),
                          dict(name="P", type="P_Trafo_Paper", sector=1, position=[60.0, 0.5, -30.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0])]
    g = solve.graph_for(m, info, log=lambda *a: None)
    inside = np.linalg.norm(g.pos - np.array([30.0, 0.5, 4.0]), axis=1) < 4.3
    assert inside.any()                                                    # there are places in its range
    extra = (trafo.rules().body_after + 1) / 66.0 * FOLLOW_SPEED
    # the coarse level's state graph: a wood ball going into its range becomes a stone one, at the transformation's cost
    from ballistics import coarse
    sg, _state_of = solve.state_graph_for(g, info, 1, solve.world_for(m, info, "strict", 1), 0.0)
    wood, stone = coarse.WOOD << sg.M, coarse.STONE << sg.M
    into, out = [], []
    for u in np.flatnonzero(~inside):
        if sg.loc[u] < 0:
            continue
        for s2, v, c, k in sg.step(wood, int(sg.loc[u])):
            if inside[sg.glob[v]] and k == "roll":
                into.append((s2, c - float(np.linalg.norm(g.pos[sg.glob[v]] - g.pos[u]))))
    for u in np.flatnonzero(inside):
        for s2, v, c, k in sg.step(stone, int(sg.loc[u])):
            if not inside[sg.glob[v]] and k == "roll":
                out.append((s2, c - float(np.linalg.norm(g.pos[sg.glob[v]] - g.pos[u]))))
    assert into and all(s2 == stone for s2, _c in into) and min(c for _s, c in into) > extra - 1.0
    assert out and all(s2 == stone for s2, _c in out) and max(c for _s, c in out) < 1.0       # going out costs nothing and it stays stone
    res = staged.solve_sector(m, info, g, 1, tries=20_000, log=lambda *a: None)
    assert res["solved"] and [(t["at"], t["to"]) for t in res["transforms"]] == [("S", "stone")] and res["end_ball"] == "stone"
