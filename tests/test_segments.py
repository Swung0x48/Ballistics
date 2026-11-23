"""The route file of version 2 (ballistics.segments): its pieces, where they are on the line, their exits and
corridors, and what they do to the mechanisms. Made-up floors and the made-up mechanisms of test_props only."""
import json

import numpy as np
import pytest

from ballistics import fine, physics as ph, route, segments
from ballistics.kernel import OK
from ballistics.solve import world_for
from test_props import board_info, board_mesh, floor, gate_info, seesaw_info, seesaw_mesh

PLUS_X, PLUS_Z = 1, 3                      # fine.PUSHES


def rolled(world, ball, state, tape, **kw):
    """A segment as the solver keeps one: the kernel run along `tape` from `state`."""
    pos, vel, om = (np.array(x, float) for x in state[:3])
    sc = np.array(state[3], float).copy()
    traces = []
    for a, n in tape:
        st, _d, pos, vel, om, _s, tr = world.rollout(ball, ph.Params(), pos, vel, om, fine.PUSHES[a:a + 1], [n], trace=True, scene=sc)
        assert st == OK
        traces.append(tr)
    return dict(type=kw.pop("type", "follow"), mode="strict", ball=ball.name, start=tuple(state), tape=list(tape), trace=np.concatenate(traces),
                end=(pos, vel, om, sc), **kw)


def anchors(info, start=(0.0, 2.0, -6.0)):
    return dict(info, start=list(start), checkpoints=[], airship=[0.0, 2.0, 30.0])


def result(sector, segs, solved=True, breaks=(), rolling=False):
    return dict(sector=sector, solved=solved, segments=segs, breaks=list(breaks), rolling_start=rolling)


def at_rest(world, pos, vel=(0.0, 0.0, 0.0)):
    v = np.array(vel, float)
    return (np.array(pos, float), v, np.array([v[2] / 2.0, 0.0, -v[0] / 2.0]), world.scene())


def test_a_roll_is_one_piece_on_the_line_with_an_exit_and_a_corridor(tmp_path):
    w = world_for(floor(), anchors(dict(placements=[], hulls=[], death_boxes=[])), "strict", 1)
    info = anchors(dict(placements=[], hulls=[], death_boxes=[]), (0.0, 0.011, 0.0))
    a = rolled(w, ph.WOOD, at_rest(w, (0.0, 0.011, 0.0)), [(PLUS_X, 200)])
    b = rolled(w, ph.WOOD, a["end"], [(PLUS_X, 200), (PLUS_Z, 100)])           # the next stretch: one piece with the first
    data = segments.build([result(1, [a, b])], info, 1, lambda s, m: w)
    segs = data["routes"][0]["segments"]
    assert len(segs) == 1 and segs[0]["kind"] == "roll" and segs[0]["ticks"] == 500 and data["routes"][0]["complete"]
    _P, _sec, t, meta = route.reference_line([result(1, [a, b])], info, 1)
    sg = segs[0]
    assert sg["s"][0] == 0.0 and abs(sg["s"][1] - meta["length"]) < 0.01 and abs(sg["t_ms"][1] - t[-1]) < 1.0
    assert sg["start"]["pos"] == [0.0, 0.011, 0.0] and not sg["assumed_rest_start"]
    e = sg["exit"]
    assert e["trials"] == 20 and e["not_lost"] == 20 and e["rough"]
    assert all(lo <= x <= hi for lo, x, hi in zip(e["box"]["min"], e["at"], e["box"]["max"]))
    assert e["speed_h"][0] <= np.hypot(e["vel"][0], e["vel"][2]) <= e["speed_h"][1] and e["vy"] is None
    assert e["dir"][1] == 0.0 and e["dir"][2] > 0.0 and 0.0 < e["dir_tol_deg"] < 30.0          # it turned to +z at the end
    c = sg["corridor"]
    assert c["trials"] == 20 * 4 and c["lost"] == 0                                           # 500 ticks: four windows (the first one shorter)
    assert len(c["half_width"]) == int(sg["s"][1] // 1.0) + 1 and 0.0 < max(c["half_width"]) < 3.0
    # written, it reads back; a line other than the one it was written with is refused
    line = route.write_line(tmp_path, 1, *route.reference_line([result(1, [a, b])], info, 1))
    data = segments.build([result(1, [a, b])], info, 1, lambda s, m: w, line=line.name, line_sha256=segments.file_sha256(line))
    path = segments.write(tmp_path, 1, data)
    assert path.name == "L01.segments.json" and segments.read(path) == json.loads(path.read_text(encoding="utf-8"))
    data["line_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        segments.read(path)


def test_after_a_break_the_pieces_are_off_the_line_and_start_at_rest():
    info = anchors(dict(placements=[], hulls=[], death_boxes=[]), (0.0, 0.011, 0.0))
    w = world_for(floor(), info, "strict", 1)
    a = rolled(w, ph.WOOD, at_rest(w, (0.0, 0.011, 0.0)), [(PLUS_X, 150)])
    far = np.array([60.0, 0.011, 0.0])
    b = rolled(w, ph.WOOD, at_rest(w, far), [(PLUS_X, 150)])
    left = dict(status="unresolved", no="B1", near=[30.0, 0.011, 0.0], far=list(far), between=[], attempts=[dict(mode="strict", tries=10)])
    c = rolled(w, ph.WOOD, at_rest(w, (0.0, 0.011, 40.0)), [(PLUS_X, 150)])
    res = [result(1, [a, dict(type="assumed", no="B1", to=far), b], solved=False, breaks=[left]), result(2, [c])]
    segs = segments.build(res, info, 1, lambda s, m: w)["routes"][0]["segments"]
    assert [s["kind"] for s in segs] == ["roll", "break", "roll", "roll"]
    on, brk, past, nxt = segs
    assert on["s"] is not None and not on["assumed_rest_start"] and "rest_start" not in on
    assert brk["break"]["no"] == "B1" and brk["break"]["near"] == [30.0, 0.011, 0.0] and brk["s"] is None
    assert past["s"] is None and past["t_ms"] is None and past["own_t_ms"] == [0.0, round(150 * route.TICK_MS, 1)]
    assert past["assumed_rest_start"] and past["rest_start"] == "far_side_of_break"
    assert nxt["assumed_rest_start"] and nxt["rest_start"] == "reset_point" and nxt["sector"] == 2


def test_a_flight_is_a_piece_of_its_own():
    # a ledge: the floor at 0 up to x 20, then 6 lower
    A = np.array([[-40, 0, -40], [20, 0, 40], [20, -6, -40], [200, -6, 40]], float)
    B = np.array([[20, 0, -40], [-40, 0, 40], [200, -6, -40], [20, -6, 40]], float)
    C = np.array([[-40, 0, 40], [20, 0, -40], [20, -6, 40], [200, -6, -40]], float)
    from ballistics.geometry import Mesh
    m = Mesh(A, B, C, np.zeros(4, np.int64), np.zeros(4, bool), np.zeros((0, 7)), [dict(name="floor", groups=[], friction=0.7, elasticity=0.3)], {})
    info = anchors(dict(placements=[], hulls=[], death_boxes=[]), (0.0, 2.011, 0.0))
    w = world_for(m, info, "strict", 1)
    a = rolled(w, ph.WOOD, at_rest(w, (0.0, 2.011, 0.0), (12.0, 0.0, 0.0)), [(PLUS_X, 400)])
    segs = segments.build([result(1, [a])], info, 1, lambda s, m: w)["routes"][0]["segments"]
    kinds = [s["kind"] for s in segs]
    assert kinds[:3] == ["roll", "flight", "roll"]
    fl = segs[1]
    assert fl["exit"]["vy"] is not None and fl["exit"]["vy"][0] < -3.0 and fl["exit"]["at"][0] > 22.0
    assert all(segs[k]["s"][1] == segs[k + 1]["s"][0] for k in range(len(segs) - 1))          # end to end on the line


def test_a_push_board_pushed_over_is_marked_with_the_speed_it_was_struck_at():
    info = anchors(board_info())
    w = world_for(board_mesh(), info, "strict", 1)
    a = rolled(w, ph.WOOD, at_rest(w, (0.0, 2.0, -6.0), (0.0, 0.0, 5.0)), [(PLUS_Z, 400)])
    segs = segments.build([result(1, [a])], info, 1, lambda s, m: w)["routes"][0]["segments"]
    ms = [m for s in segs for m in s["mechanisms"]]
    assert len(ms) == 1
    m = ms[0]
    assert m["name"] == "B" and m["type"] == "push_board" and m["does"] == "push_over" and m["state"] == [0, 1] and m["ball"] == ["wood", "stone"]
    # it starts at 5 towards the board with the key held, and strikes it at about 8.4 (the board wakes in that tick and
    # the ball loses most of its speed in it): the route's speed then is in the range
    lo, hi = m["impact_speed"]
    sp = a["trace"][:, 6]
    struck = float(sp[int(np.argmax(np.diff(sp) < -3.0))])
    assert 7.5 < lo <= struck <= hi < 9.5 and m["pushed_over"] >= 10


def test_a_gate_push_and_a_tipped_seesaw():
    info = anchors(gate_info(), (-7.55, -0.81, 0.65))
    w = world_for(floor(-2.81), info, "strict", 1)
    a = rolled(w, ph.STONE, at_rest(w, (-7.55, -0.81, 0.65)), [(PLUS_X, 198)], operates="G")
    segs = segments.build([result(1, [a])], info, 1, lambda s, m: w)["routes"][0]["segments"]
    assert [s["kind"] for s in segs] == ["operate"]
    (g,) = segs[0]["mechanisms"]
    assert g == dict(name="G", type="gate", does="push_open", ball=["stone"], key="+x", hold_ms=round(198 * route.TICK_MS, 1), opened=True)
    info = anchors(seesaw_info(), (-11.0, 2.51, 0.0))
    w = world_for(seesaw_mesh(), info, "strict", 1)
    a = rolled(w, ph.STONE, at_rest(w, (-11.0, 2.51, 0.0), (7.4, 0.0, 0.0)), [(PLUS_X, 400)])
    ms = [m for s in segments.build([result(1, [a])], info, 1, lambda s, m: w)["routes"][0]["segments"] for m in s["mechanisms"]]
    assert [(m["name"], m["type"], m["does"], m["tips"], m["state"]) for m in ms] == [("W", "seesaw", "cross", True, [0, 1])]


def test_the_line_keeps_its_ticks_where_the_pieces_can_find_them():
    info = anchors(dict(placements=[], hulls=[], death_boxes=[]), (0.0, 0.011, 0.0))
    w = world_for(floor(), info, "strict", 1)
    a = rolled(w, ph.WOOD, at_rest(w, (0.0, 0.011, 0.0)), [(PLUS_X, 300)])
    P, sec, t, meta, tick_s = route.line_and_ticks([result(1, [a])], info, 1)
    assert len(tick_s) == 300 and (np.diff(tick_s) >= 0).all() and abs(tick_s[-1] - meta["length"]) < 1e-6
    assert all(np.array_equal(x, y) for x, y in zip(route.reference_line([result(1, [a])], info, 1)[:3], (P, sec, t)))
