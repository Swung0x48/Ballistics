"""Loose balls (ballistics.props and the kernel's tick of several bodies): made-up floors only. What the numbers
are held against is in docs/mechanisms.md."""
import numpy as np

from ballistics import physics as ph, props, things
from ballistics.geometry import Mesh
from ballistics.kernel import OK, World
from ballistics.solve import world_for


def floor(y=-2.0, size=400.0):
    A = np.array([[-size, y, -size], [size, y, size]])
    B = np.array([[size, y, -size], [-size, y, size]])
    C = np.array([[-size, y, size], [size, y, -size]])
    return Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="floor", groups=[], friction=0.7, elasticity=0.3)], {})


def prop(kind, x=0.0, z=0.0):
    return dict(props.bodies()[kind], name=kind, rest=[x, 0.011, z])


def strike(ball, kind, v0, key, ticks=40):
    """A ball rolling at v0 into a loose ball at rest: per tick (the striker's vx, the struck ball's vx, the striker's height)."""
    w = World.of(floor(), props=[prop(kind)])
    scene = w.scene()
    pos, vel, om = np.array([-4.3, 0.011, 0.0]), np.array([v0, 0.0, 0.0]), np.array([0.0, 0.0, -v0 / 2.0])
    push = np.array([[1.0 if key else 0.0, 0.0, 0.0]])
    rec = []
    for _ in range(ticks):
        st, _d, pos, vel, om, _s, _t = w.rollout(ball, ph.Params(), pos, vel, om, push, [1], scene=scene)
        assert st == OK
        rec.append((vel[0], scene[0, 3], pos[1]))
    rec = np.array(rec)
    return rec, int(np.argmax(rec[:, 1] > 0.01))


def test_the_bodies_are_the_catalogues():
    b = props.bodies()
    cat = {x["name"]: x for x in things.load()["P_Ball"].bodies}
    assert (b["wood"]["mass"], b["wood"]["friction"], b["wood"]["lin_damp"]) == (2.0, 0.6, 0.6) == tuple(cat["P_Ball_Wood_MF"][k]["v"] for k in ("mass", "friction", "lin_damp"))
    assert (b["stone"]["mass"], b["stone"]["elasticity"]) == (10.0, 0.1) and b["paper"]["mass"] == 0.2
    assert abs(b["stone"]["inertia"] - 0.4 * 10.0 * 4.0) < 1e-12 and abs(b["paper"]["inertia"] - 0.277 * 0.2 * 4.0) < 1e-12


def test_only_the_sectors_and_they_settle():
    info = dict(death_boxes=[], placements=[dict(name="a", type="P_Ball_Wood", sector=1, position=[0.0, 3.0, 0.0]),
                                            dict(name="b", type="P_Ball_Stone", sector=2, position=[10.0, 0.5, 0.0]),
                                            dict(name="c", type="P_Box", sector=1, position=[20.0, 0.5, 0.0])])
    assert props.of(info, None) == [] and [p["name"] for p in props.of(info, 1)] == ["a", "c"]       # the sector's ball and its box
    w = world_for(floor(), info, "strict", 1)
    assert len(w.props) == 2 and w.props[0]["settled"] and abs(w.props[0]["rest"][1] - 0.011) < 0.02      # it fell 3 and lies on the floor
    assert len(world_for(floor(), info, "free", 1).props) == 0 and len(world_for(floor(), info, "strict").props) == 0
    assert w.scene().shape == (2, 11) and w.scene()[0, 9] == 0.0


def test_without_a_loose_ball_near_the_tick_is_the_old_one():
    P = ph.Params()
    plain, with_one = World.of(floor()), World.of(floor(), props=[prop("stone", x=100.0)])
    a = plain.rollout(ph.WOOD, P, [0.0, 0.011, 0.0], [5.0, 0.0, 1.0], [0.3, 0.0, -2.5], np.array([[1.0, 0.0, 1.0]]), [200], trace=True)
    b = with_one.rollout(ph.WOOD, P, [0.0, 0.011, 0.0], [5.0, 0.0, 1.0], [0.3, 0.0, -2.5], np.array([[1.0, 0.0, 1.0]]), [200], trace=True)
    assert np.array_equal(a[6], b[6]) and np.array_equal(a[3], b[3])


def test_a_struck_ball_takes_the_two_body_rule_in_the_tick_and_the_floor_drags_on_it_after():
    # MEASURED (docs/mechanisms.md): in the tick of the impact the struck ball has the plain rule's speed to within 3%;
    # 10 to 20 ticks later 0.45 to 0.56 of it
    for ball, kind in ((ph.WOOD, "stone"), (ph.WOOD, "wood"), (ph.STONE, "stone")):
        b = props.bodies()[kind]
        rule = 12.0 * (1 + np.sqrt(ball.elasticity * b["elasticity"])) * ball.mass / (ball.mass + b["mass"])
        rec, hit = strike(ball, kind, 12.0, key=False)
        assert abs(rec[hit, 1] / rule - 1.0) < 0.05, (ball.name, kind, rec[hit, 1] / rule)
        if ball is ph.WOOD:
            later = rec[hit + 10:hit + 20, 1] / rule
            assert 0.40 < later.min() and later.max() < 0.70, (ball.name, kind, later)


def test_what_the_striker_keeps():
    # MEASURED on the rebuilt engine (the catalogue's P_Ball): wood on wood 0.38-0.43, wood on stone 0.04-0.09 (and lifted),
    # stone on stone about 0.45, stone on wood 0.82-0.87, stone on paper 0.98-1.04, wood on paper 0.92-0.99 (keys held)
    for ball, kind, key, lo, hi in ((ph.WOOD, "wood", False, 0.33, 0.45), (ph.WOOD, "stone", False, 0.02, 0.10), (ph.STONE, "stone", True, 0.40, 0.50),
                                    (ph.STONE, "wood", True, 0.78, 0.90), (ph.STONE, "paper", True, 0.95, 1.06), (ph.WOOD, "paper", True, 0.85, 1.0)):
        rec, hit = strike(ball, kind, 12.0, key)
        assert lo < rec[hit, 0] / 12.0 < hi, (ball.name, kind, rec[hit, 0] / 12.0)
    rec, hit = strike(ph.WOOD, "stone", 12.0, key=False)
    assert rec[:, 2].max() - 0.011 > 0.05                              # a rolling wood ball is lifted off the floor by a stone one


def test_a_frozen_ball_wakes_when_struck_and_freezes_again():
    w = World.of(floor(), props=[prop("wood")])
    scene = w.scene()
    P = ph.Params()
    st, d, pos, vel, om, _s, _t = w.rollout(ph.WOOD, P, [-6.0, 0.011, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, -4.0], np.zeros((1, 3)), [30], scene=scene)
    assert st == OK and scene[0, 9] == 1.0 and scene[0, 0] > 0.2        # awake and moved on
    st, d, pos, vel, om, _s, _t = w.rollout(ph.WOOD, P, pos, vel, om, np.zeros((1, 3)), [1500], scene=scene)
    assert scene[0, 9] == 0.0 and not scene[0, 3:9].any()               # at rest again: frozen where it stopped
    left = w.scene()                                                    # a rollout without a scene leaves the world's own untouched
    assert left[0, 0] == 0.0 and left[0, 9] == 0.0


def test_a_row_of_stone_balls_holds_a_wood_ball_up_and_gives_way_to_a_stone_one():
    row = [prop("stone", x=0.0, z=z) for z in (-4.5, 0.0, 4.5)]
    out = {}
    for ball in (ph.WOOD, ph.STONE):
        w = World.of(floor(), props=row)
        scene = w.scene()
        st, d, pos, vel, om, _s, tr = w.rollout(ball, ph.Params(), [-10.0, 0.011, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.array([[1.0, 0.0, 0.0]]), [400],
                                                trace=True, scene=scene)
        assert st == OK
        out[ball.name] = pos[0]
    assert out["stone"] > out["wood"] + 5.0                             # in 3 game seconds the stone ball is well through; the wood one shoves slowly


def box(x, y=0.56, z=0.0):
    return dict(props.bodies()["box"], name="box", axes=np.eye(3).tolist(), rest=[x, y, z])


def push(ball, boxes, x0, ticks=400):
    """A ball at rest x0 behind boxes on a floor, the key held towards them: (per box how far it moved, seconds to
    move the first one a box's length or None, the ball's speed at the end)."""
    w = World.of(floor(), props=boxes)
    scene = w.scene()
    scene[:, 9] = 1.0
    w.idle(scene, 300)                                                 # they settle onto the floor and each other
    assert (scene[:, 9] == 0.0).all()
    start = scene[:, :3].copy()
    pos, vel, om = np.array([x0, 0.011, 0.0]), np.zeros(3), np.zeros(3)
    first = five = None
    for t in range(ticks):
        st, _d, pos, vel, om, _s, _t = w.rollout(ball, ph.Params(), pos, vel, om, np.array([[1.0, 0.0, 0.0]]), [1], scene=scene)
        assert st == OK
        moved = scene[0, 0] - start[0, 0]
        if first is None and moved > 0.01:
            first = t
        if five is None and moved >= 5.0:
            five = (t - first) / 132.0
    return scene[:, 0] - start[:, 0], five, float(np.linalg.norm(vel)), scene[:, 1] - start[:, 1]


def test_the_box_is_the_catalogues():
    b = props.bodies()["box"]
    cat = things.load()["P_Box"]
    assert (b["mass"], b["friction"], b["elasticity"]) == (1.0, 0.7, 0.3) == tuple(cat.bodies[0][k]["v"] for k in ("mass", "friction", "elasticity"))
    assert b["half"] == (2.5, 2.5, 2.5) and b["shape"] == "box" and b["inertia"] == props.LOCKED
    info = dict(death_boxes=[], placements=[dict(name="b", type="P_Box", sector=1, position=[0.0, 1.0, 0.0], x_axis=[0.0, 0.0, 1.0], y_axis=[0.0, 1.0, 0.0], z_axis=[-1.0, 0.0, 0.0])])
    w = world_for(floor(), info, "strict", 1)
    assert len(w.props) == 1 and w.props[0]["settled"] and abs(w.props[0]["rest"][1] - 0.56) < 0.05      # it stands on the floor
    assert w.props[0]["axes"][0] == [0.0, 0.0, 1.0] and w.prop_consts.shape == (1, 20) and w.prop_consts[0, 6] == 1.0


def test_who_can_push_what():
    # MEASURED on the rebuilt engine (the catalogue's P_Box): a wood ball pushes one box (a box's length in 1.0 game s from
    # touching, 0.29-0.50 with a run-up), not two in a row nor a pile of three; a stone ball pushes all of these (one box:
    # 0.76-0.77 from touching); a paper ball moves a box 0.48 at most
    moved, five, _v, _dy = push(ph.WOOD, [box(4.6)], 0.0)
    assert moved[0] > 5.0 and 0.9 < five < 1.5                          # here 1.33: slower than measured
    moved, five, _v, _dy = push(ph.WOOD, [box(4.6)], -12.0)
    assert 0.25 < five < 0.55
    moved, five, _v, _dy = push(ph.STONE, [box(4.6)], 0.0)
    assert 0.7 < five < 1.0
    moved, five, _v, _dy = push(ph.PAPER, [box(4.6)], -12.0)
    assert five is None and moved[0] < 0.5
    moved, five, speed, _dy = push(ph.WOOD, [box(4.6), box(9.7)], 0.0, ticks=600)
    assert moved.max() < 1.0 and speed < 0.5                            # two in a row: it comes to a stop
    moved, five, speed, dy = push(ph.WOOD, [box(4.6), box(4.6, y=5.6), box(4.6, y=10.7)], 0.0, ticks=600)
    assert moved.max() < 1.0 and np.abs(dy).max() < 0.1                 # a pile of three: it does not move
    moved, five, speed, _dy = push(ph.STONE, [box(4.6), box(9.7)], 0.0, ticks=600)
    assert moved.min() > 20.0
    moved, five, speed, dy = push(ph.STONE, [box(4.6), box(4.6, y=5.6), box(4.6, y=10.7)], 0.0, ticks=600)
    assert moved.min() > 10.0 and np.abs(dy).max() < 0.1                # the whole pile slides


def test_a_ball_can_stand_on_a_box_and_a_box_goes_over_an_edge():
    w = World.of(floor(), props=[box(0.0)])
    scene = w.scene()
    st, _d, pos, vel, _o, _s, _t = w.rollout(ph.WOOD, ph.Params(), [0.0, 9.0, 0.0], [0, 0, 0], [0, 0, 0], np.zeros((1, 3)), [300], scene=scene)
    assert st == OK and abs(pos[1] - (0.56 + 2.5 + 2.0)) < 0.1          # at rest on its top
    A = np.array([[-40.0, -2.0, -40.0], [12.0, -2.0, 40.0]]); B = np.array([[12.0, -2.0, -40.0], [-40.0, -2.0, 40.0]]); C = np.array([[-40.0, -2.0, 40.0], [12.0, -2.0, -40.0]])
    ledge = Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="ledge", groups=[], friction=0.7, elasticity=0.3)], {})
    w = World.of(ledge, props=[box(4.6)])
    scene = w.scene()
    pos, vel, om = np.array([0.0, 0.011, 0.0]), np.zeros(3), np.zeros(3)
    for _ in range(900):
        st, _d, pos, vel, om, _s, _t = w.rollout(ph.STONE, ph.Params(), pos, vel, om, np.array([[1.0, 0.0, 0.0]]), [1], scene=scene)
        if st != OK or scene[0, 9] == 2.0:
            break
    assert scene[0, 9] == 2.0 or scene[0, 1] < -10.0                    # pushed over the edge at x = 12: it is gone, or falling


def rock(x):
    """A falling rock as the kernel takes it: a crate on the floor and, on it, a block on an upright rail."""
    table = {b["name"]: b for b in things.load()["P_Modul_34"].bodies}
    def body(name, half, y, slider):
        b = table[name]
        return dict(kind=name, name=name, shape="box", mass=float(b["mass"]["v"]), friction=float(b["friction"]["v"]), elasticity=float(b["elasticity"]["v"]),
                    lin_damp=float(b["lin_damp"]["v"]), rot_damp=float(b["rot_damp"]["v"]), inertia=props.LOCKED, half=half, axes=np.eye(3).tolist(),
                    slider=slider, rest=[x, y, 0.0])
    return [body("P_Modul_34_Kiste", (2.05, 2.5, 2.05), 0.56, False), body("P_Modul_34_Schiebestein", (2.5, 2.5, 2.5), 5.8, True)]


def strike_rock(ball, v0, ticks=450):
    w = World.of(floor(), props=rock(4.4))
    scene = w.scene()
    scene[:, 9] = 1.0
    w.idle(scene, 400)
    start = scene[:, :3].copy()
    pos, vel, om = np.array([0.0, 0.011, 0.0]), np.array([v0, 0.0, 0.0]), np.array([0.0, 0.0, -v0 / 2.0])
    for _ in range(ticks):
        st, _d, pos, vel, om, _s, _t = w.rollout(ball, ph.Params(), pos, vel, om, np.array([[1.0, 0.0, 0.0]]), [1], scene=scene)
        assert st == OK
    return scene[0, 0] - start[0, 0], start[1, 1] - scene[1, 1], np.abs(scene[1, [0, 2]] - start[1, [0, 2]]).max()


def test_the_falling_rock():
    # MEASURED on the rebuilt engine (the catalogue's P_Modul_34), keys held: a stone ball at 5.95 to 8.20 pushes the crate
    # out and the block comes down (8 of 8), at 3.33 it does not; a wood ball moves the crate 1.03-1.06 at 8.44 and 1.58-1.73
    # at 10.94, the block stays; a paper ball does not move it
    crate, drop, side = strike_rock(ph.STONE, 7.0)
    assert crate > 5.0 and 4.5 < drop < 5.5 and side < 1e-9             # the block is down on the floor, and never left its rail
    crate, drop, side = strike_rock(ph.STONE, 3.33)
    assert crate < 5.0 and drop < 0.1
    crate, drop, side = strike_rock(ph.WOOD, 8.44)
    assert 0.8 < crate < 1.3 and drop < 0.1
    crate, drop, side = strike_rock(ph.WOOD, 10.94)
    assert 1.3 < crate < 1.9 and drop < 0.1
    crate, drop, side = strike_rock(ph.PAPER, 11.04)
    assert crate < 0.2 and drop < 0.1


def test_a_mechanisms_parts_are_read_from_their_shapes():
    def slab(x0, x1, y0, y1, z0, z1):
        c = [np.array((x, y, z), float) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        q = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
        return [t for a, b, d, e in q for t in ((c[a], c[b], c[d]), (c[a], c[d], c[e]))]
    parts = [(slab(-40, 40, -2.0, -2.0, -40, 40)[4:6], dict(name="floor", groups=["Phys_Floors"])),
             (slab(7.95, 12.05, -1.9, 2.2, -2.5, 2.5), dict(name="P_Modul_34_Kiste", placed=True, type="P_Modul_34", placement="R1", sector=1)),
             (slab(7.5, 12.5, 2.4, 7.4, -2.5, 2.5), dict(name="P_Modul_34_Schiebestein", placed=True, type="P_Modul_34", placement="R1", sector=1))]
    A, B, C, owner, ents = [], [], [], [], []
    for tris, e in parts:
        ents.append(e)
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents, {})
    info = dict(death_boxes=[], placements=[dict(name="R1", type="P_Modul_34", sector=1, position=[0.0, 0.0, 0.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0])])
    got = {b["kind"]: b for b in props.part_boxes(info, m)}
    assert np.allclose(got["P_Modul_34_Kiste"]["half"], (2.05, 2.05, 2.5)) and got["P_Modul_34_Schiebestein"]["slider"] and not got["P_Modul_34_Kiste"]["slider"]
    assert np.allclose(got["P_Modul_34_Schiebestein"]["placed"], (10.0, 4.9, 0.0))
    w = world_for(m, info, "strict", 1)
    assert [p["kind"] for p in w.props] == ["P_Modul_34_Kiste", "P_Modul_34_Schiebestein"] and all(p["settled"] for p in w.props)
    assert not any(e.get("type") == "P_Modul_34" and (w.mesh.owner == i).any() for i, e in enumerate(w.mesh.entities))    # their shapes left the mesh
    assert w.props[1]["rest"][1] < 4.9 and w.props[1]["rest"][1] > 4.0      # the block came down onto the crate


def test_boxes_one_on_another_and_on_a_slope():
    w = World.of(floor(), props=[box(0.0, y=0.9), box(0.0, y=6.2)])        # exactly one above the other, a little apart
    scene = w.scene()
    scene[:, 9] = 1.0
    w.idle(scene, 600)
    assert (scene[:, 9] == 0.0).all() and abs(scene[1, 1] - scene[0, 1] - 5.0) < 0.2     # the upper one stands on the lower one
    A = np.array([[-40.0, -2.0 - 40 * 0.36, -40.0], [40.0, -2.0 + 40 * 0.36, 40.0]]); B = np.array([[40.0, -2.0 + 40 * 0.36, -40.0], [-40.0, -2.0 - 40 * 0.36, 40.0]])
    C = np.array([[-40.0, -2.0 - 40 * 0.36, 40.0], [40.0, -2.0 + 40 * 0.36, -40.0]])
    slope = Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="slope", groups=[], friction=0.7, elasticity=0.3)], {})
    w = World.of(slope, props=[box(0.0, y=2.5)])                            # a slope of 20 degrees: the box rests on its lower edge
    scene = w.scene()
    scene[:, 9] = 1.0
    w.idle(scene, 900)
    assert scene[0, 9] == 0.0 and not scene[0, 3:6].any() and abs(scene[0, 0]) < 3.0     # at rest, not stuck with a growing speed, not slid away
