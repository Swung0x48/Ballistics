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
    assert props.of(info, None) == [] and [p["name"] for p in props.of(info, 1)] == ["a"]
    w = world_for(floor(), info, "strict", 1)
    assert len(w.props) == 1 and w.props[0]["settled"] and abs(w.props[0]["rest"][1] - 0.011) < 0.02      # it fell 3 and lies on the floor
    assert len(world_for(floor(), info, "free", 1).props) == 0 and len(world_for(floor(), info, "strict").props) == 0
    assert w.scene().shape == (1, 11) and w.scene()[0, 9] == 0.0


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
