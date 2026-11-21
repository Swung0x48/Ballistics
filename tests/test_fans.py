"""Fans (ballistics.fans and the kernel's tick): the push on made-up fans. The comparison with the original
engine's records needs game data and is not in the repository."""
import numpy as np

from ballistics import fans, physics as ph, things
from ballistics.coarse import without_bodiless
from ballistics.geometry import Mesh
from ballistics.kernel import OK, World
from ballistics.solve import world_for

FAN = dict(name="F1", type="P_Modul_18", sector=2, position=[0.0, 0.0, 0.0], x_axis=[1.0, 0.0, 0.0], y_axis=[0.0, 1.0, 0.0], z_axis=[0.0, 0.0, 1.0])
INFO = dict(placements=[FAN, dict(FAN, name="F2", sector=3, position=[40.0, 0.0, 0.0])], death_boxes=[])


def floor(y=-100.0, size=400.0):
    A = np.array([[-size, y, -size], [size, y, size]])
    B = np.array([[size, y, -size], [-size, y, size]])
    C = np.array([[-size, y, size], [size, y, -size]])
    return Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="floor", groups=[], friction=0.7, elasticity=0.3)], {})


def test_numbers_are_the_catalogues():
    R = fans.Rules.load()
    law = things.load().rules["fan_law"]["numbers"]
    assert R.impulse == law["impulse"]["v"] == 0.1
    assert (R.half, R.bottom, R.top) == (1.0, 1.14, 17.14)
    assert R.reach == things.load()["P_Modul_18"].numbers["push_distance_by_ball"]["v"]
    assert R.params(ph.STONE)[3] == R.reach["wood"]             # no record for stone: wood's distance
    for ball in (ph.WOOD, ph.PAPER):
        lo, hi = R.bounds[ball.name]
        assert lo < R.reach[ball.name] < hi
        assert (R.params(ball, "least")[3], R.params(ball)[3], R.params(ball, "most")[3]) == (lo, R.reach[ball.name], hi)


def test_only_the_sectors_fans():
    assert len(fans.of(INFO, None)) == 0 and len(fans.of(INFO, 1)) == 0
    f = fans.of(INFO, 2)
    assert f.shape == (1, 12) and list(f[0, :3]) == [0.0, 0.0, 0.0] and list(f[0, 6:9]) == [0.0, 1.0, 0.0]
    assert len(world_for(floor(), INFO, "strict", 3).fans) == 1 and len(world_for(floor(), INFO, "strict").fans) == 0


def test_the_box_and_the_distance():
    R = fans.Rules.load()
    f, par = fans.of(INFO, 2), R.params(ph.WOOD)
    d = R.reach["wood"]
    for pos, on in (([0, 9, 0], True), ([1 + d - 0.01, 9, 0], True), ([1 + d + 0.01, 9, 0], False), ([0, 1.14 - d + 0.01, 0], True),
                    ([0, 1.14 - d - 0.01, 0], False), ([0, 17.14 + d - 0.01, 0], True), ([0, 17.14 + d + 0.01, 0], False),
                    ([1 + d * 0.8, 9, 1 + d * 0.8], False), ([40, 9, 0], False)):
        assert fans.pushing(pos, f, par) == int(on), pos
    two = np.vstack([f, f])
    assert fans.pushing([0, 9, 0], two, par) == 2                 # boxes add up
    tilted = dict(FAN, x_axis=[0.0, 1.0, 0.0], y_axis=[1.0, 0.0, 0.0])        # the fan's own y axis along world x
    ft = fans.of(dict(placements=[tilted]), 2)
    assert fans.pushing([9, 0, 0], ft, par) == 1 and fans.pushing([0, 9, 0], ft, par) == 0


def test_the_tick_adds_the_impulse_between_damping_and_gravity():
    m = floor()
    off, on = World.of(m), World.of(m, fans=fans.of(INFO, 2))
    P = ph.Params()
    for ball in (ph.WOOD, ph.PAPER, ph.STONE):
        s = ph.State(np.array([0.5, 9.0, -0.3]), np.array([1.0, -3.0, 2.0]), np.zeros(3))
        a = off.step(ball, P, s.pos, s.vel, s.omega, np.zeros(3))
        b = on.step(ball, P, s.pos, s.vel, s.omega, np.zeros(3))
        assert np.array_equal(a[0], b[0]) and b[1][0] == a[1][0] and b[1][2] == a[1][2]      # the same move; only vy differs
        want = -3.0 * (1.0 - ball.lin_damp * ph.DT) + 0.1 / ball.mass - ph.GRAVITY * ph.DT
        assert abs(b[1][1] - want) < 1e-12 and abs(b[1][1] - a[1][1] - 0.1 / ball.mass) < 1e-12
        ref, _ = ph.step_reference(m, ball, s, np.zeros(3), P, fans=fans.of(INFO, 2))
        assert np.array_equal(ref.vel, b[1]) and np.array_equal(ref.pos, b[0])              # the plain tick is the same tick
    out = ph.State(np.array([20.0, 9.0, 0.0]), np.array([0.0, -3.0, 0.0]), np.zeros(3))      # outside: no push
    assert np.array_equal(on.step(ph.WOOD, P, out.pos, out.vel, out.omega, np.zeros(3))[1], off.step(ph.WOOD, P, out.pos, out.vel, out.omega, np.zeros(3))[1])


def test_decided_where_the_tick_begins():
    R = fans.Rules.load()
    m, P = floor(), ph.Params()
    on, off = World.of(m, fans=fans.of(INFO, 2)), World.of(m)
    edge = 1.0 + R.reach["wood"]
    for x0, pushed in ((edge + 0.05, False), (edge - 0.05, True)):        # moving inwards at 6.6 a second: 0.1 a tick
        s = (np.array([x0, 9.0, 0.0]), np.array([-6.6, 0.0, 0.0]), np.zeros(3))
        dv = on.step(ph.WOOD, P, *s, np.zeros(3))[1][1] - off.step(ph.WOOD, P, *s, np.zeros(3))[1][1]
        assert bool(dv > 0) is pushed, x0


def test_the_bounds_bracket_a_paper_ball_in_the_column():
    m, P = floor(-4000.0), ph.Params()
    ys = {}
    for edge in ("least", "best", "most"):                     # dropped beside the box's edge, drifting in: pushed sooner the wider the reach
        w = World.of(m, fans=fans.of(INFO, 2), fan_edge=edge)
        st, done, pos, vel, om, _s, tr = w.rollout(ph.PAPER, P, [5.5, 12.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros((1, 3)), [200], trace=True)
        assert st == OK
        ys[edge] = tr[:, 1]
    top = int(np.argmax(ys["most"]))                           # in order until the highest one tops out (after that they swing)
    assert (ys["least"][:top] <= ys["best"][:top] + 1e-9).all() and (ys["best"][:top] <= ys["most"][:top] + 1e-9).all()
    assert ys["least"][-1] < 0.0 < ys["best"][-1]              # at the edge the bounds differ in kind: carried up, or dropped


def test_a_wood_ball_falls_slower_in_the_column():
    m, P = floor(-4000.0), ph.Params()
    speeds = []
    for w in (World.of(m), World.of(m, fans=fans.of(INFO, 2))):
        st, done, pos, vel, om, _s, _t = w.rollout(ph.WOOD, P, [0.0, 17.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.zeros((1, 3)), [40])
        assert st == OK and done == 40
        speeds.append(-vel[1])
    assert speeds[1] < speeds[0] and abs((speeds[0] - speeds[1]) - sum(0.1 / 1.9 * (1 - 0.9 / 66) ** k for k in range(40))) < 0.2


def test_fans_and_markers_are_not_obstacles():
    tri = np.array([[0.0, 0.0, 0.0]])
    ents = [dict(name="floor", groups=["Phys_Floors"]), dict(name="P_Modul_18_Rotor", groups=[], placed=True, type="P_Modul_18"),
            dict(name="P_Modul_18_04", groups=["P_Modul_18", "Phys_Floors", "Sector_05"]), dict(name="P_Box", groups=[], placed=True, type="P_Box")]
    A = np.repeat(tri, 4, 0) + np.arange(4)[:, None] * 10.0
    m = Mesh(A, A + [1.0, 0.0, 0.0], A + [0.0, 0.0, 1.0], np.arange(4), np.zeros(4, bool), np.zeros((0, 7)), ents, {})
    kept = without_bodiless(m)
    assert sorted(set(kept.owner.tolist())) == [0, 3]
    assert without_bodiless(kept) is kept
