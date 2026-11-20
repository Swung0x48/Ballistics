"""The hand-written ball physics against cases with a known answer, and against a recorded engine run."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ballistics import physics as ph
from ballistics.geometry import CONTACT_MARGIN, Mesh

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1_MESH = DATA / "Level_01-mesh.npz"
L1_RUN = DATA / "runs" / "SR01_1_14_457_264.npz"


def plane(slope: float = 0.0, size: float = 400.0) -> Mesh:
    """A big square through the origin that rises by `slope` per unit of x."""
    h = size / 2
    p = [np.array((x, slope * x, z), float) for x, z in ((-h, -h), (h, -h), (h, h), (-h, h))]
    tris = [(p[0], p[2], p[1]), (p[0], p[3], p[2])]
    A, B, C = (np.array([t[k] for t in tris]) for k in range(3))
    return Mesh(A, B, C, np.zeros(2, np.int64), np.zeros(2, bool), np.zeros((0, 7)), [dict(name="plane", friction=0.7, elasticity=0.3)])


def resting(mesh: Mesh, x: float = 0.0) -> ph.State:
    y = mesh.rest(x, 0.0, 50.0) + CONTACT_MARGIN
    return ph.State(np.array([x, y, 0.0]), np.zeros(3))


def test_keys_push_along_the_camera():
    assert np.allclose(ph.key_push(ph.KEY_UP, 0), (0, 0, 1)) and np.allclose(ph.key_push(ph.KEY_RIGHT, 0), (1, 0, 0))
    # seen in the recorded run: camera quarter 2, UP and RIGHT held, the ball is pushed along -x and -z
    assert np.allclose(ph.key_push(ph.KEY_UP | ph.KEY_RIGHT, 2), (-1, 0, -1), atol=1e-12)
    assert np.allclose(ph.key_push(ph.KEY_UP | ph.KEY_DOWN, 1), (0, 0, 0), atol=1e-12)


def test_free_flight_is_the_games_update():
    m = plane()
    s = ph.State(np.array([0.0, 50.0, 0.0]), np.array([3.0, -0.30303, 0.0]))
    n, _ = ph.step(m, ph.WOOD, s, ph.key_push(ph.KEY_UP | ph.KEY_RIGHT, 2))
    assert np.allclose(n.pos, s.pos + s.vel / 66.0)                         # moved with the velocity it had
    lam = 1.0 - 0.9 / 66.0
    assert np.allclose(n.vel, (3.0 * lam - 0.43 / 1.9, -0.30303 * lam - 20.0 / 66.0, -0.43 / 1.9))


def test_a_ball_at_rest_stays():
    m = plane()
    s = resting(m)
    for _ in range(200):
        s, _ = ph.step(m, ph.WOOD, s, np.zeros(3))
    assert np.linalg.norm(s.vel) < 1e-9 and abs(s.pos[1] - (2.0 + CONTACT_MARGIN)) < 1e-9


def test_rolling_falls_out_of_friction():
    m = plane(size=4000.0)
    s, _ = ph.step(m, ph.WOOD, resting(m, -1900.0), ph.key_push(ph.KEY_RIGHT, 0))
    assert s.vel[0] == pytest.approx(0.43 / (1.4 * 1.9), rel=1e-9)             # a push accelerates by F / (1.4 m)
    assert np.allclose(s.omega, ph.rolling_spin(s.vel, (0, 1, 0)))             # and the ball rolls without slipping
    for _ in range(1500):
        s, _ = ph.step(m, ph.WOOD, s, ph.key_push(ph.KEY_RIGHT, 0))
    assert s.vel[0] == pytest.approx(15.9, abs=0.1)                            # the damping plateau measured in the game for one key
    assert abs(s.pos[1] - (2.0 + CONTACT_MARGIN)) < 1e-6                       # still on the floor
    s = resting(m, -1900.0)
    for _ in range(1500):
        s, _ = ph.step(m, ph.WOOD, s, ph.key_push(ph.KEY_RIGHT | ph.KEY_UP, 0))
    assert np.linalg.norm(s.vel) == pytest.approx(22.5, abs=0.15)              # two keys: 22.5 in the game


def test_a_slope_accelerates_by_five_sevenths_of_gravity_along_it():
    slope = 0.2
    m = plane(slope)
    s0 = resting(m)
    s, _ = ph.step(m, ph.WOOD, s0, np.zeros(3))
    theta = math.atan(slope)
    along = np.array([-math.cos(theta), -math.sin(theta), 0.0])                # downhill
    assert float(s.vel @ along) == pytest.approx((5.0 / 7.0) * 20.0 * math.sin(theta) / 66.0, rel=1e-6)
    assert abs(float(s.vel @ np.array([-math.sin(theta), math.cos(theta), 0.0]))) < 1e-9     # none of it into the slope


def test_the_stone_ball_holds_on_a_slope_up_to_its_push_over_gravity():
    # from a standstill a ball climbs while slope < push / gravity: stone 6.07 / 20, about 16.9 degrees
    for slope, climbs in ((0.27, True), (0.33, False)):
        m = plane(slope)
        s = resting(m)
        for _ in range(300):
            s, _ = ph.step(m, ph.STONE, s, ph.key_push(ph.KEY_RIGHT, 0))
            s = ph.settle(m, s)
        assert (s.pos[0] > 0.5) == climbs


@pytest.mark.skipif(not (L1_MESH.is_file() and L1_RUN.is_file()), reason="level 1 mesh or the recorded run not in the data folder")
def test_one_tick_ahead_against_the_engine():
    from ballistics.calibrate import one_tick
    from ballistics.trace import Run

    m = Mesh.load(L1_MESH)
    e = one_tick(m.subset(~m.placed), Run.load(L1_RUN), every=4)
    assert np.median(e["air"]) < 1e-4                    # free flight is the engine's own update
    assert np.median(e["floor"]) < 0.02 and np.median(e["rail"]) < 0.03
    assert len(e["floor"]) > 800 and len(e["air"]) > 800


def test_a_landing_stops_at_the_surface_and_keeps_a_quarter_of_the_approach_speed():
    # measured on 65 landings of the engine: gap 0.010 at the hit, 0.245 of the approach speed back (floor and rail alike)
    m = plane()
    s = ph.State(np.array([0.0, 2.1, 0.0]), np.array([0.0, -12.0, 0.0]))
    n, info = ph.step(m, ph.WOOD, s, np.zeros(3))
    assert info["hit"]
    assert n.pos[1] - 2.0 == pytest.approx(0.0099, abs=2e-4)                   # stopped where the engine stops it
    approach = 12.0 * (1.0 - 0.9 / 66.0) + 20.0 / 66.0                         # the tick's own update comes first
    assert n.vel[1] == pytest.approx(math.sqrt(0.2 * 0.3) * approach, rel=1e-6)
    # a ball that comes down spinning as if it rolled keeps its sideways speed; one that does not spin loses some to friction
    v = np.array([10.0, -12.0, 0.0])
    rolling, _ = ph.step(m, ph.WOOD, ph.State(s.pos.copy(), v, ph.rolling_spin(v * [1, 0, 1], (0, 1, 0))), np.zeros(3))
    dead, _ = ph.step(m, ph.WOOD, ph.State(s.pos.copy(), v), np.zeros(3))
    assert rolling.vel[0] == pytest.approx(10.0 * (1.0 - 0.9 / 66.0), abs=0.05) and dead.vel[0] < rolling.vel[0] - 1.0


def test_a_free_running_drop_bounces_with_the_measured_restitution_whatever_the_phase():
    # the engine: 0.245 of the approach speed on 65 landings. Running on its own ticks the kernel must give the same
    # for every height, i.e. wherever the last tick before the impact leaves the ball
    from ballistics.kernel import World
    w = World.of(plane(size=200.0))
    P = ph.Params()
    want = math.sqrt(ph.WOOD.elasticity * 0.3)
    for h in (6.0, 6.05, 6.1, 6.15, 6.2, 8.07, 10.11):
        pos, vel, omega = np.array([0.0, 2.0 + h, 0.0]), np.zeros(3), np.zeros(3)
        before = 0.0
        for _ in range(400):
            _s, _d, pos, vel, omega, _st, _ = w.rollout(ph.WOOD, P, pos, vel, omega, np.zeros((1, 3)), [1])
            if vel[1] > before + 0.5:
                break
            before = vel[1]
        assert before < -10.0 and vel[1] / -before == pytest.approx(want, abs=0.01)
