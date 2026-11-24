import math

import numpy as np
import pytest
from numba import njit
from scipy.special import lambertw

from ballistics import closedform as cf, fans
from ballistics.physics import BALLS, DT, GRAVITY, KEY_RIGHT, KEY_UP, PAPER, STONE, WOOD, Params, State, key_push


def state(y=10.0, v=(1.0, 4.0, -2.0)):
    return State(np.array([0., y, 0.]), np.array(v), np.array([1., -2., 3.]))


def reference(pos, vel, loss, b, n):
    """Independent literal recurrence for tests only; never calls the project stepper."""
    p, v = np.array(pos, dtype=float), np.array(vel, dtype=float)
    for _ in range(n):
        p = p + DT * v
        v = (1 - loss) * v + b
    return p, v


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
@pytest.mark.parametrize('keys', [0, KEY_UP, KEY_UP | KEY_RIGHT])
def test_flight_formula(ball, keys):
    s = state()
    ns = [0, 1, 5, 66, 500, 1000]
    b = ball.push * key_push(keys, 3) + [0, -GRAVITY * DT, 0]
    arr = cf.flight(ball, s, ns, keys, 3)
    for i, n in enumerate(ns):
        p, v = reference(s.pos, s.vel, ball.lin_damp * DT, b, n)
        np.testing.assert_allclose(arr.pos[i], p, atol=1e-9, rtol=0)
        np.testing.assert_allclose(arr.vel[i], v, atol=1e-9, rtol=0)
        np.testing.assert_allclose(arr.omega[i], s.omega * (1 - ball.rot_damp * DT)**n, atol=1e-9, rtol=0)


@pytest.mark.parametrize('branch', [0, -1])
def test_lambert_w_numba(branch):
    @njit
    def compiled(x, b):
        return cf.lambertw_real(x, b)

    x = np.r_[-1 / math.e + np.geomspace(1e-6, .02, 50), -np.geomspace(1e-300, .3, 90)]
    if branch == 0:
        x = np.r_[x, 0., np.geomspace(1e-300, 1e300, 120)]
    np.testing.assert_allclose([compiled(a, branch) for a in x], lambertw(x, branch).real, atol=0, rtol=5e-12)
    assert compiled(-1 / math.e, branch) == -1
    w = compiled(np.nextafter(-1 / math.e, 0), branch)
    assert (w > -1) == (branch == 0)
    assert math.isnan(compiled(-1, branch))
    assert compiled(0, branch) == (0 if branch == 0 else -math.inf)


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
def test_landing_and_arbitrary_plane(ball):
    s = state()
    heights = cf.flight(ball, s, np.arange(250)).pos[:, 1]
    for h in (-10, 0, 9.9, 10, 11):
        below = np.flatnonzero(heights <= h)
        assert cf.landing_step(ball, s, h) == (below[0] if len(below) else None)
    point = np.array([-3., 2., 0.])
    normal = np.array([1., 2., -1.])
    normal /= np.linalg.norm(normal)
    traj = cf.flight(ball, s, np.arange(250), KEY_UP, 1).pos
    k = np.flatnonzero((traj - point) @ normal <= ball.radius)[0]
    hit = cf.plane_hit(ball, s, point, 3 * normal, KEY_UP, 1)
    assert hit.step == k
    assert 0 < hit.fraction <= 1
    np.testing.assert_allclose(hit.center, traj[k-1] + hit.fraction * (traj[k] - traj[k-1]), atol=1e-12)
    assert (hit.center - hit.point) @ normal == pytest.approx(ball.radius)
    assert cf.plane_hit(ball, s, [0, 100, 0], [0, -1, 0]) is None


def test_plane_exact_integer_and_between_samples_near_apex():
    s = state(0, (0., 1., 0.))
    # A real discrete endpoint equality should not require an additional tick.
    ns = 16
    height = cf.flight(WOOD, s, ns).pos[1]
    assert cf.landing_step(WOOD, s, height) == ns
    shifted = state(-679.5143392504639, (0., 1., 0.))
    y = cf.flight(WOOD, shifted, 130).pos[1]
    assert cf.landing_step(WOOD, shifted, y) == 130
    b = -GRAVITY * DT
    peak = cf._turn(s.vel[1], b, WOOD.lin_damp * DT)
    high = cf.flight(WOOD, s, peak).pos[1]
    sampled = cf.flight(WOOD, s, [math.floor(peak), math.ceil(peak)]).pos[:, 1].max()
    assert cf.plane_hit(WOOD, s, [0, WOOD.radius + (high + sampled) / 2, 0], [0, -1, 0]) is None


def test_triangle_faces_and_sampled_edges():
    s = state(10, (0., -1., 0.))
    tri = np.array([[-8., 0., -8.], [8., 0., -8.], [0., 0., 8.]])
    for t in (tri, tri[::-1]):
        hit = cf.triangle_face_hit(WOOD, s, t)
        assert hit.step == cf.landing_step(WOOD, s, WOOD.radius)
        np.testing.assert_allclose(hit.point, [0, 0, 0], atol=1e-12)
    assert cf.triangle_face_hit(WOOD, s, tri + [30, 0, 0]) is None
    tri = np.array([[1., 0., -4.], [1., 0., 4.], [7., 0., 0.]])
    assert cf.triangle_face_hit(WOOD, s, tri) is None
    np.testing.assert_array_equal(cf.triangle_edge_samples(WOOD, s, tri, [0, 3.5, cf.landing_step(WOOD, s, 1)]), [False, False, True])
    for t in (tri, tri + [0, 0, 4.5]):  # closest is edge and vertex respectively
        assert cf.triangle_edge_samples(WOOD, state(0.5), t, [0])[0]


def test_triangle_return_crossing():
    # The ascending crossing of this plane is outside the triangle; the descending one is inside.
    s = state(0, (8., 18., 0.))
    point = [0, 3, 0]
    first = cf.plane_hit(WOOD, s, point, [0, -1, 0])
    assert first.point[0] < 2
    tri = np.array([[2., 3., -4.], [8., 3., -4.], [5., 3., 4.]])
    hit = cf.triangle_face_hit(WOOD, s, tri)
    assert hit.step > first.step
    assert hit.normal[1] == pytest.approx(1)


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
def test_reach_square(ball):
    s = state()
    for q in range(4):
        k = KEY_UP | KEY_RIGHT
        sq = cf.reach_square(ball, s, 66, q)
        corner = cf.flight(ball, s, 66, k, q).pos[[0, 2]]
        assert sq.contains(corner)
        assert np.linalg.norm(corner - sq.center) == pytest.approx(sq.half * math.sqrt(2))
        assert not sq.contains(sq.center + (corner - sq.center) * 1.0001)
        assert cf.reach_square(ball, s, 1, q).half == 0


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
def test_rolling_vector_and_inverse(ball):
    nrm = np.array([-0.1, math.sqrt(0.99), 0])
    axis = np.array([nrm[1], -nrm[0], 0])
    law = cf.rolling(ball, nrm, KEY_RIGHT)
    a = law.along(axis)
    assert a.climbs_from_rest()
    ratio = ball.inertia / (ball.mass * ball.radius**2)
    b = (ball.push * axis[0] - GRAVITY * DT * axis[1]) / (1 + ratio)
    loss = (ball.lin_damp + ratio * ball.rot_damp) / (1 + ratio) * DT
    assert a.terminal_speed == pytest.approx(b / loss)
    for n in (1, 17, 125, 400):
        p, v = reference(0., 2., loss, b, n)
        np.testing.assert_allclose(law.at(2 * axis, n), [p * axis, v * axis], atol=1e-9, rtol=0)
        assert a.over(2., float(p)) == pytest.approx((float(v), n * DT), abs=1e-9)
    uphill = cf.rolling(ball, nrm).along(axis)
    assert not uphill.climbs_from_rest()
    assert uphill.over(0., .1) is None
    assert uphill.over(2., 1000.) is None
    p, v = uphill.at(2., 3.)
    assert uphill.over(2., float(p)) == pytest.approx((float(v), 3 * DT))
    coast = cf.rolling(ball, [0, 1, 0]).along([1, 0, 0])
    assert coast.over(1., DT / coast.loss) is None
    p, v = coast.at(1., 25.5)
    assert coast.over(1., p) == pytest.approx((v, 25.5 * DT))


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
def test_rebound(ball):
    s = state(2, (2., -10., 1.))
    p = Params()
    r = cf.rebound(ball, s, [0, 3, 0], .3, p)
    np.testing.assert_allclose(r.vel, [2, 10 * math.sqrt(ball.elasticity * .3), 1])
    np.testing.assert_allclose(r.omega, s.omega)
    assert cf.flight(ball, r, 1).pos[1] > s.pos[1]
    assert cf.rebound(ball, state(2, (0., -.5, 0.)), [0, 1, 0], .3, p).vel[1] == 0
    assert cf.rebound(ball, state(2, (0., .5, 0.)), [0, 1, 0], .3, p).vel[1] == .5


@pytest.mark.parametrize('ball', BALLS.values(), ids=BALLS.keys())
def test_fan_formula_max_height_and_exit(ball):
    s = state()
    ts = np.arange(401)
    b = fans.rules().impulse / ball.mass - GRAVITY * DT
    a = cf.fan_flight(ball, s, ts)
    for k in (0, 1, 9, 80, 400):
        y, v = reference(s.pos[1], s.vel[1], ball.lin_damp * DT, b, k)
        assert a.pos[k, 1] == pytest.approx(y, abs=1e-9)
        assert a.vel[k, 1] == pytest.approx(v, abs=1e-9)
    assert cf.fan_max_height(ball, s, max_ticks=400) == pytest.approx(max(a.pos[:, 1]))
    assert math.isinf(cf.fan_max_height(ball, s)) == (ball == PAPER)
    assert cf.fan_terminal_speed(ball) == pytest.approx(b / (ball.lin_damp * DT))
    assert cf.fan_upper_boundary(ball, (100, 0)) is None
    top = cf.fan_upper_boundary(ball, (0, 0))
    exit_at = cf.fan_top_exit_step(ball, s, 0., (0, 0))
    if ball == PAPER:
        assert exit_at == np.flatnonzero(a.pos[:, 1] >= top)[0]
        after = cf.after_fan(ball, s, exit_at, 3)
        np.testing.assert_allclose(after.pos, cf.flight(ball, cf.fan_flight(ball, s, exit_at), 3).pos)
    else:
        assert exit_at is None
        assert cf.fan_rise_step(ball, s, cf.fan_max_height(ball, s) + .01) is None


def test_no_damping_limit():
    from dataclasses import replace
    ball = replace(STONE, lin_damp=0.)
    s = state(-3, (1., 2., 3.))
    p, v = reference(s.pos, s.vel, 0., [0, -GRAVITY * DT, 0], 40)
    np.testing.assert_allclose(cf.flight(ball, s, 40).pos, p, atol=1e-9)
    np.testing.assert_allclose(cf.flight(ball, s, 40).vel, v, atol=1e-9)
    assert cf.landing_step(ball, s, -5) == np.flatnonzero(cf.flight(ball, s, np.arange(100)).pos[:, 1] <= -5)[0]

