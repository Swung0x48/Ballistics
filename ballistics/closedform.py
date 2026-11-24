"""Closed-form, constant-regime ball physics. Time arguments are in PSI ticks.

Positions move with the old velocity. Real tick arguments use the exponential
extension of the integer samples; the actual within-tick move is their chord.
No function advances a physics state tick by tick.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from . import fans
from .geometry import _closest, FACE
from .physics import DT, GRAVITY, KEY_RIGHT, KEY_UP, Ball, Params, State, key_push


@njit(cache=True)
def lambertw_real(x: float, branch: int = 0) -> float:
    """Real W0 or W-1, including their limiting values. Outside domain: NaN."""
    edge = -1.0 / math.e
    if branch not in (0, -1) or math.isnan(x) or x < edge or (branch == -1 and x > 0.0):
        return math.nan
    if x == edge:
        return -1.0
    if x == 0.0:
        return x if branch == 0 else -math.inf
    if x == math.inf:
        return x
    if branch == 0 and abs(x) < 1e-8:
        return x * (1.0 - x * (1.0 - 1.5 * x))
    if x < 0.0 and x - edge < 1e-6:
        # Invert the Taylor series at the double root, where log residuals
        # lose precision. p has the sign of W+1.
        p = math.sqrt(2.0 * math.e * (x - edge))
        if branch == -1:
            p = -p
        return -1.0 + p * (1.0 + p * (-1.0 / 3.0 + p * (11.0 / 72.0 + p * (-43.0 / 540.0 + p * 769.0 / 17280.0))))
    return _w_log(math.log(abs(x)), x < 0, branch)


@njit(cache=True)
def _w_log(logabs, negative, branch):
    """Solve w + log(abs(w)) = logabs without exponentiating large arguments."""
    if negative:
        if logabs > -1.0:
            return math.nan
        if logabs == -1.0:
            return -1.0
        if branch == 0 and logabs < -19.0:
            x = -math.exp(logabs)
            return x * (1.0 - x)
    elif logabs < -19.0:
        x = math.exp(logabs)
        return x * (1.0 - x)
    if not negative:
        lo, hi = 0.0, max(1.0, logabs)
    elif branch == 0:
        lo, hi = -1.0, 0.0
    else:
        lo, hi = 2.0 * logabs - 2.0, -1.0
    for _ in range(80):
        w = lo + (hi - lo) * 0.5
        if w == lo or w == hi:
            return w
        residual = w + math.log(abs(w)) - logabs
        if residual == 0.0:
            return w
        if (residual < 0.0) == (not negative or branch == -1):
            lo = w
        else:
            hi = w
    return lo + (hi - lo) * 0.5


def _coeff(n, loss):
    """d**n, S=sum(d**k), T=sum((1-d**k)/loss), k=0..n-1."""
    n = np.asarray(n, dtype=float)
    if np.any(~np.isfinite(n)) or np.any(n < 0) or not 0 <= loss < 1:
        raise ValueError("expected finite n >= 0 and 0 <= damping loss < 1")
    if loss == 0:
        return np.ones_like(n), n, n * (n - 1) / 2
    l = math.log1p(-loss)
    x = n * l
    s = -np.expm1(x) / loss
    # t=(n-s)/loss. Special-case the small argument of expm1(x)-x;
    # this also avoids subtracting two nearly identical O(n) quantities.
    if loss < 1e-3:
        delta = (-loss / 2 - loss**2 / 3 - loss**3 / 4 - loss**4 / 5
                 - loss**5 / 6 - loss**6 / 7)
    else:
        delta = 1 + l / loss
    with np.errstate(over="ignore", invalid="ignore"):
        series = (n * delta + (x / loss)**2 * loss *
                  (1/2 + x * (1/6 + x * (1/24 + x * (1/120 + x * (1/720 + x / 5040)))))) / loss
        t = np.where(abs(x) < 1e-3, series, (n - s) / loss)
    return np.exp(x), np.where(n == 1, 1.0, s), np.where(n == 1, 0.0, t)


def _motion(n, v0, impulse, loss):
    power, s, t = _coeff(n, loss)
    if np.ndim(v0) or np.ndim(impulse):
        power, s, t = power[..., None], s[..., None], t[..., None]
    return DT * (s * v0 + t * impulse), power * v0 + s * impulse


def _air_impulse(ball, keys, quarter, fan_count=0):
    if fan_count < 0 or int(fan_count) != fan_count:
        raise ValueError("fan_count must be a nonnegative integer")
    b = ball.push * key_push(keys, quarter)
    b[1] = (fan_count * fans.rules().impulse / ball.mass if fan_count else 0.0) - GRAVITY * DT
    return b


def flight(ball: Ball, start: State, n, keys=0, quarter=0) -> State:
    """Free flight at n (scalar or array); fixed keys and camera quarter."""
    return _flight(ball, start, n, _air_impulse(ball, keys, quarter))


def _flight(ball, start, n, b):
    dx, v = _motion(n, np.asarray(start.vel), b, ball.lin_damp * DT)
    spin = _coeff(n, ball.rot_damp * DT)[0][..., None] * start.omega
    return State(start.pos + dx, v, spin)


def _turn(v, b, loss):
    """Stationary point of the continuous position, if strictly after zero."""
    if loss == 0:
        n = 0.5 - v / b if b else 0.0
    else:
        l = math.log1p(-loss)
        # derivative = u + (v-u)*(-log(d)/loss)*d**n, u=b/loss
        den = (loss * v - b) * (-l / loss)
        n = (math.log(abs(b)) - math.log(abs(den))) / l if b * den < 0 else 0.0
    return max(0.0, n)


def _w_roots(v, b, loss, target):
    """Proposals for DT*[u*n + a*(1-exp(l*n))] = target."""
    if loss == 0:
        return ()  # monotone inversion handles the quadratic too
    u, l = b / loss, math.log1p(-loss)
    a = (v - u) / loss
    if a == 0:
        return (target / (DT * u),) if u else ()
    if u == 0:
        ratio = target / (DT * a)
        return (math.log1p(-ratio) / l,) if ratio < 1 else ()
    h = (target / DT - a) / u
    q = -l * a / u
    if not math.isfinite(h) or not math.isfinite(q) or q == 0:
        return ()
    logabs = math.log(abs(q)) + l * h
    if not math.isfinite(logabs):
        return ()
    return tuple(h - _w_log(logabs, q < 0, branch) / l for branch in ((0, -1) if q < 0 else (0,)))


def _increasing_root(v, b, loss, target, lo, hi=math.inf):
    """Invert a known increasing branch. None if target is not attained."""
    def residual(n):
        return float(_motion(n, v, b, loss)[0]) - target

    if residual(lo) >= 0:
        return lo
    if math.isinf(hi) and b <= 0 and (b < 0 or (loss and target >= DT * v / loss)):
        return None
    for guess in _w_roots(v, b, loss, target):
        if lo < guess < hi:
            r = residual(guess)
            if r == 0:
                return guess
            if r < 0:
                lo = guess
            else:
                hi = guess
    if math.isinf(hi):
        hi = max(1.0, lo + 1.0, lo * 2)
        while math.isfinite(hi) and residual(hi) < 0:
            hi *= 2
        if not math.isfinite(hi):
            return None
    elif residual(hi) < 0:
        return None
    for _ in range(100):
        mid = lo + (hi - lo) * 0.5
        if mid in (lo, hi):
            break
        r = residual(mid)
        if r == 0:
            return mid
        if r < 0:
            lo = mid
        else:
            hi = mid
    return hi


def _crossing_step(v, b, loss, target, reached=None):
    """First integer sample reaching an upper bound; zero if already there."""
    if reached is None:
        def reached(n):
            return _motion(n, v, b, loss)[0] >= target

    if reached(0):
        return 0
    if v <= 0 and b <= 0:
        return None
    turn = _turn(v, b, loss)
    lo, hi = (0.0, turn) if b < 0 else (turn, math.inf)
    if b < 0 and turn == 0:
        return None
    root = _increasing_root(v, b, loss, target, lo, hi)
    if root is None:
        # For absolute-coordinate comparisons a target at a rounded maximum
        # can equal an integer sample even when the relative root is absent.
        if b < 0:
            for n in (math.floor(turn), math.ceil(turn)):
                if reached(n):
                    return n
        return None
    # Near a maximum the continuous curve can hit while neither integer
    # endpoint does. Test the actual integer positions, not just ceil(root).
    k = math.floor(root)
    for n in (max(0, k - 1), k, k + 1):
        if reached(n):
            return n
    return None


def landing_step(ball: Ball, start: State, height: float) -> int | None:
    """First sampled centre height <= height; zero if initially at/below it."""
    v, b, loss = start.vel[1], -GRAVITY * DT, ball.lin_damp * DT
    return _crossing_step(-v, -b, loss, start.pos[1] - height,
                          lambda n: start.pos[1] + _motion(n, v, b, loss)[0] <= height)


@dataclass(frozen=True)
class ReachSquare:
    center: np.ndarray            # world x,z
    axes: np.ndarray              # rows: camera right and forward, in x,z
    half: float

    def contains(self, xz, *, atol=1e-12) -> bool:
        """Membership of the convex hull, NOT exact discrete key-sequence equality."""
        return bool(np.max(np.abs(self.axes @ (np.asarray(xz) - self.center))) <= self.half + atol)


def reach_square(ball: Ball, start: State, n: int, quarter=0) -> ReachSquare:
    """Convex hull of horizontal positions reachable with arbitrary held keys per tick."""
    if int(n) != n:
        raise ValueError("reach_square needs an integer n")
    xz = [0, 2]
    axes = np.array([key_push(k, quarter)[xz] for k in (KEY_RIGHT, KEY_UP)])
    half = DT * ball.push * float(_coeff(n, ball.lin_damp * DT)[2])
    return ReachSquare(flight(ball, start, n).pos[xz], axes, half)


def _unit(normal):
    n = np.asarray(normal, dtype=float)
    length = np.linalg.norm(n)
    if not length > 0 or not math.isfinite(length):
        raise ValueError("expected a finite nonzero normal/direction")
    return n / length


@dataclass(frozen=True)
class Hit:
    step: int                     # tick containing the contact (0 if already touching)
    fraction: float               # fraction of that tick's straight move
    center: np.ndarray
    point: np.ndarray
    normal: np.ndarray


def plane_hit(ball: Ball, start: State, point, normal, keys=0, quarter=0, *, radius=None) -> Hit | None:
    """First contact from the positive side of an oriented plane, including overlap.

    Pass radius=ball.radius+Params().floor for a skin rather than bare-sphere contact.
    The reported point is the projection of the centre onto the plane.
    """
    p, unit = np.asarray(point, float), _unit(normal)
    r = ball.radius if radius is None else radius
    gap = float((start.pos - p) @ unit) - r
    b = _air_impulse(ball, keys, quarter)
    k = _crossing_step(-float(start.vel @ unit), -float(b @ unit), ball.lin_damp * DT, gap,
                       lambda n: float((_flight(ball, start, n, b).pos - p) @ unit) <= r)
    if k is None:
        return None
    if k == 0:
        center, fraction = start.pos.copy(), 0.0
    else:
        s = _flight(ball, start, k - 1, b)
        left = float((s.pos - p) @ unit) - r
        travel = -DT * float(s.vel @ unit)
        fraction = float(np.clip(left / travel, 0, 1)) if travel > 0 else 1.0
        center = s.pos + fraction * DT * s.vel
    return Hit(k, fraction, center, center - ((center - p) @ unit) * unit, unit)


def _in_triangle(p, tri, normal):
    edges = np.roll(tri, -1, axis=0) - tri
    return bool(np.all((np.cross(edges, p - tri) @ normal) >= -1e-10))


def triangle_face_hit(ball: Ball, start: State, triangle, keys=0, quarter=0, *, radius=None) -> Hit | None:
    """Two-sided face query; an intersection outside the finite face is not a hit."""
    tri = np.asarray(triangle, float)
    normal = _unit(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    r = ball.radius if radius is None else radius
    b = _air_impulse(ball, keys, quarter)
    hits = []
    # Find incoming crossings on either side, including a return to the
    # starting side after a plane crossing outside the finite triangle.
    for unit in (normal, -normal):
        side = float((start.pos - tri[0]) @ unit)
        offset = 0
        if side < r:
            if abs(side) <= r:
                p = start.pos - side * unit
                if side >= 0 and _in_triangle(p, tri, normal):
                    return Hit(0, 0.0, start.pos.copy(), p, unit)
            # From below this offset plane, only a maximum can precede a
            # second (incoming) crossing. Start at the discrete maximum.
            v, impulse, loss = float(start.vel @ unit), float(b @ unit), ball.lin_damp * DT
            if impulse >= 0 or v <= 0:
                continue
            peak = _turn(v, impulse, loss)
            offset = max((math.floor(peak), math.ceil(peak)), key=lambda k: _motion(k, v, impulse, loss)[0])
            if _motion(offset, v, impulse, loss)[0] <= r - side:
                continue
        s = start if offset == 0 else _flight(ball, start, offset, b)
        hit = plane_hit(ball, s, tri[0], unit, keys, quarter, radius=r)
        if hit is not None and _in_triangle(hit.point, tri, normal):
            hits.append(Hit(offset + hit.step, hit.fraction, hit.center, hit.point, hit.normal))
    return min(hits, key=lambda h: (h.step, h.fraction)) if hits else None


def triangle_edge_samples(ball: Ball, start: State, triangle, ticks, keys=0, quarter=0, *, radius=None):
    """Boolean edge/vertex overlap at caller-selected ticks (sampled, not swept).

    Fractional ticks use the actual straight within-tick move. May miss
    contacts between samples; use triangle_face_hit for analytic face tests.
    """
    ticks = np.atleast_1d(ticks)
    if ticks.ndim != 1 or np.any(ticks < 0):
        raise ValueError("expected nonnegative sample ticks in a one-dimensional array")
    n = np.floor(ticks)
    s = flight(ball, start, n, keys, quarter)
    tri = np.asarray(triangle, float).ravel()
    r = ball.radius if radius is None else radius
    points = s.pos + ((ticks - n) * DT)[:, None] * s.vel
    out = []
    for p in points:
        x, y, z, feature = _closest(*p, *tri)
        out.append(feature != FACE and np.linalg.norm(p - (x, y, z)) <= r)
    return np.array(out, dtype=bool)


@dataclass(frozen=True)
class RollingLaw:
    """Pure rolling on a fixed plane, with constant input and no slip [derived]."""
    impulse: np.ndarray          # tangent COM velocity gain per tick
    loss: float                  # effective damping loss per tick

    @property
    def terminal_velocity(self):
        return self.impulse / self.loss

    def at(self, velocity, n):
        """(Tangent displacement, velocity) at n, with tangent initial velocity."""
        return _motion(n, np.asarray(velocity, float), self.impulse, self.loss)

    def along(self, direction):
        """Scalar law along a chosen uphill/travel axis (a projection, not arc length)."""
        return RollingAxis(float(self.impulse @ _unit(direction)), self.loss)


@dataclass(frozen=True)
class RollingAxis:
    impulse: float
    loss: float

    @property
    def terminal_speed(self):
        """Signed terminal velocity component."""
        return self.impulse / self.loss

    def climbs_from_rest(self) -> bool:
        return self.impulse > 0

    def at(self, speed: float, n):
        return _motion(n, speed, self.impulse, self.loss)

    def over(self, speed: float, distance: float) -> tuple[float, float] | None:
        """(Signed velocity, physics seconds) at first forward crossing of distance.

        Initial speed and target distance must be nonnegative. The result uses
        the continuous extension, not an integer step or the within-tick chord.
        """
        if speed < 0 or distance < 0:
            raise ValueError("expected nonnegative speed and distance")
        if distance == 0:
            return speed, 0.0
        if speed == 0 and self.impulse <= 0:
            return None
        hi = _turn(speed, self.impulse, self.loss) if self.impulse < 0 else math.inf
        n = _increasing_root(speed, self.impulse, self.loss, distance, 0.0, hi)
        return None if n is None else (float(self.at(speed, n)[1]), n * DT)


def rolling(ball: Ball, normal, keys=0, quarter=0) -> RollingLaw:
    """Project gravity and key impulse onto the plane; contact enforces no slip."""
    unit = _unit(normal)
    b = _air_impulse(ball, keys, quarter)
    inertia_ratio = ball.inertia / (ball.mass * ball.radius**2)
    effective_mass = 1 + inertia_ratio
    damping = (ball.lin_damp + inertia_ratio * ball.rot_damp) / effective_mass
    return RollingLaw((b - (b @ unit) * unit) / effective_mass, damping * DT)


def rebound(ball: Ball, at_impact: State, normal, surface_elasticity: float, params: Params | None = None) -> State:
    """Normal-only bounce. Input velocity is AFTER the landing tick's free update.

    No friction or spin impulse here; threshold is the existing assumption.
    """
    p, unit = params or Params(), _unit(normal)
    vn = float(at_impact.vel @ unit)
    e = math.sqrt(ball.elasticity * surface_elasticity) * p.restitution_scale if vn < -p.bounce_speed else 0.0
    return State(at_impact.pos.copy(), at_impact.vel - (1 + e) * min(vn, 0.0) * unit, at_impact.omega.copy())


def fan_flight(ball: Ball, start: State, n, keys=0, quarter=0, *, fan_count=1) -> State:
    """State after n ticks known to stay under the given number of fan impulses."""
    return _flight(ball, start, n, _air_impulse(ball, keys, quarter, fan_count))


def fan_rise_step(ball: Ball, start: State, height: float, *, fan_count=1) -> int | None:
    """First integer tick reaching height assuming the same fans remain active."""
    b = _air_impulse(ball, 0, 0, fan_count)[1]
    v, loss = start.vel[1], ball.lin_damp * DT
    return _crossing_step(v, b, loss, height - start.pos[1],
                          lambda n: start.pos[1] + _motion(n, v, b, loss)[0] >= height)


def fan_terminal_speed(ball: Ball, *, fan_count=1) -> float:
    """Signed vertical terminal velocity under sustained lift; not a height."""
    return float(_air_impulse(ball, 0, 0, fan_count)[1] / (ball.lin_damp * DT))


def fan_max_height(ball: Ball, start: State, *, fan_count=1, max_ticks: int | None = None) -> float:
    """Maximum sampled height (supremum for infinite duration); optionally cap active ticks."""
    b, loss, v = _air_impulse(ball, 0, 0, fan_count)[1], ball.lin_damp * DT, start.vel[1]
    if max_ticks is not None and (max_ticks < 0 or int(max_ticks) != max_ticks):
        raise ValueError("max_ticks must be a nonnegative integer")
    if max_ticks is None and b > 0:
        return math.inf
    if max_ticks is None and b == 0:
        return float(start.pos[1] + max(0.0, DT * v / loss))
    end = max_ticks or 0
    peak = _turn(v, b, loss) if b < 0 else 0.0
    if max_ticks is not None:
        peak = min(peak, end)
    return float(start.pos[1] + max(_motion(n, v, b, loss)[0] for n in (0, end, math.floor(peak), math.ceil(peak))))


def fan_upper_boundary(ball: Ball, local_xz, *, edge="best") -> float | None:
    """Upper centre-y boundary relative to the fan origin at fixed fan-local x,z.

    Uses the catalogue's spherical distance approximation, not the rotated
    mesh box or frame-level fan latch. None if the column misses the box.
    """
    half, _, top, reach, _ = fans.rules().params(ball, edge)
    outside = np.maximum(np.abs(np.asarray(local_xz, float)) - half, 0)
    remaining = reach**2 - outside @ outside
    return float(top + math.sqrt(remaining)) if remaining > 0 else None


def after_fan(ball: Ball, start: State, active_ticks: int, air_ticks, keys=0, quarter=0, *, fan_count=1) -> State:
    """Join two closed forms at a known box exit (last forced tick is included)."""
    return flight(ball, fan_flight(ball, start, active_ticks, keys, quarter, fan_count=fan_count), air_ticks, keys, quarter)


def fan_top_exit_step(ball: Ball, start: State, origin_y: float, local_xz, *, edge="best", fan_count=1) -> int | None:
    """First sample at/above the box top for an upright, fixed-x,z column.

    Caller establishes that the initial pose is inside and the active count
    is unchanged until this upper exit (no bottom/side exits or other boxes).
    At that sample no further impulse from this box is included by after_fan.
    """
    top = fan_upper_boundary(ball, local_xz, edge=edge)
    return None if top is None else fan_rise_step(ball, start, origin_y + top, fan_count=fan_count)
