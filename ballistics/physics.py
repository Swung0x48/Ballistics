"""The ball's physics, written by hand: a rigid sphere on the static collision triangles.

One step is one physics tick of the game (a PSI): 1/66 of a physics second, and the game runs two physics seconds
per game second. Velocities are in units per physics second, as the game holds them.

Per tick, in the order the game's own state shows:

1. The ball moves with the velocity it has: p += v / 66.
2. Its velocity changes: linear and angular damping, the key push (an impulse at the centre, horizontal, one unit
   push per held arrow key, so a diagonal is two pushes), gravity.
3. Where it touches something (exact geometry, at the new position), contact impulses act at the touching points:
   along the normal so that it does not go into the surface (with the pair's restitution when it arrives fast),
   and across it by friction, up to the pair's friction times the normal impulse. A surface that is near but not
   yet touched does nothing: the move of a later tick stops at it (below), and the bounce then acts on the full
   approach speed.

Hitting something inside a tick, as measured on the engine's landings (65 of them in one level 1 run): the ball's
move stops where its gap to the surface is 0.010, the tick's velocity update is applied first, and the ball leaves
with 0.245 of its approach speed along the normal whatever that speed is. 0.245 is the geometric mean of the two
elasticities (wood 0.2, floor and rail 0.3). Friction at the contact is the product of the two frictions (the
engine's own contact record says 0.8 x 0.7 = 0.56).

Rolling is not a separate rule. Friction at the contact point turns part of the push into spin, and the familiar
rolling behaviour falls out: a push accelerates the ball by F / (1.4 m), a slope by 5/7 of gravity along it. Two
rails, a kerb's edge and a wall are the same rule with other contact points.

Where each rule comes from (MEASURED on engine data, READ from the game, ASSUMED):

* move with the old velocity, then update it: MEASURED, exact in free flight (3,824 ticks, median error 0);
* damping, gravity 20, the tick of 1/66, the push of force / mass per held key: READ from the game's tables and
  MEASURED exact in free flight;
* the push direction per camera quarter (`key_push`): MEASURED exact for quarters 1 and 2 with the key sets UP,
  UP+LEFT, UP+RIGHT, RIGHT, DOWN+RIGHT (1,609 free-flight ticks); quarters 0 and 3 and the other key sets follow
  by the same formula but are NOT in the data;
* the tick's velocity update comes before a bounce: MEASURED (21 single-contact landings: restitution spreads
  0.2423 to 0.2449 that way, 0.2449 to 0.2659 the other way);
* rolling as a consequence of friction at the contact point: not a measurement but a derivation; what it predicts
  is MEASURED to agree (plateau speeds 15.9 and 22.5, one-tick error on the floor 0.008);
* contact impulses solved one point after another (sequential impulses): ASSUMED as a method; its agreement where
  several points touch (rails, kerbs, walls) is only as good as the calibration tables say;
* stone and paper balls: their table rows are READ, but nothing about them has been compared with the engine
  here, and the paper ball is not a sphere in the game.

Which of this agrees with the engine, and how closely, is measured by `ballistics.calibrate`."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import BALL_R, CONTACT_MARGIN, Mesh

PSI_HZ = 66.0           # physics ticks per physics second
DT = 1.0 / PSI_HZ
GRAVITY = 20.0          # units / physics second^2, along -y
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 1, 2, 4, 8     # arrow bits of a recorded input frame


@dataclass(frozen=True)
class Ball:
    name: str
    mass: float
    force: float          # impulse per tick and per held key (the game's Force Value): dv = force / mass
    lin_damp: float       # per physics second
    rot_damp: float
    friction: float
    elasticity: float
    radius: float = BALL_R

    @property
    def push(self) -> float:
        return self.force / self.mass

    @property
    def inertia(self) -> float:       # solid sphere
        return 0.4 * self.mass * self.radius * self.radius


# the game's Physicalize_GameBall rows (rotational damping from its physicalize records)
WOOD = Ball("wood", 1.9, 0.43, 0.9, 0.1, 0.8, 0.2)
PAPER = Ball("paper", 0.2, 0.065, 1.5, 0.1, 0.5, 0.4)     # the paper ball is a polyhedron in the game; as a sphere it is a rough stand-in
STONE = Ball("stone", 10.0, 0.92, 0.3, 0.1, 0.5, 0.1)
BALLS = {0: WOOD, 1: PAPER, 2: STONE}      # by the game's ball type number


def key_push(keys: int, quarter: int) -> np.ndarray:
    """World push direction (x, y, z) of the held arrow keys under a camera that is turned `quarter` quarter turns:
    UP pushes along the camera's forward, RIGHT along its right; two keys are two unit pushes added."""
    yaw = quarter * (math.pi / 2)
    f = (bool(keys & KEY_UP) - bool(keys & KEY_DOWN))
    r = (bool(keys & KEY_RIGHT) - bool(keys & KEY_LEFT))
    return np.array([f * math.sin(yaw) + r * math.cos(yaw), 0.0, f * math.cos(yaw) - r * math.sin(yaw)])


@dataclass
class Params:
    """What the rules leave open. Each value says where it comes from: MEASURED on engine data (level 1, one wood
    ball run unless said otherwise), READ from the game, or ASSUMED (not checked; nothing called a proof may rest
    on an assumed value alone)."""
    touch: float = 0.02            # MEASURED: a rolling ball's gap to its surface is 0.010 to 0.020 (57.8% of 20,483 frames), median 0.0141
    floor: float = 0.0099          # MEASURED: the smallest gap in 20,483 frames, and the gap at which every one of 65 landings was left
    friction_scale: float = 1.0    # READ: contact friction = ball friction x surface friction (the engine's own contact record: 0.8 x 0.7 = 0.56)
    restitution_scale: float = 1.0  # MEASURED: restitution = sqrt(ball elasticity x surface elasticity): 0.245 on 65 landings at approach speeds 2 to 21, wood on floor and rail
    bounce_speed: float = 1.0      # ASSUMED: below this approach speed there is no bounce. The data has no landing slower than 2, so the real threshold is unknown
    sweep: bool = True             # MEASURED: on a landing tick the ball is left where its gap is 0.010 (all 65 landings)
    iterations: int = 8            # ASSUMED: passes of the contact solver when several points touch; the engine's own solver is not known


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "State":
        return State(self.pos.copy(), self.vel.copy(), self.omega.copy())


def rolling_spin(vel, normal, radius: float = BALL_R) -> np.ndarray:
    """The spin of a ball that rolls without slipping on a surface with this normal at this velocity."""
    return np.cross(np.asarray(normal, float), np.asarray(vel, float)) / radius


def merge_contacts(contacts, tol: float = 1e-3) -> list:
    """One contact per touching point: triangles that meet at an edge or corner report the same point, and
    coplanar triangles report parallel face contacts beside each other."""
    out = []
    for c in contacts:
        same = False
        for o in out:
            if np.linalg.norm(o.point - c.point) < tol or (c.feature == "face" and o.feature == "face" and float(o.normal @ c.normal) > 1.0 - 1e-6):
                same = True
                break
        if not same:
            out.append(c)
    return out


def step(mesh: Mesh, ball: Ball, s: State, push, params: Params | None = None):
    """One tick. push: world push direction from `key_push` (zero vector for no keys). Returns (new state, info);
    info has the merged contacts used and the normal impulse on each. Runs the compiled tick of `kernel`, which is
    `step_reference` operation for operation (held together by tests/test_kernel.py)."""
    from .geometry import FEATURE, Contact
    from .kernel import World
    w = getattr(mesh, "_world", None)
    if w is None:
        w = mesh._world = World.of(mesh, stop_on_placed=False)
    P = params or Params()
    pos, vel, omega, hit, rows, n_imp = w.step(ball, P, s.pos, s.vel, s.omega, push)
    contacts = [Contact(t, q, float(np.linalg.norm(pos - q)), gap, FEATURE[f], n, int(mesh.owner[t])) for t, q, f, n, gap in rows]
    return State(pos, vel, omega), dict(contacts=contacts, normal_impulse=n_imp, hit=hit)


def step_reference(mesh: Mesh, ball: Ball, s: State, push, params: Params | None = None):
    """The tick written out in plain Python: the definition the compiled tick is checked against."""
    P = params or Params()
    r = ball.radius
    move = s.vel * DT
    pos = s.pos + move
    hit = False
    if P.sweep and mesh.distance(pos, r + 1.0)[0] - r < P.floor and mesh.distance(s.pos, r + 1.0)[0] - r > P.floor:
        lo, hi = 0.0, 1.0                        # the move would go into a surface: stop where the gap is the engine's minimum
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if mesh.distance(s.pos + mid * move, r + 1.0)[0] - r > P.floor:
                lo = mid
            else:
                hi = mid
        pos = s.pos + lo * move
        hit = True
    vel = s.vel * (1.0 - ball.lin_damp * DT) + ball.push * np.asarray(push, float) + np.array([0.0, -GRAVITY * DT, 0.0])
    omega = s.omega * (1.0 - ball.rot_damp * DT)
    contacts = [c for c in merge_contacts(mesh.contacts(pos, r, P.touch)) if c.gap <= P.touch]
    n_imp = np.zeros(len(contacts))
    if contacts:
        inv_m, inv_i = 1.0 / ball.mass, 1.0 / ball.inertia
        k_t = inv_m + r * r * inv_i                 # response of the contact point to a tangential impulse
        fr, el, target = [], [], []
        for c in contacts:
            e = mesh.entities[c.entity] if mesh.entities else {}
            fr.append(ball.friction * float(e.get("friction", 0.7)) * P.friction_scale)
            el.append(math.sqrt(ball.elasticity * float(e.get("elasticity", 0.3))) * P.restitution_scale)
            vn = float(vel @ c.normal)
            target.append(-el[-1] * vn if vn < -P.bounce_speed else 0.0)
        t_imp = np.zeros((len(contacts), 3))
        for _ in range(P.iterations if len(contacts) > 1 else 1):
            for i, c in enumerate(contacts):
                n = c.normal
                # normal: no approach beyond the target; the accumulated impulse only pushes
                vn = float(vel @ n)
                j = (target[i] - vn) * ball.mass
                new = max(0.0, n_imp[i] + j)
                j, n_imp[i] = new - n_imp[i], new
                vel = vel + j * inv_m * n
                # friction: stop the contact point's sliding, up to friction * normal impulse
                arm = -r * n
                u = vel + np.cross(omega, arm)
                ut = u - float(u @ n) * n
                jt = -ut / k_t
                acc = t_imp[i] + jt
                lim = fr[i] * n_imp[i]
                size = float(np.linalg.norm(acc))
                if size > lim:
                    acc = acc * (lim / size) if size > 0 else acc
                jt, t_imp[i] = acc - t_imp[i], acc
                vel = vel + jt * inv_m
                omega = omega + np.cross(arm, jt) * inv_i
    return State(pos, vel, omega), dict(contacts=contacts, normal_impulse=n_imp, hit=hit)


def settle(mesh: Mesh, s: State, radius: float = BALL_R, margin: float = CONTACT_MARGIN, reach: float = 0.2) -> State:
    """Keep the ball out of the surfaces it touches: where it is closer than the engine's contact margin, move it
    back out along the contact normal. Used when the kernel runs on its own positions for many ticks."""
    pos = s.pos.copy()
    for _ in range(4):
        moved = False
        for c in merge_contacts(mesh.contacts(pos, radius, reach)):
            if c.gap < margin - 1e-6:
                pos = pos + (margin - c.gap) * c.normal
                moved = True
        if not moved:
            break
    return State(pos, s.vel, s.omega)
