"""The physics kernel's tick, compiled.

`physics.step` describes one tick in plain Python; this is the same tick, operation for operation, fast enough to
run millions of ticks in a search. `tests/test_kernel.py` holds the two against each other on recorded engine
states. Nothing here adds a rule: every number comes in through the `Ball` and `Params` of `physics`.

`World` gathers what a tick needs from a mesh. `tick` advances one state (step, then `physics.settle`, the pair the
open-loop calibration measures). `rollout` holds a sequence of pushes and reports what the ball met on the way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .geometry import FACE, Mesh, _distance, _near

CAP = 64                # contacts looked at in one tick
OK, DEAD, TRIGGER, PLACED = 0, 1, 2, 3
LOW_CELL, LOW_SPAN = 8.0, 4     # a search bound, not physics: a ball lower than every surface within 32 to 40 units around is given up


@njit(cache=True)
def _contacts(px, py, pz, radius, margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
              ci, cd, cq, cf, cn, cg, order):
    """Merged contacts of a ball at p, nearest first, as `physics.merge_contacts(mesh.contacts(...))`: fills
    ci (triangle), cq (point), cf (feature), cn (normal), cg (gap) for the first m entries and returns m."""
    tickbox[0] += 1
    n = _near(px, py, pz, radius + margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, stamp, tickbox[0],
              ci, cd, cq, cf)
    for k in range(n):
        order[k] = k
    for a in range(1, n):                       # stable, by distance
        k = order[a]
        b = a - 1
        while b >= 0 and cd[order[b]] > cd[k]:
            order[b + 1] = order[b]
            b -= 1
        order[b + 1] = k
    m = 0
    for a in range(n):
        k = order[a]
        d, t = cd[k], ci[k]
        if d > 1e-12:
            ux, uy, uz = (px - cq[k, 0]) / d, (py - cq[k, 1]) / d, (pz - cq[k, 2]) / d
        else:
            ux, uy, uz = N[t, 0], N[t, 1], N[t, 2]
        f = cf[k]
        if f != FACE and abs(ux * N[t, 0] + uy * N[t, 1] + uz * N[t, 2]) > 1.0 - 1e-9:
            f = FACE                              # a seam between coplanar triangles
        same = False
        for o in range(m):
            dx, dy, dz = cq[order[o], 0] - cq[k, 0], cq[order[o], 1] - cq[k, 1], cq[order[o], 2] - cq[k, 2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-3:
                same = True
                break
            if f == FACE and cf[order[o]] == FACE and cn[order[o], 0] * ux + cn[order[o], 1] * uy + cn[order[o], 2] * uz > 1.0 - 1e-6:
                same = True
                break
        if same:
            continue
        cf[k] = f
        cn[k, 0], cn[k, 1], cn[k, 2] = ux, uy, uz
        cg[k] = d - radius
        order[m] = k                              # m <= a: the kept ones stay in front, in order
        m += 1
    return m


@njit(cache=True)
def _step(pos, vel, omega, push, ball, par, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el,
          stamp, tickbox, ci, cd, cq, cf, cn, cg, order, n_imp, t_imp, fr, el, target):
    """One tick in place (pos, vel, omega are updated). Returns (hit, number of merged contacts); the contacts are
    left in the work arrays, `order[:m]` indexing them. ball = (mass, push, lin_damp, rot_damp, friction, elasticity,
    radius, inertia); par = (reach, touch, floor, friction_scale, restitution_scale, bounce_speed, sweep,
    iterations, dt, gravity)."""
    mass, pushacc, lin_damp, rot_damp, b_fr, b_el, r, inertia = ball[0], ball[1], ball[2], ball[3], ball[4], ball[5], ball[6], ball[7]
    reach, touch, floor, f_scale, r_scale, bounce, sweep, iters, dt, grav = par[0], par[1], par[2], par[3], par[4], par[5], par[6], int(par[7]), par[8], par[9]
    mx, my, mz = vel[0] * dt, vel[1] * dt, vel[2] * dt
    qx, qy, qz = pos[0] + mx, pos[1] + my, pos[2] + mz
    hit = False
    if sweep != 0.0:
        d1, _t = _distance(qx, qy, qz, r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
        if d1 - r < floor:
            d0, _t = _distance(pos[0], pos[1], pos[2], r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
            if d0 - r > floor:
                lo, hi = 0.0, 1.0
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    dm, _t = _distance(pos[0] + mid * mx, pos[1] + mid * my, pos[2] + mid * mz, r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                    if dm - r > floor:
                        lo = mid
                    else:
                        hi = mid
                qx, qy, qz = pos[0] + lo * mx, pos[1] + lo * my, pos[2] + lo * mz
                hit = True
    pos[0], pos[1], pos[2] = qx, qy, qz
    k_lin = 1.0 - lin_damp * dt
    vx = vel[0] * k_lin + pushacc * push[0]
    vy = vel[1] * k_lin + pushacc * push[1] - grav * dt
    vz = vel[2] * k_lin + pushacc * push[2]
    k_rot = 1.0 - rot_damp * dt
    wx, wy, wz = omega[0] * k_rot, omega[1] * k_rot, omega[2] * k_rot
    m = _contacts(qx, qy, qz, r, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                  ci, cd, cq, cf, cn, cg, order)
    if m > 0:
        inv_m, inv_i = 1.0 / mass, 1.0 / inertia
        k_t = inv_m + r * r * inv_i
        for i in range(m):
            k = order[i]
            t = ci[k]
            fr[i] = b_fr * tri_fr[t] * f_scale
            el[i] = math.sqrt(b_el * tri_el[t]) * r_scale
            vn = vx * cn[k, 0] + vy * cn[k, 1] + vz * cn[k, 2]
            if cg[k] <= touch:
                target[i] = -el[i] * vn if vn < -bounce else 0.0
            else:
                target[i] = -(cg[k] - floor) / dt
            n_imp[i] = 0.0
            t_imp[i, 0] = t_imp[i, 1] = t_imp[i, 2] = 0.0
        for _ in range(iters if m > 1 else 1):
            for i in range(m):
                k = order[i]
                n0, n1, n2 = cn[k, 0], cn[k, 1], cn[k, 2]
                vn = vx * n0 + vy * n1 + vz * n2
                j = (target[i] - vn) * mass
                new = max(0.0, n_imp[i] + j)
                j = new - n_imp[i]
                n_imp[i] = new
                vx += j * inv_m * n0
                vy += j * inv_m * n1
                vz += j * inv_m * n2
                a0, a1, a2 = -r * n0, -r * n1, -r * n2
                u0 = vx + (wy * a2 - wz * a1)
                u1 = vy + (wz * a0 - wx * a2)
                u2 = vz + (wx * a1 - wy * a0)
                un = u0 * n0 + u1 * n1 + u2 * n2
                c0 = t_imp[i, 0] - (u0 - un * n0) / k_t
                c1 = t_imp[i, 1] - (u1 - un * n1) / k_t
                c2 = t_imp[i, 2] - (u2 - un * n2) / k_t
                lim = fr[i] * n_imp[i]
                size = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
                if size > lim and size > 0.0:
                    c0, c1, c2 = c0 * (lim / size), c1 * (lim / size), c2 * (lim / size)
                j0, j1, j2 = c0 - t_imp[i, 0], c1 - t_imp[i, 1], c2 - t_imp[i, 2]
                t_imp[i, 0], t_imp[i, 1], t_imp[i, 2] = c0, c1, c2
                vx += j0 * inv_m
                vy += j1 * inv_m
                vz += j2 * inv_m
                wx += (a1 * j2 - a2 * j1) * inv_i
                wy += (a2 * j0 - a0 * j2) * inv_i
                wz += (a0 * j1 - a1 * j0) * inv_i
    vel[0], vel[1], vel[2] = vx, vy, vz
    omega[0], omega[1], omega[2] = wx, wy, wz
    return hit, m


@njit(cache=True)
def _settle(pos, r, margin, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
            ci, cd, cq, cf, cn, cg, order):
    """`physics.settle` in place: out of the surfaces to the engine's contact margin."""
    for _ in range(4):
        m = _contacts(pos[0], pos[1], pos[2], r, reach, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N,
                      stamp, tickbox, ci, cd, cq, cf, cn, cg, order)
        moved = False
        for i in range(m):
            k = order[i]
            if cg[k] < margin - 1e-6:
                push = margin - cg[k]
                pos[0] += push * cn[k, 0]
                pos[1] += push * cn[k, 1]
                pos[2] += push * cn[k, 2]
                moved = True
        if not moved:
            break


@njit(cache=True)
def _rollout(pos, vel, omega, pushes, holds, ball, par, margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y,
             N, tri_fr, tri_el, tri_placed, tri_rail, boxes, spheres, y_min, low, low_ox, low_oz, low_cell, stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
             n_imp, t_imp, fr, el, target, trace, stats):
    """Hold pushes[k] for holds[k] ticks each, in place. Stops early when the ball's centre enters a death box or
    falls below everything near it (DEAD), enters a trigger sphere (TRIGGER) or touches a placed object (PLACED).
    Returns (status, ticks done). trace (T, 9), if not empty, gets after every tick: position, how many points
    it touches, whether one is an edge or corner, whether the tick began with a hit, speed, whether something
    carries it (a rail, or a face under it), whether it touches a rail.
    stats: [0] ticks that began with a hit, [1] ticks touching an edge or corner, [2] ticks touching two or more
    points, [3] ticks touching nothing, [4] top speed, [5] touching at the end (1/0), [6] the last touching
    contact's normal y (or -1), [8] carried at the end (on a rail, or a face under it)."""
    done = 0
    touch = par[1]
    r = ball[6]
    for k in range(len(holds)):
        for _ in range(holds[k]):
            hit, m = _step(pos, vel, omega, pushes[k], ball, par, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y,
                           N, tri_fr, tri_el, stamp, tickbox, ci, cd, cq, cf, cn, cg, order, n_imp, t_imp, fr, el, target)
            touching, edge, best_ny, carried, on_rail = 0, False, -1.0, False, False
            for i in range(m):
                c = order[i]
                if cg[c] <= touch:
                    touching += 1
                    if cf[c] != FACE:
                        edge = True
                    if tri_rail[ci[c]]:
                        on_rail = True
                    if tri_rail[ci[c]] or (cf[c] == FACE and cn[c, 1] >= 0.6):
                        carried = True            # on a rail, or a face under it
                    if cn[c, 1] > best_ny:
                        best_ny = cn[c, 1]
                    if tri_placed[ci[c]]:
                        return PLACED, done
            _settle(pos, r, margin, 0.2, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                    ci, cd, cq, cf, cn, cg, order)
            done += 1
            if hit:
                stats[0] += 1
            if edge:
                stats[1] += 1
            if touching >= 2:
                stats[2] += 1
            if touching == 0:
                stats[3] += 1
            speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
            if speed > stats[4]:
                stats[4] = speed
            stats[5] = 1.0 if touching > 0 else 0.0
            stats[6] = best_ny
            stats[8] = 1.0 if carried else 0.0
            if len(trace) > 0 and done <= len(trace):
                trace[done - 1, 0], trace[done - 1, 1], trace[done - 1, 2] = pos[0], pos[1], pos[2]
                trace[done - 1, 3], trace[done - 1, 4], trace[done - 1, 5] = touching, 1.0 if edge else 0.0, 1.0 if hit else 0.0
                trace[done - 1, 6] = speed
                trace[done - 1, 7], trace[done - 1, 8] = 1.0 if carried else 0.0, 1.0 if on_rail else 0.0
            if pos[1] < y_min:
                return DEAD, done
            gi, gj = int(math.floor((pos[0] - low_ox) / low_cell)), int(math.floor((pos[2] - low_oz) / low_cell))
            if gi < 0 or gj < 0 or gi >= low.shape[0] or gj >= low.shape[1] or pos[1] < low[gi, gj] - 3.0:
                return DEAD, done                 # below everything for LOW_SPAN around: it is falling out of the level
            for b in range(len(boxes)):
                if boxes[b, 0] <= pos[0] <= boxes[b, 3] and boxes[b, 1] <= pos[1] <= boxes[b, 4] and boxes[b, 2] <= pos[2] <= boxes[b, 5]:
                    return DEAD, done
            for s in range(len(spheres)):
                dx, dy, dz = pos[0] - spheres[s, 0], pos[1] - spheres[s, 1], pos[2] - spheres[s, 2]
                if dx * dx + dy * dy + dz * dz <= spheres[s, 3] * spheres[s, 3]:
                    return TRIGGER, done
    return OK, done


def ball_array(ball) -> np.ndarray:
    return np.array([ball.mass, ball.push, ball.lin_damp, ball.rot_damp, ball.friction, ball.elasticity, ball.radius, ball.inertia])


def params_array(P) -> np.ndarray:
    from .physics import DT, GRAVITY
    return np.array([P.reach, P.touch, P.floor, P.friction_scale, P.restitution_scale, P.bounce_speed, 1.0 if P.sweep else 0.0,
                     float(P.iterations), DT, GRAVITY])


@dataclass
class World:
    """A mesh prepared for the compiled tick, with what ends a rollout."""
    mesh: Mesh
    boxes: np.ndarray        # (M, 6) death boxes: lo x y z, hi x y z
    spheres: np.ndarray      # (S, 4) trigger spheres the ball must stay out of: x y z, distance
    y_min: float
    stop_on_placed: bool = True

    def __post_init__(self):
        m = self.mesh
        self.tri_fr = np.array([float(m.entities[o].get("friction", 0.7)) if m.entities else 0.7 for o in m.owner])
        self.tri_el = np.array([float(m.entities[o].get("elasticity", 0.3)) if m.entities else 0.3 for o in m.owner])
        self.tri_placed = m.placed.copy() if self.stop_on_placed else np.zeros(len(m.A), np.bool_)
        self.tri_rail = np.ascontiguousarray(m.rail, np.bool_)
        # the lowest surface within LOW_SPAN cells of each LOW_CELL cell: a ball below that is falling out of the level
        lo = np.minimum(np.minimum(m.A, m.B), m.C)
        hi = np.maximum(np.maximum(m.A, m.B), m.C)
        self.low_ox, self.low_oz = float(lo[:, 0].min()) - LOW_CELL * (LOW_SPAN + 1), float(lo[:, 2].min()) - LOW_CELL * (LOW_SPAN + 1)
        gx = int((hi[:, 0].max() - self.low_ox) / LOW_CELL) + LOW_SPAN + 2
        gz = int((hi[:, 2].max() - self.low_oz) / LOW_CELL) + LOW_SPAN + 2
        cellmin = np.full((gx, gz), np.inf)
        for a, b in zip(lo, hi):
            i0, i1 = int((a[0] - self.low_ox) / LOW_CELL), int((b[0] - self.low_ox) / LOW_CELL)
            j0, j1 = int((a[2] - self.low_oz) / LOW_CELL), int((b[2] - self.low_oz) / LOW_CELL)
            cellmin[i0:i1 + 1, j0:j1 + 1] = np.minimum(cellmin[i0:i1 + 1, j0:j1 + 1], a[1])
        self.low = np.full((gx, gz), np.inf)
        for i in range(gx):
            for j in range(gz):
                self.low[i, j] = cellmin[max(i - LOW_SPAN, 0):i + LOW_SPAN + 1, max(j - LOW_SPAN, 0):j + LOW_SPAN + 1].min()
        self.low[np.isinf(self.low)] = 1e18          # nothing anywhere near: out of the level
        self.stamp = np.zeros(len(m.A), np.int64)
        self.tickbox = np.zeros(1, np.int64)
        self.work = (np.empty(CAP, np.int64), np.empty(CAP), np.empty((CAP, 3)), np.empty(CAP, np.int64), np.empty((CAP, 3)),
                     np.empty(CAP), np.empty(CAP, np.int64))
        self.solve = (np.empty(CAP), np.empty((CAP, 3)), np.empty(CAP), np.empty(CAP), np.empty(CAP))

    @classmethod
    def of(cls, mesh: Mesh, death_boxes=(), keep_out=(), stop_on_placed: bool = True) -> "World":
        boxes = np.array([[*b["lo"], *b["hi"]] for b in death_boxes], np.float64).reshape(-1, 6)
        spheres = np.array([list(s) for s in keep_out], np.float64).reshape(-1, 4)
        y_min = float(mesh._lo_y.min()) - 30.0 if len(mesh.A) else -1e9
        return cls(mesh, boxes, spheres, y_min, stop_on_placed)

    def _geo(self):
        m = self.mesh
        return (*m._args(), m.N)

    def step(self, ball, P, pos, vel, omega, push):
        """One tick without settling, as `physics.step`: returns (pos, vel, omega, hit, merged contacts as
        (triangle, point, feature, normal, gap) rows, normal impulses)."""
        pos, vel, omega = np.array(pos, np.float64), np.array(vel, np.float64), np.array(omega, np.float64)
        hit, m = _step(pos, vel, omega, np.asarray(push, np.float64), ball_array(ball), params_array(P), *self._geo(),
                       self.tri_fr, self.tri_el, self.stamp, self.tickbox, *self.work, *self.solve)
        ci, cd, cq, cf, cn, cg, order = self.work
        rows = [(int(ci[k]), cq[k].copy(), int(cf[k]), cn[k].copy(), float(cg[k])) for k in order[:m]]
        return pos, vel, omega, bool(hit), rows, self.solve[0][:m].copy()

    def rollout(self, ball, P, pos, vel, omega, pushes, holds, trace: bool = False, margin: float | None = None):
        """Hold each push for its number of ticks. Returns (status, ticks, pos, vel, omega, stats, trace or None)."""
        from .geometry import CONTACT_MARGIN
        pos, vel, omega = np.array(pos, np.float64), np.array(vel, np.float64), np.array(omega, np.float64)
        pushes = np.ascontiguousarray(pushes, np.float64).reshape(-1, 3)
        holds = np.ascontiguousarray(holds, np.int64)
        tr = np.empty((int(holds.sum()) if trace else 0, 9))
        stats = np.zeros(10)
        status, done = _rollout(pos, vel, omega, pushes, holds, ball_array(ball), params_array(P),
                                CONTACT_MARGIN if margin is None else margin, *self._geo(), self.tri_fr, self.tri_el,
                                self.tri_placed, self.tri_rail, self.boxes, self.spheres, self.y_min, self.low, self.low_ox, self.low_oz, LOW_CELL, self.stamp, self.tickbox, *self.work,
                                *self.solve, tr, stats)
        return int(status), int(done), pos, vel, omega, stats, (tr[:done] if trace else None)
