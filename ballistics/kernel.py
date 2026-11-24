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

from .geometry import FACE, Mesh, _closest, _distance, _near

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
def _fans_pushing(px, py, pz, fans, fanpar):
    """How many fans push a ball whose centre is at p (`fans.pushing`): its distance to the fan's box, in the fan's
    own axes, is under the ball kind's distance. fans (F, 12): origin, x, y, z axes; fanpar = (half, bottom, top,
    reach, impulse)."""
    half, bottom, top, reach = fanpar[0], fanpar[1], fanpar[2], fanpar[3]
    n = 0
    for k in range(len(fans)):
        dx, dy, dz = px - fans[k, 0], py - fans[k, 1], pz - fans[k, 2]
        lx = dx * fans[k, 3] + dy * fans[k, 4] + dz * fans[k, 5]
        ly = dx * fans[k, 6] + dy * fans[k, 7] + dz * fans[k, 8]
        lz = dx * fans[k, 9] + dy * fans[k, 10] + dz * fans[k, 11]
        ex, ey, ez = max(abs(lx) - half, 0.0), max(bottom - ly, ly - top, 0.0), max(abs(lz) - half, 0.0)
        if ex * ex + ey * ey + ez * ez < reach * reach:
            n += 1
    return n


@njit(cache=True)
def _step(pos, vel, omega, push, ball, par, fans, fanpar, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el,
          stamp, tickbox, ci, cd, cq, cf, cn, cg, order, n_imp, t_imp, fr, el, target):
    """One tick in place (pos, vel, omega are updated). Returns (hit, number of merged contacts); the contacts are
    left in the work arrays, `order[:m]` indexing them. ball = (mass, push, lin_damp, rot_damp, friction, elasticity,
    radius, inertia); par = (reach, touch, floor, friction_scale, restitution_scale, bounce_speed, sweep,
    iterations, dt, gravity); fans, fanpar: the fans that push (`_fans_pushing`), decided where the tick begins."""
    lift = _fans_pushing(pos[0], pos[1], pos[2], fans, fanpar) * fanpar[4] / ball[0] if len(fans) > 0 else 0.0
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
    vy = vel[1] * k_lin + pushacc * push[1] + lift - grav * dt      # damping, the fans' impulse, gravity: the game's order
    vz = vel[2] * k_lin + pushacc * push[2]
    k_rot = 1.0 - rot_damp * dt
    wx, wy, wz = omega[0] * k_rot, omega[1] * k_rot, omega[2] * k_rot
    m = _contacts(qx, qy, qz, r, touch, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
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
            target[i] = -el[i] * vn if vn < -bounce else 0.0
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


MCAP = 1024             # contacts of all the bodies together in one tick
MAX_VERTS, MAX_TRIS = 16, 48   # a piece that is a hull: at most this many corners and triangles (a near-flat face may come in more pieces)
VERT0 = 39                     # where a piece's hull corners start in its constants (about its centre, as it was put)
TRI0 = VERT0 + 3 * MAX_VERTS   # where its triangles start (the same)
PROP_WIDTH = TRI0 + 9 * MAX_TRIS
BC_WIDTH = 40                  # a body's row in the tick: ...; [29:38] its turn from how it was put (a hinged body), [38] the angle, [39] hinged
STILL_TICKS = 20        # a loose body that has not moved for this many ticks is frozen (0.3 physics s: the game's short test)
STILL_SPEED = 0.05
CORNER = 0.05           # a box touches by its corners: each is a little sphere of this radius
HULL_CORNER = 0.0       # a piece with a hull's own corners (a hinged board, a stopper) touches by those points (near a hinge a corner's size is degrees)


@njit(cache=True)
def _box_point(px, py, pz, c, h, rad):
    """A little sphere of radius rad at p against the box c (bc row: centre in c[21:24], half extents c[9:12], its
    axes in world in c[12:21]): (gap, normal x, y, z), the normal pointing from the box to p."""
    dx, dy, dz = px - c[21], py - c[22], pz - c[23]
    l0 = dx * c[12] + dy * c[13] + dz * c[14]
    l1 = dx * c[15] + dy * c[16] + dz * c[17]
    l2 = dx * c[18] + dy * c[19] + dz * c[20]
    q0, q1, q2 = min(max(l0, -h[0]), h[0]), min(max(l1, -h[1]), h[1]), min(max(l2, -h[2]), h[2])
    e0, e1, e2 = l0 - q0, l1 - q1, l2 - q2
    d = math.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
    if d > 1e-12:
        n0, n1, n2, gap = e0 / d, e1 / d, e2 / d, d - rad
    else:                                                # inside: out through the nearest face
        m0, m1, m2 = h[0] - abs(l0), h[1] - abs(l1), h[2] - abs(l2)
        n0 = n1 = n2 = 0.0
        if m0 <= m1 and m0 <= m2:
            n0, gap = (1.0 if l0 >= 0 else -1.0), -m0 - rad
        elif m1 <= m2:
            n1, gap = (1.0 if l1 >= 0 else -1.0), -m1 - rad
        else:
            n2, gap = (1.0 if l2 >= 0 else -1.0), -m2 - rad
    return (gap, n0 * c[12] + n1 * c[15] + n2 * c[18], n0 * c[13] + n1 * c[16] + n2 * c[19], n0 * c[14] + n1 * c[17] + n2 * c[20])


@njit(cache=True)
def _corner(c, k):
    """The k-th corner (0..7) of the box c (a bc row), in world."""
    s0 = c[9] if k & 1 else -c[9]
    s1 = c[10] if k & 2 else -c[10]
    s2 = c[11] if k & 4 else -c[11]
    return (c[21] + s0 * c[12] + s1 * c[15] + s2 * c[18], c[22] + s0 * c[13] + s1 * c[16] + s2 * c[19], c[23] + s0 * c[14] + s1 * c[17] + s2 * c[20])


@njit(cache=True)
def _box_over_ground(c, x, y, z, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """Whether there is a surface straight under the middle of the box's underside (within 1.5 below it), with the
    box's centre at (x, y, z): on a floor or a slope yes, with its middle out over an edge no."""
    bx, by, bz = x - c[10] * c[15], y - c[10] * c[16], z - c[10] * c[17]
    for k in range(7):
        d = 0.25 * k
        dist, _t = _distance(bx - d * c[15], by - d * c[16], bz - d * c[17], 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
        if dist < 0.2:
            return True
    return False


@njit(cache=True)
def _face_axis(cs, cx):
    """For the box cs against the box cx (bc rows): (k, side): the axis of cx along which cs lies off it most, counted
    in their sizes, and on which side. Two boxes touch across that face of cx: one stands on the other across the
    upright axis, one pushes the other across a level one."""
    dx, dy, dz = cs[21] - cx[21], cs[22] - cx[22], cs[23] - cx[23]
    best, kbest, side = -1.0, 0, 1.0
    for i in range(3):
        a0, a1, a2 = cx[12 + 3 * i], cx[13 + 3 * i], cx[14 + 3 * i]
        along = dx * a0 + dy * a1 + dz * a2
        reach = 0.0
        for j in range(3):
            reach += abs(cs[12 + 3 * j] * a0 + cs[13 + 3 * j] * a1 + cs[14 + 3 * j] * a2) * cs[9 + j]
        ratio = abs(along) / (cx[9 + i] + reach)
        if ratio > best:
            best, kbest, side = ratio, i, (1.0 if along >= 0.0 else -1.0)
    return kbest, side


@njit(cache=True)
def _corner_face(px, py, pz, cx, k, side):
    """A corner at p against the face (k, side) of the box cx: (gap, normal x, y, z), the normal out of that face;
    the gap is 1e9 when the corner is not over the face."""
    dx, dy, dz = px - cx[21], py - cx[22], pz - cx[23]
    out = 0.0
    for i in range(3):
        along = dx * cx[12 + 3 * i] + dy * cx[13 + 3 * i] + dz * cx[14 + 3 * i]
        if i == k:
            out = side * along - cx[9 + i] - CORNER
        elif abs(along) > cx[9 + i] + CORNER:
            return (1e9, 0.0, 0.0, 0.0)
    return (out, side * cx[12 + 3 * k], side * cx[13 + 3 * k], side * cx[14 + 3 * k])


@njit(cache=True)
def _box_gap(c, x, y, z, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y):
    """How far the nearest corner of the box c (a bc row) is from any surface, with its centre at (x, y, z)."""
    worst = 1e9
    for k in range(8):
        s0 = c[9] if k & 1 else -c[9]
        s1 = c[10] if k & 2 else -c[10]
        s2 = c[11] if k & 4 else -c[11]
        d, _t = _distance(x + s0 * c[12] + s1 * c[15] + s2 * c[18], y + s0 * c[13] + s1 * c[16] + s2 * c[19], z + s0 * c[14] + s1 * c[17] + s2 * c[20],
                          CORNER + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
        if d - CORNER < worst:
            worst = d - CORNER
    return worst


@njit(cache=True)
def _hull_point(px, py, pz, c, prow, rad):
    """A sphere of radius rad at p against a piece that is a convex hull (its triangles about the centre c[21:24]
    in prow[TRI0:], prow[23] of them, as it was put; c[29:38] turns that pose into the one it has now): (gap,
    normal x, y, z), the normal pointing from the hull to p."""
    wx, wy, wz = px - c[21], py - c[22], pz - c[23]
    qx = c[29] * wx + c[32] * wy + c[35] * wz            # back into the pose it was put in (the turn's transpose)
    qy = c[30] * wx + c[33] * wy + c[36] * wz
    qz = c[31] * wx + c[34] * wy + c[37] * wz
    best, bx, by, bz = 1e300, 0.0, 0.0, 0.0
    inside, depth, dn0, dn1, dn2 = True, 1e300, 0.0, 0.0, 0.0
    for t in range(int(prow[23])):
        o = TRI0 + 9 * t
        ax, ay, az = prow[o], prow[o + 1], prow[o + 2]
        bbx, bby, bbz = prow[o + 3], prow[o + 4], prow[o + 5]
        ccx, ccy, ccz = prow[o + 6], prow[o + 7], prow[o + 8]
        x, y, z, _f = _closest(qx, qy, qz, ax, ay, az, bbx, bby, bbz, ccx, ccy, ccz)
        d2 = (qx - x) ** 2 + (qy - y) ** 2 + (qz - z) ** 2
        if d2 < best:
            best, bx, by, bz = d2, x, y, z
        ux, uy, uz = bbx - ax, bby - ay, bbz - az
        vx, vy, vz = ccx - ax, ccy - ay, ccz - az
        n0, n1, n2 = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        nl = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        if nl < 1e-12:
            continue
        n0, n1, n2 = n0 / nl, n1 / nl, n2 / nl
        if n0 * ax + n1 * ay + n2 * az < 0.0:            # outward: away from the centre, which is inside the hull
            n0, n1, n2 = -n0, -n1, -n2
        s = n0 * (qx - ax) + n1 * (qy - ay) + n2 * (qz - az)
        if s > 0.0:
            inside = False
        elif -s < depth:
            depth, dn0, dn1, dn2 = -s, n0, n1, n2
    if not inside:
        d = math.sqrt(best)
        if d > 1e-12:
            dn0, dn1, dn2, gap = (qx - bx) / d, (qy - by) / d, (qz - bz) / d, d - rad
        else:
            gap = -rad
    else:
        gap = -depth - rad
    return (gap, c[29] * dn0 + c[30] * dn1 + c[31] * dn2, c[32] * dn0 + c[33] * dn1 + c[34] * dn2, c[35] * dn0 + c[36] * dn1 + c[37] * dn2)


@njit(cache=True)
def _vert(c, prow, k):
    """The k-th corner of a piece, in world: its hull's corners where it has them (prow[38] of them), else its box's eight."""
    if prow[38] > 0.0:
        o = VERT0 + 3 * k
        vx, vy, vz = prow[o], prow[o + 1], prow[o + 2]
        return (c[21] + c[29] * vx + c[30] * vy + c[31] * vz, c[22] + c[32] * vx + c[33] * vy + c[34] * vz, c[23] + c[35] * vx + c[36] * vy + c[37] * vz)
    return _corner(c, k)


@njit(cache=True)
def _shape_point(px, py, pz, c, prow, rad):
    """A little sphere at p against a piece: its hull, or its box."""
    if prow[23] > 0.0:
        return _hull_point(px, py, pz, c, prow, rad)
    return _box_point(px, py, pz, c, c[9:12], rad)


@njit(cache=True)
def _turn(ax, ay, az, th, R):
    """R (9, row-major) = the turn by th about the unit axis a (Rodrigues)."""
    s, cth = math.sin(th), math.cos(th)
    v = 1.0 - cth
    R[0], R[1], R[2] = cth + ax * ax * v, ax * ay * v - az * s, ax * az * v + ay * s
    R[3], R[4], R[5] = ay * ax * v + az * s, cth + ay * ay * v, ay * az * v - ax * s
    R[6], R[7], R[8] = az * ax * v - ay * s, az * ay * v + ax * s, cth + az * az * v


@njit(cache=True)
def _hinge_angle(props, k, px, py, pz):
    """The angle a hinged piece has turned from where it was put, from where its centre is now."""
    p = props[k]
    ax, ay, az = p[28], p[29], p[30]
    u0, u1, u2 = p[32] - p[25], p[33] - p[26], p[34] - p[27]
    w0, w1, w2 = px - p[25], py - p[26], pz - p[27]
    d = u0 * ax + u1 * ay + u2 * az
    u0, u1, u2 = u0 - d * ax, u1 - d * ay, u2 - d * az
    d = w0 * ax + w1 * ay + w2 * az
    w0, w1, w2 = w0 - d * ax, w1 - d * ay, w2 - d * az
    cr0, cr1, cr2 = u1 * w2 - u2 * w1, u2 * w0 - u0 * w2, u0 * w1 - u1 * w0
    return math.atan2(cr0 * ax + cr1 * ay + cr2 * az, u0 * w0 + u1 * w1 + u2 * w2)


@njit(cache=True)
def _hinge_pose(bc, bp, bv, bw, props, nb, lead, th, om):
    """Put every piece of the hinged body whose first piece is `lead` at angle th, turning at om about its axis."""
    p = props[int(bc[lead, 6])]
    ax, ay, az = p[28], p[29], p[30]
    R = np.empty(9)
    _turn(ax, ay, az, th, R)
    for s in range(1, nb):
        if bc[s, 25] != float(lead):
            continue
        q = props[int(bc[s, 6])]
        r0, r1, r2 = q[32] - q[25], q[33] - q[26], q[34] - q[27]
        cx = q[25] + R[0] * r0 + R[1] * r1 + R[2] * r2
        cy = q[26] + R[3] * r0 + R[4] * r1 + R[5] * r2
        cz = q[27] + R[6] * r0 + R[7] * r1 + R[8] * r2
        bp[s, 0], bp[s, 1], bp[s, 2] = cx, cy, cz
        bc[s, 21], bc[s, 22], bc[s, 23] = cx, cy, cz
        for i in range(3):                                # the box's axes, turned (props [10:19]: the axes as it was put)
            e0, e1, e2 = q[10 + 3 * i], q[11 + 3 * i], q[12 + 3 * i]
            bc[s, 12 + 3 * i] = R[0] * e0 + R[1] * e1 + R[2] * e2
            bc[s, 13 + 3 * i] = R[3] * e0 + R[4] * e1 + R[5] * e2
            bc[s, 14 + 3 * i] = R[6] * e0 + R[7] * e1 + R[8] * e2
        for i in range(9):
            bc[s, 29 + i] = R[i]
        bc[s, 38] = th
        dx, dy, dz = cx - q[25], cy - q[26], cz - q[27]
        bw[s, 0], bw[s, 1], bw[s, 2] = om * ax, om * ay, om * az
        bv[s, 0], bv[s, 1], bv[s, 2] = om * (ay * dz - az * dy), om * (az * dx - ax * dz), om * (ax * dy - ay * dx)


@njit(cache=True)
def _pvel(x, px, py, pz, bp, bv, bw, bc, props):
    """The velocity of body x at the point p: a hinged body turns about its axis, a ball also spins, a box only moves."""
    if bc[x, 39] != 0.0:
        q = props[int(bc[x, 6])]
        om = bw[x, 0] * q[28] + bw[x, 1] * q[29] + bw[x, 2] * q[30]
        dx, dy, dz = px - q[25], py - q[26], pz - q[27]
        return (om * (q[29] * dz - q[30] * dy), om * (q[30] * dx - q[28] * dz), om * (q[28] * dy - q[29] * dx))
    if bc[x, 8] == 0.0:
        dx, dy, dz = px - bp[x, 0], py - bp[x, 1], pz - bp[x, 2]
        return (bv[x, 0] + bw[x, 1] * dz - bw[x, 2] * dy, bv[x, 1] + bw[x, 2] * dx - bw[x, 0] * dz, bv[x, 2] + bw[x, 0] * dy - bw[x, 1] * dx)
    return (bv[x, 0], bv[x, 1], bv[x, 2])


@njit(cache=True)
def _kdir(x, px, py, pz, d0, d1, d2, bp, bc, props):
    """How much body x gives at the point p along the direction d, per unit impulse (its inverse mass there)."""
    if bc[x, 39] != 0.0:
        q = props[int(bc[x, 6])]
        dx, dy, dz = px - q[25], py - q[26], pz - q[27]
        j = (dy * d2 - dz * d1) * q[28] + (dz * d0 - dx * d2) * q[29] + (dx * d1 - dy * d0) * q[30]
        return j * j / q[31]
    k = 1.0 / bc[x, 0]
    if bc[x, 8] == 0.0:
        dx, dy, dz = px - bp[x, 0], py - bp[x, 1], pz - bp[x, 2]
        c0, c1, c2 = dy * d2 - dz * d1, dz * d0 - dx * d2, dx * d1 - dy * d0
        return k + (c0 * c0 + c1 * c1 + c2 * c2) / bc[x, 5]
    if bc[x, 24] != 0.0:
        k *= (d0 * bc[x, 15] + d1 * bc[x, 16] + d2 * bc[x, 17]) ** 2
    return k


@njit(cache=True)
def _impulse(x, sign, px, py, pz, j0, j1, j2, bp, bv, bw, bc, props, nb):
    """Give body x the impulse sign * j at the point p."""
    j0, j1, j2 = sign * j0, sign * j1, sign * j2
    if bc[x, 39] != 0.0:
        q = props[int(bc[x, 6])]
        dx, dy, dz = px - q[25], py - q[26], pz - q[27]
        om = bw[x, 0] * q[28] + bw[x, 1] * q[29] + bw[x, 2] * q[30]
        om += ((dy * j2 - dz * j1) * q[28] + (dz * j0 - dx * j2) * q[29] + (dx * j1 - dy * j0) * q[30]) / q[31]
        _hinge_pose(bc, bp, bv, bw, props, nb, int(bc[x, 25]), bc[x, 38], om)
        return
    m = bc[x, 0]
    bv[x, 0] += j0 / m
    bv[x, 1] += j1 / m
    bv[x, 2] += j2 / m
    if bc[x, 8] == 0.0:
        dx, dy, dz = px - bp[x, 0], py - bp[x, 1], pz - bp[x, 2]
        bw[x, 0] += (dy * j2 - dz * j1) / bc[x, 5]
        bw[x, 1] += (dz * j0 - dx * j2) / bc[x, 5]
        bw[x, 2] += (dx * j1 - dy * j0) / bc[x, 5]
    if bc[x, 24] != 0.0:
        along = bv[x, 0] * bc[x, 15] + bv[x, 1] * bc[x, 16] + bv[x, 2] * bc[x, 17]
        bv[x, 0], bv[x, 1], bv[x, 2] = along * bc[x, 15], along * bc[x, 16], along * bc[x, 17]
    _sync(bv, bw, bc, nb, x)


@njit(cache=True)
def _pair_ok(bc, a, b):
    """Whether two bodies of the tick touch at all: not two pieces of one body; a fixed part of a mechanism (role 2)
    only with the parts of that mechanism that move and do not touch the level (role 1)."""
    la = bc[a, 25]
    if la >= 0.0 and la == bc[b, 25]:
        return False
    ra, rb = bc[a, 26], bc[b, 26]
    if ra == 4.0 or rb == 4.0:                            # a stopper (group Ball): stops mechanisms and props, not the player ball (same group)
        o = b if ra == 4.0 else a
        return o != 0 and bc[o, 26] != 2.0 and bc[o, 26] != 4.0
    if ra == 1.0 and rb == 1.0:
        return False                                      # two parts in the group Floor: the same group does not collide (READ)
    if ra == 2.0 or rb == 2.0:
        return ra + rb == 3.0 and bc[a, 27] == bc[b, 27]
    return True


@njit(cache=True)
def _sync(bv, bw, bc, nb, b):
    """The pieces of one body move as one: give b's velocity to the others."""
    lead = bc[b, 25]
    if lead < 0.0:
        return
    for s in range(1, nb):
        if s != b and bc[s, 25] == lead:
            for q in range(3):
                bv[s, q], bw[s, q] = bv[b, q], bw[b, q]


@njit(cache=True)
def _shift(bp, bc, nb, b, dx, dy, dz):
    """Move body b (and the other pieces of its body) by d."""
    lead = bc[b, 25]
    for s in range(nb):
        if s == b or (lead >= 0.0 and bc[s, 25] == lead):
            bp[s, 0] += dx
            bp[s, 1] += dy
            bp[s, 2] += dz
            bc[s, 21], bc[s, 22], bc[s, 23] = bp[s, 0], bp[s, 1], bp[s, 2]


@njit(cache=True)
def _multi_tick(pos, vel, omega, push, ball, par, fans, fanpar, props, scene, margin, y_min,
                ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
                stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
                bp, bv, bw, bc, m_a, m_b, m_n, m_fr, m_target, m_nimp, m_timp, m_p, out):
    """One tick of the player ball together with the loose bodies (the tick of `_step`, for several bodies at once).
    Every body moves with the velocity it has and has its velocity updated as in `_step`; then all contacts, body
    on mesh and body on body, are solved together, one after another, a few times over: a ball struck by a rolling
    ball is pressed onto the floor by the friction between the two, and the floor then drags on it.
    A loose body is a sphere, or a box that slides and does not turn (it touches by its eight corners).
    props (K, 20): mass, friction, elasticity, linear damping, angular damping, inertia, shape (0 sphere, 1 box),
    half extents, the box's three axes in world, 1 if it is on a rail (it only slides along its own y axis), then
    [20] the first piece of its body (-1: a body of one piece), [21] its role (0 ordinary; 1 a part that does not touch
    the level's mesh; 2 fixed, only with its mechanism's role 1 parts; 3 fixed in its opening pose; 4 a stopper: it touches every
    body but the player ball), [22] which mechanism, [23] how many triangles its hull has (0: it is the box), [24] 1
    if it turns about a fixed hinge, [25:28] the hinge's point, [28:31] its axis, [31] the inertia about it, [32:35]
    the piece's centre as it was put, [35:38] the body's mass centre as it was put, [38] how many hull corners, then
    the corners (VERT0) and the triangles (TRI0) about its centre as it was put. The pieces of one body each carry
    the body's mass and move as one; a hinged body's state is its angle (read from where its first piece is) and
    its angular velocity. scene (K, 11), in place: position, velocity, angular velocity,
    flag (0 frozen, 1 awake, 2 gone), ticks it has been still.
    out: [0] the player's tick began with a hit, [1] how many points the player touches, [2] one is an edge or
    corner, [3] the best normal y of its mesh contacts, [4] carried, [5] on a rail, [6] it touches a placed object."""
    r = ball[6]
    reach, touch, floor, f_scale, r_scale, bounce, sweep, iters, dt, grav = par[0], par[1], par[2], par[3], par[4], par[5], par[6], int(par[7]), par[8], par[9]
    K = len(props)
    nb = 1
    # ---- the bodies: 0 the player, then every loose body that is still there
    bp[0, 0], bp[0, 1], bp[0, 2] = pos[0], pos[1], pos[2]
    bv[0, 0], bv[0, 1], bv[0, 2] = vel[0], vel[1], vel[2]
    bw[0, 0], bw[0, 1], bw[0, 2] = omega[0], omega[1], omega[2]
    bc[0, 0], bc[0, 1], bc[0, 2], bc[0, 3], bc[0, 4], bc[0, 5], bc[0, 6], bc[0, 7], bc[0, 8] = ball[0], ball[4], ball[5], ball[2], ball[3], ball[7], -1.0, 1.0, 0.0
    bc[0, 24] = 0.0
    bc[0, 25], bc[0, 26], bc[0, 27] = -1.0, 0.0, 0.0
    for k in range(K):
        if scene[k, 9] == 2.0:
            continue
        for q in range(3):
            bp[nb, q], bv[nb, q], bw[nb, q] = scene[k, q], scene[k, 3 + q], scene[k, 6 + q]
        for q in range(6):
            bc[nb, q] = props[k, q]
        bc[nb, 6], bc[nb, 7] = float(k), scene[k, 9]                    # which loose body; awake (1) or frozen (0)
        for q in range(13):
            bc[nb, 8 + q] = props[k, 6 + q]                             # shape, half extents, axes
        bc[nb, 24] = props[k, 19]                                       # 1: it only slides along its own y axis (a rail)
        bc[nb, 25], bc[nb, 26], bc[nb, 27] = -1.0, props[k, 21], props[k, 22]
        bc[nb, 29], bc[nb, 30], bc[nb, 31], bc[nb, 32], bc[nb, 33], bc[nb, 34], bc[nb, 35], bc[nb, 36], bc[nb, 37] = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        bc[nb, 38], bc[nb, 39] = 0.0, props[k, 24]
        nb += 1
    for b in range(1, nb):                                              # the first piece of each body, as a body of this tick
        lk = props[int(bc[b, 6]), 20]
        if lk >= 0.0:
            for s in range(1, nb):
                if bc[s, 6] == lk:
                    bc[b, 25] = float(s)
    for b in range(1, nb):                                              # a hinged body: its pose from its angle
        if bc[b, 39] != 0.0 and bc[b, 25] == float(b):
            k = int(bc[b, 6])
            om = scene[k, 6] * props[k, 28] + scene[k, 7] * props[k, 29] + scene[k, 8] * props[k, 30]
            _hinge_pose(bc, bp, bv, bw, props, nb, b, _hinge_angle(props, k, bp[b, 0], bp[b, 1], bp[b, 2]), om)
    lift = _fans_pushing(pos[0], pos[1], pos[2], fans, fanpar) * fanpar[4] / ball[0] if len(fans) > 0 else 0.0
    out[0] = 0.0
    # ---- move, then the velocity update (a frozen body does neither: nothing acts on it until it is struck)
    for b in range(1, nb):
        if bc[b, 39] == 0.0 or bc[b, 25] != float(b) or bc[b, 7] == 0.0:
            continue
        # a hinged body turns with the angular velocity it has; then gravity pulls on its mass centre, and its damping
        # (ASSUMED: the linear damping on the mass centre's speed, the angular one on the turn about the mass centre)
        q = props[int(bc[b, 6])]
        ax, ay, az, I = q[28], q[29], q[30], q[31]
        om = bw[b, 0] * ax + bw[b, 1] * ay + bw[b, 2] * az
        th = bc[b, 38] + om * dt
        Rm = np.empty(9)
        _turn(ax, ay, az, th, Rm)
        g0, g1, g2 = q[35] - q[25], q[36] - q[26], q[37] - q[27]
        c0 = Rm[0] * g0 + Rm[1] * g1 + Rm[2] * g2
        c2 = Rm[6] * g0 + Rm[7] * g1 + Rm[8] * g2
        c1 = Rm[3] * g0 + Rm[4] * g1 + Rm[5] * g2
        mg = q[0] * grav
        om += mg * (c2 * ax - c0 * az) / I * dt
        d = c0 * ax + c1 * ay + c2 * az
        perp2 = (c0 - d * ax) ** 2 + (c1 - d * ay) ** 2 + (c2 - d * az) ** 2
        kd = (q[3] * q[0] * perp2 + q[4] * max(I - q[0] * perp2, 0.0)) / I
        om *= 1.0 - kd * dt
        _hinge_pose(bc, bp, bv, bw, props, nb, b, th, om)
    for b in range(nb):
        if bc[b, 7] == 0.0 or bc[b, 26] >= 2.0 or bc[b, 39] != 0.0:     # (a fixed part never moves; a hinged one is done)
            continue
        mx, my, mz = bv[b, 0] * dt, bv[b, 1] * dt, bv[b, 2] * dt
        qx, qy, qz = bp[b, 0] + mx, bp[b, 1] + my, bp[b, 2] + mz
        if sweep != 0.0 and bc[b, 8] == 0.0:
            d1, _t = _distance(qx, qy, qz, r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
            if d1 - r < floor:
                d0, _t = _distance(bp[b, 0], bp[b, 1], bp[b, 2], r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                if d0 - r > floor:
                    lo, hi = 0.0, 1.0
                    for _ in range(24):
                        mid = 0.5 * (lo + hi)
                        dm, _t = _distance(bp[b, 0] + mid * mx, bp[b, 1] + mid * my, bp[b, 2] + mid * mz, r + 1.0, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                        if dm - r > floor:
                            lo = mid
                        else:
                            hi = mid
                    qx, qy, qz = bp[b, 0] + lo * mx, bp[b, 1] + lo * my, bp[b, 2] + lo * mz
                    if b == 0:
                        out[0] = 1.0
        elif sweep != 0.0 and bc[b, 26] == 0.0:
            # a box: none of its corners may pass through a surface within the tick (a falling block lands, as a ball does)
            # (a corner is small: the move is looked at in pieces no longer than the corner, or it could step right through)
            pieces = int(math.sqrt(mx * mx + my * my + mz * mz) / CORNER) + 1
            if _box_gap(bc[b], bp[b, 0], bp[b, 1], bp[b, 2], ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y) > floor:
                for piece in range(1, pieces + 1):
                    f1 = piece / pieces
                    if _box_gap(bc[b], bp[b, 0] + f1 * mx, bp[b, 1] + f1 * my, bp[b, 2] + f1 * mz, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y) < floor:
                        lo, hi = (piece - 1) / pieces, f1
                        for _ in range(12):
                            mid = 0.5 * (lo + hi)
                            if _box_gap(bc[b], bp[b, 0] + mid * mx, bp[b, 1] + mid * my, bp[b, 2] + mid * mz, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y) > floor:
                                lo = mid
                            else:
                                hi = mid
                        qx, qy, qz = bp[b, 0] + lo * mx, bp[b, 1] + lo * my, bp[b, 2] + lo * mz
                        break
        bp[b, 0], bp[b, 1], bp[b, 2] = qx, qy, qz
        k_lin, k_rot = 1.0 - bc[b, 3] * dt, 1.0 - bc[b, 4] * dt
        if b == 0:
            pushacc = ball[1]
            bv[b, 0] = bv[b, 0] * k_lin + pushacc * push[0]
            bv[b, 1] = bv[b, 1] * k_lin + pushacc * push[1] + lift - grav * dt
            bv[b, 2] = bv[b, 2] * k_lin + pushacc * push[2]
        else:
            bv[b, 0], bv[b, 1], bv[b, 2] = bv[b, 0] * k_lin, bv[b, 1] * k_lin - grav * dt, bv[b, 2] * k_lin
        bw[b, 0], bw[b, 1], bw[b, 2] = bw[b, 0] * k_rot, bw[b, 1] * k_rot, bw[b, 2] * k_rot
        if bc[b, 24] != 0.0:
            along = bv[b, 0] * bc[b, 15] + bv[b, 1] * bc[b, 16] + bv[b, 2] * bc[b, 17]
            bv[b, 0], bv[b, 1], bv[b, 2] = along * bc[b, 15], along * bc[b, 16], along * bc[b, 17]
    for b in range(nb):
        bc[b, 21], bc[b, 22], bc[b, 23] = bp[b, 0], bp[b, 1], bp[b, 2]   # where each is, for the box tests
    held = bp[:nb].copy()                                 # where each is before anything pushes it out of anything: a slider goes back onto its rail
    # ---- the contacts: each body on the mesh, then each pair of bodies
    nc = 0
    touching, edge, best_ny, carried, on_rail, placed = 0, False, -1.0, False, False, False
    for b in range(nb):
        if bc[b, 26] != 0.0:
            continue                                      # a part that does not touch the level's mesh
        if bc[b, 8] == 0.0:
            m = _contacts(bp[b, 0], bp[b, 1], bp[b, 2], r, touch, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                          ci, cd, cq, cf, cn, cg, order)
            for i in range(m):
                k = order[i]
                t = ci[k]
                if b == 0 and cg[k] <= touch:
                    touching += 1
                    if cf[k] != FACE:
                        edge = True
                    if tri_rail[t]:
                        on_rail = True
                    if tri_rail[t] or (cf[k] == FACE and cn[k, 1] >= 0.6):
                        carried = True
                    if cn[k, 1] > best_ny:
                        best_ny = cn[k, 1]
                    if tri_placed[t]:
                        placed = True
                if nc >= len(m_a):
                    break
                m_a[nc], m_b[nc] = b, -1
                m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = cn[k, 0], cn[k, 1], cn[k, 2]
                m_fr[nc] = bc[b, 1] * tri_fr[t] * f_scale
                vn = bv[b, 0] * cn[k, 0] + bv[b, 1] * cn[k, 1] + bv[b, 2] * cn[k, 2]
                m_target[nc] = -math.sqrt(bc[b, 2] * tri_el[t]) * r_scale * vn if vn < -bounce else 0.0
                nc += 1
        else:
            # a box stands on what is under its corners, as long as there is something under its middle (once its
            # middle is over an edge it goes down: it does not turn here, so it cannot tip over the edge). A hinged
            # piece touches by its hull's own corners, and the hinge holds it: it does not go down
            hinged = bc[b, 39] != 0.0
            prow = props[int(bc[b, 6])]
            over = True if hinged else _box_over_ground(bc[b], bp[b, 0], bp[b, 1], bp[b, 2], ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
            nv = int(prow[38]) if prow[38] > 0.0 else 8
            crad = HULL_CORNER if prow[38] > 0.0 else CORNER
            for kc in range(nv):
                qx, qy, qz = _vert(bc[b], prow, kc)
                ahead = touch
                if hinged:                                # a turning board's corner may cross a thin thing within a tick: look ahead that far
                    u0, u1, u2 = _pvel(b, qx, qy, qz, bp, bv, bw, bc, props)
                    ahead = touch + math.sqrt(u0 * u0 + u1 * u1 + u2 * u2) * dt
                m = _contacts(qx, qy, qz, crad, ahead, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                              ci, cd, cq, cf, cn, cg, order)
                for i in range(m):
                    k = order[i]
                    t = ci[k]
                    if cn[k, 1] > 0.5 and not over:
                        continue
                    if nc >= len(m_a):
                        break
                    m_a[nc], m_b[nc] = b, -1
                    m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = cn[k, 0], cn[k, 1], cn[k, 2]
                    m_p[nc, 0], m_p[nc, 1], m_p[nc, 2] = qx, qy, qz
                    m_fr[nc] = bc[b, 1] * tri_fr[t] * f_scale
                    if hinged:
                        u0, u1, u2 = _pvel(b, qx, qy, qz, bp, bv, bw, bc, props)
                    else:
                        u0, u1, u2 = bv[b, 0], bv[b, 1], bv[b, 2]
                    vn = u0 * cn[k, 0] + u1 * cn[k, 1] + u2 * cn[k, 2]
                    m_target[nc] = -math.sqrt(bc[b, 2] * tri_el[t]) * r_scale * vn if vn < -bounce else 0.0
                    if cg[k] > touch:                     # not touching yet: it may come as near as the gap within this tick
                        m_target[nc] = -cg[k] / dt
                    nc += 1
    first_pair = nc
    for a in range(nb):
        for b in range(a + 1, nb):
            if bc[a, 7] == 0.0 and bc[b, 7] == 0.0:
                continue                                  # two frozen bodies resting against each other
            if not _pair_ok(bc, a, b):
                continue
            sa, sb = bc[a, 8], bc[b, 8]
            if sa == 0.0 and sb == 0.0:                   # ball on ball
                dx, dy, dz = bp[a, 0] - bp[b, 0], bp[a, 1] - bp[b, 1], bp[a, 2] - bp[b, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d - 2.0 * r > touch or d < 1e-9 or nc >= len(m_a):
                    continue
                m_a[nc], m_b[nc] = a, b
                m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = dx / d, dy / d, dz / d
                m_fr[nc] = bc[a, 1] * bc[b, 1] * f_scale
                vn = (bv[a, 0] - bv[b, 0]) * dx / d + (bv[a, 1] - bv[b, 1]) * dy / d + (bv[a, 2] - bv[b, 2]) * dz / d
                m_target[nc] = -math.sqrt(bc[a, 2] * bc[b, 2]) * r_scale * vn if vn < -bounce else 0.0     # two moving bodies: no lower bound
                if a == 0:
                    touching += 1
                nc += 1
            elif sa == 0.0 or sb == 0.0:                  # ball on box: the ball is the contact's first body
                s, x = (a, b) if sa == 0.0 else (b, a)
                gap, n0, n1, n2 = _box_point(bp[s, 0], bp[s, 1], bp[s, 2], bc[x], bc[x, 9:12], r)
                if gap > touch or nc >= len(m_a):
                    continue
                if props[int(bc[x, 6]), 23] > 0.0:      # a hull inside its box: nearer than the box, it may still be clear
                    gap, n0, n1, n2 = _hull_point(bp[s, 0], bp[s, 1], bp[s, 2], bc[x], props[int(bc[x, 6])], r)
                    if gap > touch:
                        continue
                m_a[nc], m_b[nc] = s, x
                m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = n0, n1, n2
                px, py, pz = bp[s, 0] - r * n0, bp[s, 1] - r * n1, bp[s, 2] - r * n2
                m_p[nc, 0], m_p[nc, 1], m_p[nc, 2] = px, py, pz
                m_fr[nc] = bc[s, 1] * bc[x, 1] * f_scale
                if bc[x, 39] != 0.0:
                    u0, u1, u2 = _pvel(x, px, py, pz, bp, bv, bw, bc, props)
                else:
                    u0, u1, u2 = bv[x, 0], bv[x, 1], bv[x, 2]
                vn = (bv[s, 0] - u0) * n0 + (bv[s, 1] - u1) * n1 + (bv[s, 2] - u2) * n2
                m_target[nc] = -math.sqrt(bc[s, 2] * bc[x, 2]) * r_scale * vn if vn < -bounce else 0.0
                if s == 0:
                    touching += 1
                    if n1 >= 0.6:
                        carried = True                    # on top of a body: as on a floor
                nc += 1
            elif props[int(bc[a, 6]), 38] > 0.0 or props[int(bc[b, 6]), 38] > 0.0:
                # a piece with a hull's corners of its own (a hinged board, a stopper's wedge): each one's corners
                # against the other's shape
                for turn in range(2):
                    s, x = (a, b) if turn == 0 else (b, a)
                    ps, px_ = props[int(bc[s, 6])], props[int(bc[x, 6])]
                    nv = int(ps[38]) if ps[38] > 0.0 else 8
                    for kc in range(nv):
                        qx, qy, qz = _vert(bc[s], ps, kc)
                        gap, n0, n1, n2 = _shape_point(qx, qy, qz, bc[x], px_, HULL_CORNER if ps[38] > 0.0 else CORNER)
                        us0, us1, us2 = _pvel(s, qx, qy, qz, bp, bv, bw, bc, props)
                        ux0, ux1, ux2 = _pvel(x, qx, qy, qz, bp, bv, bw, bc, props)
                        ahead = touch + math.sqrt((us0 - ux0) ** 2 + (us1 - ux1) ** 2 + (us2 - ux2) ** 2) * dt
                        if gap > ahead or nc >= len(m_a):
                            continue
                        m_a[nc], m_b[nc] = s, x
                        m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = n0, n1, n2
                        m_p[nc, 0], m_p[nc, 1], m_p[nc, 2] = qx, qy, qz
                        m_fr[nc] = bc[s, 1] * bc[x, 1] * f_scale
                        vn = (us0 - ux0) * n0 + (us1 - ux1) * n1 + (us2 - ux2) * n2
                        m_target[nc] = -math.sqrt(bc[s, 2] * bc[x, 2]) * r_scale * vn if vn < -bounce else 0.0
                        if gap > touch:
                            m_target[nc] = -gap / dt
                        nc += 1
            else:                                         # box on box: each one's corners against the other
                for turn in range(2):
                    s, x = (a, b) if turn == 0 else (b, a)
                    face, side = _face_axis(bc[s], bc[x])
                    for kc in range(8):
                        qx, qy, qz = _corner(bc[s], kc)
                        gap, n0, n1, n2 = _corner_face(qx, qy, qz, bc[x], face, side)
                        if gap > touch or nc >= len(m_a):
                            continue
                        m_a[nc], m_b[nc] = s, x
                        m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = n0, n1, n2
                        m_p[nc, 0], m_p[nc, 1], m_p[nc, 2] = qx, qy, qz
                        m_fr[nc] = bc[s, 1] * bc[x, 1] * f_scale
                        vn = (bv[s, 0] - bv[x, 0]) * n0 + (bv[s, 1] - bv[x, 1]) * n1 + (bv[s, 2] - bv[x, 2]) * n2
                        m_target[nc] = -math.sqrt(bc[s, 2] * bc[x, 2]) * r_scale * vn if vn < -bounce else 0.0
                        nc += 1
    for i in range(nc):
        m_nimp[i] = 0.0
        m_timp[i, 0] = m_timp[i, 1] = m_timp[i, 2] = 0.0
    # a frozen body that nothing awake touches takes no part; one that is touched is in the solve from this tick.
    # Being in the solve passes on: what a touched body leans on or carries is touched in turn (a row, a pile)
    changed = True
    while changed:
        changed = False
        for i in range(first_pair, nc):
            a, b = m_a[i], m_b[i]
            if bc[a, 7] == 0.0 and bc[b, 7] != 0.0:
                bc[a, 7] = 0.5
                changed = True
            elif bc[b, 7] == 0.0 and bc[a, 7] != 0.0:
                bc[b, 7] = 0.5
                changed = True
        for b in range(1, nb):                            # a body's pieces wake together
            lead = int(bc[b, 25])
            if lead >= 0 and lead != b:
                if bc[b, 7] == 0.0 and bc[lead, 7] != 0.0:
                    bc[b, 7] = 0.5
                    changed = True
                elif bc[lead, 7] == 0.0 and bc[b, 7] != 0.0:
                    bc[lead, 7] = 0.5
                    changed = True
    # A ball striking a ball is an impact between those two alone: it is settled after everything else, once, and
    # what it does to their contacts with the floor shows from the next tick on. MEASURED: the struck ball leaves
    # the tick with the speed of the plain two-body rule (within 3%), and loses 6 to 16% of it in the tick after.
    for phase in range(2):
        # (a pile of boxes has many contacts that lean on each other: it takes more passes to carry the weight down)
        for _ in range((iters * (4 if nc > 12 else 1) if nc > 1 else 1) if phase == 0 else 1):
            for i in range(nc):
                a, b = m_a[i], m_b[i]
                if bc[a, 7] == 0.0 or (b >= 0 and bc[b, 7] == 0.0):
                    continue
                late = b >= 0 and m_target[i] != 0.0 and bc[a, 8] == 0.0 and bc[b, 8] == 0.0
                if late != (phase == 1):
                    continue
                n0, n1, n2 = m_n[i, 0], m_n[i, 1], m_n[i, 2]
                if bc[a, 39] != 0.0 or (b >= 0 and bc[b, 39] != 0.0):
                    # a hinged body: what it gives at the contact point depends on where the point is from the hinge
                    px, py, pz = m_p[i, 0], m_p[i, 1], m_p[i, 2]
                    ua0, ua1, ua2 = _pvel(a, px, py, pz, bp, bv, bw, bc, props)
                    ub0 = ub1 = ub2 = 0.0
                    if b >= 0:
                        ub0, ub1, ub2 = _pvel(b, px, py, pz, bp, bv, bw, bc, props)
                    vn = (ua0 - ub0) * n0 + (ua1 - ub1) * n1 + (ua2 - ub2) * n2
                    kn = _kdir(a, px, py, pz, n0, n1, n2, bp, bc, props) + (_kdir(b, px, py, pz, n0, n1, n2, bp, bc, props) if b >= 0 else 0.0)
                    if kn < 1e-6:
                        continue                          # neither side can give along this normal (a hinge against its axis, on a fixed thing): an impulse there only makes its friction lock the hinge
                    j = (m_target[i] - vn) / kn
                    new = max(0.0, m_nimp[i] + j)
                    j = new - m_nimp[i]
                    m_nimp[i] = new
                    _impulse(a, 1.0, px, py, pz, j * n0, j * n1, j * n2, bp, bv, bw, bc, props, nb)
                    if b >= 0:
                        _impulse(b, -1.0, px, py, pz, j * n0, j * n1, j * n2, bp, bv, bw, bc, props, nb)
                    # friction, along the way the point slides
                    ua0, ua1, ua2 = _pvel(a, px, py, pz, bp, bv, bw, bc, props)
                    if b >= 0:
                        ub0, ub1, ub2 = _pvel(b, px, py, pz, bp, bv, bw, bc, props)
                    u0, u1, u2 = ua0 - ub0, ua1 - ub1, ua2 - ub2
                    un = u0 * n0 + u1 * n1 + u2 * n2
                    t0, t1, t2 = u0 - un * n0, u1 - un * n1, u2 - un * n2
                    tl = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
                    if tl < 1e-9:
                        continue
                    t0, t1, t2 = t0 / tl, t1 / tl, t2 / tl
                    kt = _kdir(a, px, py, pz, t0, t1, t2, bp, bc, props) + (_kdir(b, px, py, pz, t0, t1, t2, bp, bc, props) if b >= 0 else 0.0)
                    if kt < 1e-12:
                        continue
                    c0 = m_timp[i, 0] - t0 * tl / kt
                    c1 = m_timp[i, 1] - t1 * tl / kt
                    c2 = m_timp[i, 2] - t2 * tl / kt
                    lim = m_fr[i] * m_nimp[i]
                    size = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
                    if size > lim and size > 0.0:
                        c0, c1, c2 = c0 * (lim / size), c1 * (lim / size), c2 * (lim / size)
                    j0, j1, j2 = c0 - m_timp[i, 0], c1 - m_timp[i, 1], c2 - m_timp[i, 2]
                    m_timp[i, 0], m_timp[i, 1], m_timp[i, 2] = c0, c1, c2
                    _impulse(a, 1.0, px, py, pz, j0, j1, j2, bp, bv, bw, bc, props, nb)
                    if b >= 0:
                        _impulse(b, -1.0, px, py, pz, j0, j1, j2, bp, bv, bw, bc, props, nb)
                    continue
                ima, iia = 1.0 / bc[a, 0], 1.0 / bc[a, 5]
                imb, iib = 0.0, 0.0
                if b >= 0:
                    imb, iib = 1.0 / bc[b, 0], 1.0 / bc[b, 5]
                vn = bv[a, 0] * n0 + bv[a, 1] * n1 + bv[a, 2] * n2
                if b >= 0:
                    vn -= bv[b, 0] * n0 + bv[b, 1] * n1 + bv[b, 2] * n2
                # a body on a rail gives way only along the rail: across it, it is as heavy as the rail
                ka, kb = ima, imb
                if bc[a, 24] != 0.0:
                    ka = ima * (n0 * bc[a, 15] + n1 * bc[a, 16] + n2 * bc[a, 17]) ** 2
                if b >= 0 and bc[b, 24] != 0.0:
                    kb = imb * (n0 * bc[b, 15] + n1 * bc[b, 16] + n2 * bc[b, 17]) ** 2
                if ka + kb < 1e-12:
                    continue
                j = (m_target[i] - vn) / (ka + kb)
                new = max(0.0, m_nimp[i] + j)
                j = new - m_nimp[i]
                m_nimp[i] = new
                bv[a, 0] += j * ima * n0
                bv[a, 1] += j * ima * n1
                bv[a, 2] += j * ima * n2
                if b >= 0:
                    bv[b, 0] -= j * imb * n0
                    bv[b, 1] -= j * imb * n1
                    bv[b, 2] -= j * imb * n2
                # friction at the contact point: on body a the point is at -r n, on body b at +r n (a box does not
                # turn: its inertia is beyond measure, so where the point is on it does not matter)
                a0, a1, a2 = -r * n0, -r * n1, -r * n2
                u0 = bv[a, 0] + (bw[a, 1] * a2 - bw[a, 2] * a1)
                u1 = bv[a, 1] + (bw[a, 2] * a0 - bw[a, 0] * a2)
                u2 = bv[a, 2] + (bw[a, 0] * a1 - bw[a, 1] * a0)
                kt = ima + r * r * iia
                if b >= 0:
                    u0 -= bv[b, 0] - (bw[b, 1] * a2 - bw[b, 2] * a1)
                    u1 -= bv[b, 1] - (bw[b, 2] * a0 - bw[b, 0] * a2)
                    u2 -= bv[b, 2] - (bw[b, 0] * a1 - bw[b, 1] * a0)
                    kt += imb + r * r * iib
                un = u0 * n0 + u1 * n1 + u2 * n2
                c0 = m_timp[i, 0] - (u0 - un * n0) / kt
                c1 = m_timp[i, 1] - (u1 - un * n1) / kt
                c2 = m_timp[i, 2] - (u2 - un * n2) / kt
                lim = m_fr[i] * m_nimp[i]
                size = math.sqrt(c0 * c0 + c1 * c1 + c2 * c2)
                if size > lim and size > 0.0:
                    c0, c1, c2 = c0 * (lim / size), c1 * (lim / size), c2 * (lim / size)
                j0, j1, j2 = c0 - m_timp[i, 0], c1 - m_timp[i, 1], c2 - m_timp[i, 2]
                m_timp[i, 0], m_timp[i, 1], m_timp[i, 2] = c0, c1, c2
                bv[a, 0] += j0 * ima
                bv[a, 1] += j1 * ima
                bv[a, 2] += j2 * ima
                bw[a, 0] += (a1 * j2 - a2 * j1) * iia
                bw[a, 1] += (a2 * j0 - a0 * j2) * iia
                bw[a, 2] += (a0 * j1 - a1 * j0) * iia
                if b >= 0:
                    bv[b, 0] -= j0 * imb
                    bv[b, 1] -= j1 * imb
                    bv[b, 2] -= j2 * imb
                    # on b the arm is +r n = -a, the impulse is -j: (-a) x (-j) = a x j
                    bw[b, 0] += (a1 * j2 - a2 * j1) * iib
                    bw[b, 1] += (a2 * j0 - a0 * j2) * iib
                    bw[b, 2] += (a0 * j1 - a1 * j0) * iib
                    if bc[b, 24] != 0.0:
                        along = bv[b, 0] * bc[b, 15] + bv[b, 1] * bc[b, 16] + bv[b, 2] * bc[b, 17]
                        bv[b, 0], bv[b, 1], bv[b, 2] = along * bc[b, 15], along * bc[b, 16], along * bc[b, 17]
                    _sync(bv, bw, bc, nb, b)
                if bc[a, 24] != 0.0:
                    along = bv[a, 0] * bc[a, 15] + bv[a, 1] * bc[a, 16] + bv[a, 2] * bc[a, 17]
                    bv[a, 0], bv[a, 1], bv[a, 2] = along * bc[a, 15], along * bc[a, 16], along * bc[a, 17]
                _sync(bv, bw, bc, nb, a)
    # ---- out of the surfaces and out of each other
    for b in range(1, nb):
        if bc[b, 39] == 0.0 or bc[b, 25] != float(b) or bc[b, 7] == 0.0:
            continue
        # a hinged body: the deepest of its corners in a surface or a stopper (or a stopper's corner in it) is put
        # back out by turning it about the hinge
        q = props[int(bc[b, 6])]
        for _ in range(2):
            best, bt = 0.0, 0.0
            for s in range(1, nb):
                if bc[s, 25] != float(b):
                    continue
                ps = props[int(bc[s, 6])]
                for kc in range(int(ps[38]) if ps[38] > 0.0 else 8):
                    qx, qy, qz = _vert(bc[s], ps, kc)
                    m = 0
                    if bc[s, 26] == 0.0:                  # (one in the group Floor goes through the level's floors)
                        m = _contacts(qx, qy, qz, HULL_CORNER, 0.2, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                                      ci, cd, cq, cf, cn, cg, order)
                    for i in range(m):
                        k = order[i]
                        if cg[k] < margin - 1e-6:
                            dx, dy, dz = qx - q[25], qy - q[26], qz - q[27]
                            jj = (dy * cn[k, 2] - dz * cn[k, 1]) * q[28] + (dz * cn[k, 0] - dx * cn[k, 2]) * q[29] + (dx * cn[k, 1] - dy * cn[k, 0]) * q[30]
                            if abs(jj) > 0.05 and margin - cg[k] > best:
                                best, bt = margin - cg[k], (margin - cg[k]) / jj
                    for o in range(1, nb):                # the stoppers and the mechanism's fixed parts it may touch
                        if (bc[o, 26] != 4.0 and bc[o, 26] != 2.0) or not _pair_ok(bc, s, o):
                            continue
                        po = props[int(bc[o, 6])]
                        g, e0, e1, e2 = _shape_point(qx, qy, qz, bc[o], po, HULL_CORNER)
                        if g < margin - 1e-6:
                            dx, dy, dz = qx - q[25], qy - q[26], qz - q[27]
                            jj = (dy * e2 - dz * e1) * q[28] + (dz * e0 - dx * e2) * q[29] + (dx * e1 - dy * e0) * q[30]
                            if abs(jj) > 0.05 and margin - g > best:
                                best, bt = margin - g, (margin - g) / jj
                for o in range(1, nb):                    # a stopper's corners in this piece
                    if (bc[o, 26] != 4.0 and bc[o, 26] != 2.0) or not _pair_ok(bc, s, o):
                        continue
                    po = props[int(bc[o, 6])]
                    for kc in range(int(po[38]) if po[38] > 0.0 else 8):
                        qx, qy, qz = _vert(bc[o], po, kc)
                        g, e0, e1, e2 = _shape_point(qx, qy, qz, bc[s], ps, HULL_CORNER)
                        if g < margin - 1e-6:                 # the piece goes the other way: -e at that point
                            dx, dy, dz = qx - q[25], qy - q[26], qz - q[27]
                            jj = -((dy * e2 - dz * e1) * q[28] + (dz * e0 - dx * e2) * q[29] + (dx * e1 - dy * e0) * q[30])
                            if abs(jj) > 0.05 and margin - g > best:
                                best, bt = margin - g, (margin - g) / jj
            if best <= 0.0:
                break
            bt = max(-0.05, min(0.05, bt))
            om = bw[b, 0] * q[28] + bw[b, 1] * q[29] + bw[b, 2] * q[30]
            _hinge_pose(bc, bp, bv, bw, props, nb, b, bc[b, 38] + bt, om)
    for b in range(nb):
        if bc[b, 7] == 0.0 or bc[b, 26] != 0.0 or bc[b, 39] != 0.0:
            continue
        if bc[b, 8] == 0.0:
            _settle(bp[b], r, margin, 0.2, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox, ci, cd, cq, cf, cn, cg, order)
        else:
            for _ in range(2):
                bc[b, 21], bc[b, 22], bc[b, 23] = bp[b, 0], bp[b, 1], bp[b, 2]
                over = _box_over_ground(bc[b], bp[b, 0], bp[b, 1], bp[b, 2], ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y)
                for kc in range(8):
                    qx, qy, qz = _corner(bc[b], kc)
                    m = _contacts(qx, qy, qz, CORNER, 0.2, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox,
                                  ci, cd, cq, cf, cn, cg, order)
                    for i in range(m):
                        k = order[i]
                        if cg[k] < margin - 1e-6 and not (cn[k, 1] > 0.5 and not over):
                            sh = (margin - cg[k]) / 4.0                 # four corners share a face's push
                            bp[b, 0] += sh * cn[k, 0]
                            bp[b, 1] += sh * cn[k, 1]
                            bp[b, 2] += sh * cn[k, 2]
    for b in range(nb):
        bc[b, 21], bc[b, 22], bc[b, 23] = bp[b, 0], bp[b, 1], bp[b, 2]
    for a in range(nb):
        for b in range(a + 1, nb):
            if bc[a, 7] == 0.0 and bc[b, 7] == 0.0:
                continue
            if not _pair_ok(bc, a, b):
                continue
            sa, sb = bc[a, 8], bc[b, 8]
            wa = 0.0 if (bc[a, 7] == 0.0 or bc[a, 39] != 0.0) else 1.0 / bc[a, 0]
            wb = 0.0 if (bc[b, 7] == 0.0 or bc[b, 39] != 0.0) else 1.0 / bc[b, 0]
            if (bc[a, 39] != 0.0 or bc[b, 39] != 0.0) and sa != 0.0 and sb != 0.0:
                continue                                  # a hinged piece and a box or a stopper: not here
            if wa + wb == 0.0:
                continue
            if sa == 0.0 and sb == 0.0:
                dx, dy, dz = bp[a, 0] - bp[b, 0], bp[a, 1] - bp[b, 1], bp[a, 2] - bp[b, 2]
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                gap = d - 2.0 * r
                if gap >= margin - 1e-6 or d < 1e-9:
                    continue
                n0, n1, n2 = dx / d, dy / d, dz / d
            elif sa == 0.0 or sb == 0.0:
                s, x = (a, b) if sa == 0.0 else (b, a)
                gap, n0, n1, n2 = _box_point(bp[s, 0], bp[s, 1], bp[s, 2], bc[x], bc[x, 9:12], r)
                if gap >= margin - 1e-6:
                    continue
                if props[int(bc[x, 6]), 23] > 0.0:
                    gap, n0, n1, n2 = _hull_point(bp[s, 0], bp[s, 1], bp[s, 2], bc[x], props[int(bc[x, 6])], r)
                    if gap >= margin - 1e-6:
                        continue
                if s == b:                                # the normal is from the box to the ball: turn it to point from b to a
                    n0, n1, n2 = -n0, -n1, -n2
            else:
                gap, n0, n1, n2 = 1e9, 0.0, 0.0, 0.0
                for turn in range(2):
                    s, x = (a, b) if turn == 0 else (b, a)
                    face, side = _face_axis(bc[s], bc[x])
                    for kc in range(8):
                        qx, qy, qz = _corner(bc[s], kc)
                        g, e0, e1, e2 = _corner_face(qx, qy, qz, bc[x], face, side)
                        if g < gap:
                            gap = g
                            n0, n1, n2 = (e0, e1, e2) if turn == 0 else (-e0, -e1, -e2)
                if gap >= margin - 1e-6:
                    continue
            sh = (margin - gap) / (wa + wb)
            _shift(bp, bc, nb, a, sh * wa * n0, sh * wa * n1, sh * wa * n2)
            _shift(bp, bc, nb, b, -sh * wb * n0, -sh * wb * n1, -sh * wb * n2)
    for b in range(1, nb):
        if bc[b, 24] != 0.0:
            along = (bp[b, 0] - held[b, 0]) * bc[b, 15] + (bp[b, 1] - held[b, 1]) * bc[b, 16] + (bp[b, 2] - held[b, 2]) * bc[b, 17]
            bp[b, 0], bp[b, 1], bp[b, 2] = held[b, 0] + along * bc[b, 15], held[b, 1] + along * bc[b, 16], held[b, 2] + along * bc[b, 17]
    # ---- back into the state
    for q in range(3):
        pos[q], vel[q], omega[q] = bp[0, q], bv[0, q], bw[0, q]
    for b in range(1, nb):
        k = int(bc[b, 6])
        bc[b, 28] = -1.0                                  # [28]: 1 if it may freeze now, 0 if not, -1 if it is not written back
        if bc[b, 7] == 0.0 or bc[b, 26] >= 2.0:
            continue                                      # (a fixed part stays as it is: frozen)
        for q in range(3):
            scene[k, q], scene[k, 3 + q], scene[k, 6 + q] = bp[b, q], bv[b, q], bw[b, q]
        scene[k, 9] = 1.0
        sp = math.sqrt(bv[b, 0] ** 2 + bv[b, 1] ** 2 + bv[b, 2] ** 2) + r * math.sqrt(bw[b, 0] ** 2 + bw[b, 1] ** 2 + bw[b, 2] ** 2)
        scene[k, 10] = scene[k, 10] + 1.0 if sp < STILL_SPEED else 0.0
        bc[b, 28] = 0.0
        if bp[b, 1] < y_min:
            scene[k, 9] = 2.0                             # fell out of the level
            bc[b, 28] = -1.0
        elif scene[k, 10] >= STILL_TICKS:
            near_moving = False
            for i in range(first_pair, nc):
                o = -1
                if m_a[i] == b:
                    o = m_b[i]
                elif m_b[i] == b:
                    o = m_a[i]
                if o >= 0:
                    so = math.sqrt(bv[o, 0] ** 2 + bv[o, 1] ** 2 + bv[o, 2] ** 2)
                    if so >= STILL_SPEED:
                        near_moving = True
            if not near_moving:
                bc[b, 28] = 1.0
    for b in range(1, nb):
        if bc[b, 28] != 1.0:
            continue
        lead = bc[b, 25]
        ok = True
        if lead >= 0.0:                                   # a body freezes only when every piece of it may
            for s in range(1, nb):
                if bc[s, 25] == lead and bc[s, 28] == 0.0:
                    ok = False
        if ok:
            k = int(bc[b, 6])
            scene[k, 9] = 0.0                             # frozen where it is
            for q in range(3):
                scene[k, 3 + q], scene[k, 6 + q] = 0.0, 0.0
    out[1], out[2], out[3] = float(touching), 1.0 if edge else 0.0, best_ny
    out[4], out[5], out[6] = 1.0 if carried else 0.0, 1.0 if on_rail else 0.0, 1.0 if placed else 0.0


@njit(cache=True)
def _needs_multi(pos, vel, props, scene, r, dt):
    """Whether this tick has to be the tick of several bodies: a loose body is awake, or the player is about to
    touch one."""
    for k in range(len(scene)):
        if scene[k, 9] == 1.0:
            return True
        if scene[k, 9] == 0.0 and props[k, 21] != 2.0 and props[k, 21] != 4.0:     # (a fixed part or a stopper does not touch the player)
            dx, dy, dz = pos[0] - scene[k, 0], pos[1] - scene[k, 1], pos[2] - scene[k, 2]
            size = r if props[k, 6] == 0.0 else math.sqrt(props[k, 7] ** 2 + props[k, 8] ** 2 + props[k, 9] ** 2)
            reach = r + size + 0.5 + dt * math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
            if dx * dx + dy * dy + dz * dz < reach * reach:
                return True
    return False


@njit(cache=True)
def _idle(ticks, ball, par, fans, fanpar, props, scene, margin, y_min,
          ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
          stamp, tickbox, ci, cd, cq, cf, cn, cg, order, mw):
    """The loose balls alone for a number of ticks (the player ball is not there: it is parked far above and takes
    no part). Returns the ticks done: it stops when none is awake."""
    pos, vel, omega, push = np.array([0.0, 1e7, 0.0]), np.zeros(3), np.zeros(3), np.zeros(3)
    for t in range(ticks):
        awake = False
        for k in range(len(scene)):
            if scene[k, 9] == 1.0:
                awake = True
        if not awake:
            return t
        pos[0], pos[1], pos[2] = 0.0, 1e7, 0.0
        vel[0] = vel[1] = vel[2] = 0.0
        _multi_tick(pos, vel, omega, push, ball, par, fans, fanpar, props, scene, margin, y_min,
                    ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
                    stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
                    mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11], mw[12])
    return ticks


@njit(cache=True)
def _rollout(pos, vel, omega, pushes, holds, ball, par, fans, fanpar, props, scene, mw, margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y,
             N, tri_fr, tri_el, tri_placed, tri_rail, boxes, spheres, sph_on, y_min, low, low_ox, low_oz, low_cell, stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
             n_imp, t_imp, fr, el, target, trace, stats):
    """Hold pushes[k] for holds[k] ticks each, in place. Stops early when the ball's centre enters a death box or
    falls below everything near it (DEAD), is transformed (TRIGGER: the nearest trigger sphere's centre is nearer
    than its distance and that sphere is on, i.e. a transformer of another kind than the ball; stats[9] is which
    one) or touches a placed object (PLACED).
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
            multi = len(scene) > 0 and _needs_multi(pos, vel, props, scene, r, par[8])
            if multi:
                # the player ball and the loose balls together (`_multi_tick`); it settles the positions itself
                _multi_tick(pos, vel, omega, pushes[k], ball, par, fans, fanpar, props, scene, margin, y_min,
                            ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
                            stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
                            mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11], mw[12])
                o = mw[12]
                hit, m = o[0] > 0.0, 0
                touching, edge, best_ny, carried, on_rail = int(o[1]), o[2] > 0.0, o[3], o[4] > 0.0, o[5] > 0.0
                if o[6] > 0.0:
                    return PLACED, done
            else:
                hit, m = _step(pos, vel, omega, pushes[k], ball, par, fans, fanpar, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y,
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
            if not multi:
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
            if len(spheres) > 0:                  # the game looks at the nearest transformer only: one of the ball's own kind shields a farther one
                near, nd = 0, 1e300
                for s in range(len(spheres)):
                    dx, dy, dz = pos[0] - spheres[s, 0], pos[1] - spheres[s, 1], pos[2] - spheres[s, 2]
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < nd:
                        near, nd = s, d2
                if nd < spheres[near, 3] * spheres[near, 3] and sph_on[near]:
                    stats[9] = near
                    return TRIGGER, done
    return OK, done


@njit(cache=True)
def _rollout_chunks(pos, vel, omega, push, n, chunk, ball, par, fans, fanpar, props, scene, mw, margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail, boxes, spheres, sph_on, y_min, low, low_ox, low_oz, low_cell, stamp, tickbox, ci, cd, cq, cf, cn, cg, order, n_imp, t_imp, fr, el, target,
                    out_state, out_stats, out_scene, out_status):
    """Hold one push for up to n chunks of `chunk` ticks, in place, keeping the state after every chunk: out_state
    (pos, vel, omega), out_stats, out_scene, out_status. Stops after the first chunk that does not end OK. Returns
    how many chunks were run. One call instead of n: a search asks for millions of these."""
    pushes = np.empty((1, 3))
    pushes[0, 0], pushes[0, 1], pushes[0, 2] = push[0], push[1], push[2]
    holds = np.empty(1, np.int64)
    holds[0] = chunk
    trace = np.empty((0, 9))
    k = 0
    while k < n:
        stats = out_stats[k]
        for q in range(len(stats)):
            stats[q] = 0.0
        status, done = _rollout(pos, vel, omega, pushes, holds, ball, par, fans, fanpar, props, scene, mw, margin, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail, boxes, spheres, sph_on, y_min, low, low_ox, low_oz, low_cell, stamp, tickbox, ci, cd, cq, cf, cn, cg, order, n_imp, t_imp, fr, el, target, trace, stats)
        for q in range(3):
            out_state[k, q], out_state[k, 3 + q], out_state[k, 6 + q] = pos[q], vel[q], omega[q]
        out_state[k, 9] = done
        for a in range(len(scene)):
            for q in range(11):
                out_scene[k, a, q] = scene[a, q]
        out_status[k] = status
        k += 1
        if status != OK:
            break
    return k


def ball_array(ball) -> np.ndarray:
    return np.array([ball.mass, ball.push, ball.lin_damp, ball.rot_damp, ball.friction, ball.elasticity, ball.radius, ball.inertia])


def params_array(P) -> np.ndarray:
    from .physics import DT, GRAVITY
    return np.array([P.touch, P.touch, P.floor, P.friction_scale, P.restitution_scale, P.bounce_speed, 1.0 if P.sweep else 0.0,
                     float(P.iterations), DT, GRAVITY])


@dataclass
class World:
    """A mesh prepared for the compiled tick, with what ends a rollout."""
    mesh: Mesh
    boxes: np.ndarray        # (M, 6) death boxes: lo x y z, hi x y z
    spheres: np.ndarray      # (S, 4) trigger spheres the ball must stay out of: x y z, distance
    y_min: float
    stop_on_placed: bool = True
    fans: np.ndarray | None = None      # (F, 12) the fans that push (`ballistics.fans.of`): origin, x, y, z axes
    fan_edge: str = "best"              # which distance stands for the box test at a fan's edge (`fans.Rules.params`)
    trafos: list | None = None          # the transformers the spheres stand for (`ballistics.trafo.of`), in the same order; None: plain keep-out spheres
    props: list | None = None           # the loose balls that are bodies (`ballistics.props.of`): rigid spheres of their own

    def __post_init__(self):
        m = self.mesh
        self.fans = np.zeros((0, 12)) if self.fans is None else np.ascontiguousarray(self.fans, np.float64).reshape(-1, 12)
        self.props = list(self.props or [])
        self._ballc: dict = {}
        # per loose body: mass, friction, elasticity, damping (linear, angular), inertia, shape (0 sphere, 1 box), the
        # box's half extents and its three axes in world
        # [20] the first piece of its body, [21] role, [22] mechanism, [23] hull triangles, [24] hinged, [25:28] the
        # hinge's point, [28:31] its axis, [31] the inertia about it, [32:35] the piece's centre as put, [35:38] the
        # body's mass centre as put, [38] hull corners, then the corners (VERT0) and the triangles (TRI0)
        self.prop_consts = np.zeros((len(self.props), PROP_WIDTH))
        for k, p in enumerate(self.props):
            tris = np.asarray(p.get("tris") or np.zeros((0, 9)), float).reshape(-1, 9)
            verts = np.asarray(p.get("verts") or np.zeros((0, 3)), float).reshape(-1, 3)
            if len(tris) > MAX_TRIS or len(verts) > MAX_VERTS:
                raise ValueError("%s: a hull of %d corners and %d triangles (at most %d and %d)" % (p.get("name"), len(verts), len(tris), MAX_VERTS, MAX_TRIS))
            hinged = p.get("hinge") is not None
            h = p.get("hinge") or {}
            self.prop_consts[k, :39] = [p["mass"], p["friction"], p["elasticity"], p["lin_damp"], p["rot_damp"], p["inertia"],
                                        1.0 if p.get("shape") == "box" else 0.0, *p.get("half", (0.0, 0.0, 0.0)),
                                        *np.asarray(p.get("axes", np.eye(3)), float).ravel(), 1.0 if p.get("slider") else 0.0,
                                        float(p.get("leader", -1)), float(p.get("role", 0)), float(p.get("mech", 0)), float(len(tris)),
                                        1.0 if hinged else 0.0, *h.get("anchor", (0.0, 0.0, 0.0)), *h.get("axis", (0.0, 0.0, 0.0)), float(h.get("inertia", 0.0)),
                                        *p.get("placed", (0.0, 0.0, 0.0)), *h.get("mass_centre", (0.0, 0.0, 0.0)), float(len(verts))]
            self.prop_consts[k, VERT0:VERT0 + verts.size] = verts.ravel()
            self.prop_consts[k, TRI0:TRI0 + tris.size] = tris.ravel()
        nb = len(self.props) + 1
        self.multi = (np.empty((nb, 3)), np.empty((nb, 3)), np.empty((nb, 3)), np.zeros((nb, BC_WIDTH)), np.empty(MCAP, np.int64), np.empty(MCAP, np.int64),
                      np.empty((MCAP, 3)), np.empty(MCAP), np.empty(MCAP), np.empty(MCAP), np.empty((MCAP, 3)), np.empty((MCAP, 3)), np.zeros(8))
        self._fanpar: dict = {}
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
    def of(cls, mesh: Mesh, death_boxes=(), keep_out=(), stop_on_placed: bool = True, fans=None, fan_edge: str = "best", trafos=None,
           props=None) -> "World":
        """keep_out: [(x, y, z, distance)] spheres that end a rollout whatever the ball. trafos: the transformers in
        place (`ballistics.trafo.of`), instead of keep_out: they end a rollout by the game's rule, for a ball of
        another kind only."""
        boxes = np.array([[*b["lo"], *b["hi"]] for b in death_boxes], np.float64).reshape(-1, 6)
        if trafos is not None:
            from .trafo import rules as trafo_rules
            keep_out = [(*t["pos"], trafo_rules().radius) for t in trafos]
        spheres = np.array([list(s) for s in keep_out], np.float64).reshape(-1, 4)
        y_min = float(mesh._lo_y.min()) - 30.0 if len(mesh.A) else -1e9
        return cls(mesh, boxes, spheres, y_min, stop_on_placed, fans, fan_edge, list(trafos) if trafos is not None else None, props)

    def scene(self) -> np.ndarray:
        """The loose balls as they are when nothing has touched them: (K, 11) position, velocity, angular velocity,
        flag (0 frozen, 1 awake, 2 gone), ticks it has been still. A rollout takes one and changes it in place."""
        s = np.zeros((len(self.props), 11))
        for k, p in enumerate(self.props):
            s[k, :3] = p["rest"]
            if p.get("gone"):
                s[k, 9] = 2.0
            elif p.get("settled") is False:
                s[k, 9] = 1.0
        return s

    def spheres_on(self, ball) -> np.ndarray:
        """Which spheres end a rollout of this ball: every keep-out sphere; of the transformers, those of another kind."""
        if self.trafos is None:
            return np.ones(len(self.spheres), np.bool_)
        return np.array([t["kind"] != ball.name for t in self.trafos], np.bool_)

    def fan_params(self, ball) -> np.ndarray:
        """The fans' numbers for this ball kind (read from the catalogue only when there is a fan)."""
        if not len(self.fans):
            return np.zeros(5)
        if ball.name not in self._fanpar:
            from .fans import rules
            self._fanpar[ball.name] = rules().params(ball, self.fan_edge)
        return self._fanpar[ball.name]

    def _geo(self):
        m = self.mesh
        return (*m._args(), m.N)

    def step(self, ball, P, pos, vel, omega, push):
        """One tick without settling, as `physics.step`: returns (pos, vel, omega, hit, merged contacts as
        (triangle, point, feature, normal, gap) rows, normal impulses)."""
        pos, vel, omega = np.array(pos, np.float64), np.array(vel, np.float64), np.array(omega, np.float64)
        hit, m = _step(pos, vel, omega, np.asarray(push, np.float64), ball_array(ball), params_array(P), self.fans, self.fan_params(ball), *self._geo(),
                       self.tri_fr, self.tri_el, self.stamp, self.tickbox, *self.work, *self.solve)
        ci, cd, cq, cf, cn, cg, order = self.work
        rows = [(int(ci[k]), cq[k].copy(), int(cf[k]), cn[k].copy(), float(cg[k])) for k in order[:m]]
        return pos, vel, omega, bool(hit), rows, self.solve[0][:m].copy()

    def idle(self, scene, ticks: int, P=None) -> int:
        """Let the loose balls move on their own for up to `ticks` ticks (the player ball is not there): changes the
        scene in place, returns the ticks done (it stops when none is awake)."""
        from . import physics as ph
        from .geometry import CONTACT_MARGIN
        if not len(scene):
            return 0
        fanpar = np.zeros(5)
        return int(_idle(int(ticks), ball_array(ph.WOOD), params_array(P or ph.Params()), np.zeros((0, 12)), fanpar, self.prop_consts, scene,
                         CONTACT_MARGIN, self.y_min, *self._geo(), self.tri_fr, self.tri_el, self.tri_placed, self.tri_rail,
                         self.stamp, self.tickbox, *self.work, self.multi))

    def rollout_chunks(self, ball, P, pos, vel, omega, push, n: int, chunk: int, scene=None):
        """Hold one push for up to n chunks of `chunk` ticks and give the state after every chunk: (chunks run,
        states (k, 10: pos, vel, omega, ticks done in the chunk), stats (k, 10), scenes (k, K, 11), statuses (k,)).
        It stops after the first chunk that does not end OK. The same ticks as n calls of `rollout`."""
        from .geometry import CONTACT_MARGIN
        pos, vel, omega = np.array(pos, np.float64), np.array(vel, np.float64), np.array(omega, np.float64)
        scene = self.scene() if scene is None else np.array(scene, np.float64)
        st, ss, sc, su = np.empty((n, 10)), np.empty((n, 10)), np.empty((n, len(scene), 11)), np.empty(n, np.int64)
        ba, fp, on = self._of_ball(ball)
        k = _rollout_chunks(pos, vel, omega, np.asarray(push, np.float64).reshape(3), int(n), int(chunk), ba, params_array(P), self.fans, fp,
                            self.prop_consts, scene, self.multi, CONTACT_MARGIN, *self._geo(), self.tri_fr, self.tri_el,
                            self.tri_placed, self.tri_rail, self.boxes, self.spheres, on, self.y_min, self.low, self.low_ox, self.low_oz, LOW_CELL,
                            self.stamp, self.tickbox, *self.work, *self.solve, st, ss, sc, su)
        return int(k), st[:k], ss[:k], sc[:k], su[:k]

    def _of_ball(self, ball):
        """What a rollout needs that depends on the ball's kind alone, made once: its numbers, the fans' numbers
        for it, which transformers act on it."""
        got = self._ballc.get(ball.name)
        if got is None:
            got = self._ballc[ball.name] = (ball_array(ball), self.fan_params(ball), self.spheres_on(ball))
        return got

    def rollout(self, ball, P, pos, vel, omega, pushes, holds, trace: bool = False, margin: float | None = None, scene=None):
        """Hold each push for its number of ticks. Returns (status, ticks, pos, vel, omega, stats, trace or None).
        scene: the loose balls' state (`World.scene`), changed in place; without it they are at rest where they were
        put, and what the rollout does to them is not kept."""
        if scene is None:
            scene = self.scene()
        from .geometry import CONTACT_MARGIN
        pos, vel, omega = np.array(pos, np.float64), np.array(vel, np.float64), np.array(omega, np.float64)
        pushes = np.ascontiguousarray(pushes, np.float64).reshape(-1, 3)
        holds = np.ascontiguousarray(holds, np.int64)
        tr = np.empty((int(holds.sum()) if trace else 0, 9))
        stats = np.zeros(10)
        ba, fp, on = self._of_ball(ball)
        status, done = _rollout(pos, vel, omega, pushes, holds, ba, params_array(P), self.fans, fp,
                                self.prop_consts, scene, self.multi,
                                CONTACT_MARGIN if margin is None else margin, *self._geo(), self.tri_fr, self.tri_el,
                                self.tri_placed, self.tri_rail, self.boxes, self.spheres, on, self.y_min, self.low, self.low_ox, self.low_oz, LOW_CELL, self.stamp, self.tickbox, *self.work,
                                *self.solve, tr, stats)
        return int(status), int(done), pos, vel, omega, stats, (tr[:done] if trace else None)
