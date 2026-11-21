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


MCAP = 512              # contacts of all the bodies together in one tick
STILL_TICKS = 20        # a loose ball that has not moved for this many ticks is frozen (0.3 physics s: the game's short test)
STILL_SPEED = 0.05


@njit(cache=True)
def _multi_tick(pos, vel, omega, push, ball, par, fans, fanpar, props, scene, margin, y_min,
                ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
                stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
                bp, bv, bw, bc, m_a, m_b, m_n, m_fr, m_target, m_nimp, m_timp, out):
    """One tick of the player ball together with the loose balls (the tick of `_step`, for several spheres at once).
    Every body moves with the velocity it has and has its velocity updated as in `_step`; then all contacts, ball
    on mesh and ball on ball, are solved together, one after another, a few times over: a ball struck by a rolling
    ball is pressed onto the floor by the friction between the two, and the floor then drags on it.
    props (K, 6): mass, friction, elasticity, linear damping, angular damping, inertia. scene (K, 11), in place:
    position, velocity, angular velocity, flag (0 frozen, 1 awake, 2 gone), ticks it has been still.
    out: [0] the player's tick began with a hit, [1] how many points the player touches, [2] one is an edge or
    corner, [3] the best normal y of its mesh contacts, [4] carried, [5] on a rail, [6] it touches a placed object."""
    r = ball[6]
    reach, touch, floor, f_scale, r_scale, bounce, sweep, iters, dt, grav = par[0], par[1], par[2], par[3], par[4], par[5], par[6], int(par[7]), par[8], par[9]
    K = len(props)
    nb = 1
    # ---- the bodies: 0 the player, then every loose ball that is still there
    bp[0, 0], bp[0, 1], bp[0, 2] = pos[0], pos[1], pos[2]
    bv[0, 0], bv[0, 1], bv[0, 2] = vel[0], vel[1], vel[2]
    bw[0, 0], bw[0, 1], bw[0, 2] = omega[0], omega[1], omega[2]
    bc[0, 0], bc[0, 1], bc[0, 2], bc[0, 3], bc[0, 4], bc[0, 5], bc[0, 6], bc[0, 7] = ball[0], ball[4], ball[5], ball[2], ball[3], ball[7], -1.0, 1.0
    for k in range(K):
        if scene[k, 9] == 2.0:
            continue
        for q in range(3):
            bp[nb, q], bv[nb, q], bw[nb, q] = scene[k, q], scene[k, 3 + q], scene[k, 6 + q]
        bc[nb, 0], bc[nb, 1], bc[nb, 2], bc[nb, 3], bc[nb, 4], bc[nb, 5] = props[k, 0], props[k, 1], props[k, 2], props[k, 3], props[k, 4], props[k, 5]
        bc[nb, 6], bc[nb, 7] = float(k), scene[k, 9]                    # which loose ball; awake (1) or frozen (0)
        nb += 1
    lift = _fans_pushing(pos[0], pos[1], pos[2], fans, fanpar) * fanpar[4] / ball[0] if len(fans) > 0 else 0.0
    out[0] = 0.0
    # ---- move, then the velocity update (a frozen ball does neither: nothing acts on it until it is struck)
    for b in range(nb):
        if bc[b, 7] == 0.0:
            continue
        mx, my, mz = bv[b, 0] * dt, bv[b, 1] * dt, bv[b, 2] * dt
        qx, qy, qz = bp[b, 0] + mx, bp[b, 1] + my, bp[b, 2] + mz
        if sweep != 0.0:
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
    # ---- the contacts: each body on the mesh, then each pair of balls
    nc = 0
    touching, edge, best_ny, carried, on_rail, placed = 0, False, -1.0, False, False, False
    for b in range(nb):
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
    first_pair = nc
    for a in range(nb):
        for b in range(a + 1, nb):
            dx, dy, dz = bp[a, 0] - bp[b, 0], bp[a, 1] - bp[b, 1], bp[a, 2] - bp[b, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d - 2.0 * r > touch or d < 1e-9 or nc >= len(m_a):
                continue
            if bc[a, 7] == 0.0 and bc[b, 7] == 0.0:
                continue                                  # two frozen balls resting against each other
            m_a[nc], m_b[nc] = a, b
            m_n[nc, 0], m_n[nc, 1], m_n[nc, 2] = dx / d, dy / d, dz / d
            m_fr[nc] = bc[a, 1] * bc[b, 1] * f_scale
            vn = (bv[a, 0] - bv[b, 0]) * dx / d + (bv[a, 1] - bv[b, 1]) * dy / d + (bv[a, 2] - bv[b, 2]) * dz / d
            m_target[nc] = -math.sqrt(bc[a, 2] * bc[b, 2]) * r_scale * vn if vn < -bounce else 0.0     # two moving bodies: no lower bound
            if a == 0:
                touching += 1
            nc += 1
    for i in range(nc):
        m_nimp[i] = 0.0
        m_timp[i, 0] = m_timp[i, 1] = m_timp[i, 2] = 0.0
    # a frozen ball that nothing awake touches takes no part (its own weight is not in the solve)
    for b in range(1, nb):
        if bc[b, 7] == 0.0:
            hit_by = False
            for i in range(first_pair, nc):
                if (m_a[i] == b and bc[m_b[i], 7] != 0.0) or (m_b[i] == b and bc[m_a[i], 7] != 0.0):
                    hit_by = True
            if hit_by:
                bc[b, 7] = 0.5                           # struck: it is in the solve from this tick, awake from the next
    # A ball striking a ball is an impact between those two alone: it is settled after everything else, once, and
    # what it does to their contacts with the floor shows from the next tick on. MEASURED: the struck ball leaves
    # the tick with the speed of the plain two-body rule (within 3%), and loses 6 to 16% of it in the tick after.
    for phase in range(2):
        for _ in range((iters if nc > 1 else 1) if phase == 0 else 1):
            for i in range(nc):
                a, b = m_a[i], m_b[i]
                if bc[a, 7] == 0.0 or (b >= 0 and bc[b, 7] == 0.0 and bc[a, 7] == 0.0):
                    continue
                if (b >= 0 and m_target[i] != 0.0) != (phase == 1):
                    continue
                n0, n1, n2 = m_n[i, 0], m_n[i, 1], m_n[i, 2]
                ima, iia = 1.0 / bc[a, 0], 1.0 / bc[a, 5]
                imb, iib = 0.0, 0.0
                if b >= 0:
                    imb, iib = 1.0 / bc[b, 0], 1.0 / bc[b, 5]
                vn = bv[a, 0] * n0 + bv[a, 1] * n1 + bv[a, 2] * n2
                if b >= 0:
                    vn -= bv[b, 0] * n0 + bv[b, 1] * n1 + bv[b, 2] * n2
                j = (m_target[i] - vn) / (ima + imb)
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
                # friction at the contact point: on body a the point is at -r n, on body b at +r n
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
    # ---- out of the surfaces and out of each other
    for b in range(nb):
        if bc[b, 7] == 0.0:
            continue
        _settle(bp[b], r, margin, 0.2, ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, stamp, tickbox, ci, cd, cq, cf, cn, cg, order)
    for a in range(nb):
        for b in range(a + 1, nb):
            if bc[a, 7] == 0.0 and bc[b, 7] == 0.0:
                continue
            dx, dy, dz = bp[a, 0] - bp[b, 0], bp[a, 1] - bp[b, 1], bp[a, 2] - bp[b, 2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            gap = d - 2.0 * r
            if gap >= margin - 1e-6 or d < 1e-9:
                continue
            wa = 0.0 if bc[a, 7] == 0.0 else 1.0 / bc[a, 0]
            wb = 0.0 if bc[b, 7] == 0.0 else 1.0 / bc[b, 0]
            sh = (margin - gap) / (wa + wb)
            bp[a, 0] += sh * wa * dx / d
            bp[a, 1] += sh * wa * dy / d
            bp[a, 2] += sh * wa * dz / d
            bp[b, 0] -= sh * wb * dx / d
            bp[b, 1] -= sh * wb * dy / d
            bp[b, 2] -= sh * wb * dz / d
    # ---- back into the state
    for q in range(3):
        pos[q], vel[q], omega[q] = bp[0, q], bv[0, q], bw[0, q]
    for b in range(1, nb):
        k = int(bc[b, 6])
        if bc[b, 7] == 0.0:
            continue
        for q in range(3):
            scene[k, q], scene[k, 3 + q], scene[k, 6 + q] = bp[b, q], bv[b, q], bw[b, q]
        scene[k, 9] = 1.0
        sp = math.sqrt(bv[b, 0] ** 2 + bv[b, 1] ** 2 + bv[b, 2] ** 2) + r * math.sqrt(bw[b, 0] ** 2 + bw[b, 1] ** 2 + bw[b, 2] ** 2)
        scene[k, 10] = scene[k, 10] + 1.0 if sp < STILL_SPEED else 0.0
        if bp[b, 1] < y_min:
            scene[k, 9] = 2.0                             # fell out of the level
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
                scene[k, 9] = 0.0                         # frozen where it is
                for q in range(3):
                    scene[k, 3 + q], scene[k, 6 + q] = 0.0, 0.0
    out[1], out[2], out[3] = float(touching), 1.0 if edge else 0.0, best_ny
    out[4], out[5], out[6] = 1.0 if carried else 0.0, 1.0 if on_rail else 0.0, 1.0 if placed else 0.0


@njit(cache=True)
def _needs_multi(pos, vel, scene, r, dt):
    """Whether this tick has to be the tick of several bodies: a loose ball is awake, or the player is about to
    touch one."""
    for k in range(len(scene)):
        if scene[k, 9] == 1.0:
            return True
        if scene[k, 9] == 0.0:
            dx, dy, dz = pos[0] - scene[k, 0], pos[1] - scene[k, 1], pos[2] - scene[k, 2]
            reach = 2.0 * r + 0.5 + dt * math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
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
                    mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11])
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
            multi = len(scene) > 0 and _needs_multi(pos, vel, scene, r, par[8])
            if multi:
                # the player ball and the loose balls together (`_multi_tick`); it settles the positions itself
                _multi_tick(pos, vel, omega, pushes[k], ball, par, fans, fanpar, props, scene, margin, y_min,
                            ox, oz, cell, nx, nz, start, items, A, B, C, lo_y, hi_y, N, tri_fr, tri_el, tri_placed, tri_rail,
                            stamp, tickbox, ci, cd, cq, cf, cn, cg, order,
                            mw[0], mw[1], mw[2], mw[3], mw[4], mw[5], mw[6], mw[7], mw[8], mw[9], mw[10], mw[11])
                o = mw[11]
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
        self.prop_consts = np.array([[p["mass"], p["friction"], p["elasticity"], p["lin_damp"], p["rot_damp"], p["inertia"]] for p in self.props],
                                    np.float64).reshape(-1, 6)
        nb = len(self.props) + 1
        self.multi = (np.empty((nb, 3)), np.empty((nb, 3)), np.empty((nb, 3)), np.empty((nb, 8)), np.empty(MCAP, np.int64), np.empty(MCAP, np.int64),
                      np.empty((MCAP, 3)), np.empty(MCAP), np.empty(MCAP), np.empty(MCAP), np.empty((MCAP, 3)), np.zeros(8))
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
        status, done = _rollout(pos, vel, omega, pushes, holds, ball_array(ball), params_array(P), self.fans, self.fan_params(ball),
                                self.prop_consts, scene, self.multi,
                                CONTACT_MARGIN if margin is None else margin, *self._geo(), self.tri_fr, self.tri_el,
                                self.tri_placed, self.tri_rail, self.boxes, self.spheres, self.spheres_on(ball), self.y_min, self.low, self.low_ox, self.low_oz, LOW_CELL, self.stamp, self.tickbox, *self.work,
                                *self.solve, tr, stats)
        return int(status), int(done), pos, vel, omega, stats, (tr[:done] if trace else None)
