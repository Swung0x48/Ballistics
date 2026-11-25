"""Closed-form ledge and rail-end proposals for the coarse graph."""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import groupby
from time import perf_counter

import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import lambertw

from . import closedform as cf, physics as ph
from .coarse import STEP, WALK_NY, KIND_INDEX, _lane_chains
from .geometry import FACE, _closest
from . import sweep, contactset, motion, rails, conformal, rail_inverse
from .sweep import chord as _chord, curve as _curve

FOLLOW_SPEED = 9.0
STOP_COST = 9.0
RULE = "closed form ledges and rail ends, three balls, supported momentum continuation and analytic manifolds and mesh rail corridors and conformal road slopes, finite rail joints, local step support and paired rail crowns, normal impulses, confirmed rail goals and paper impact hypotheses v19"
GRAVITY_IMPULSE = ph.GRAVITY * ph.DT
FLOOR = ph.Params().floor
XZ = [0, 2]
EPS = 1e-7
GOAL_NEAR = 3 * STEP  # ASSUMED: keep individual landings within three coarse spacings of a goal
BAND = 32 * STEP      # broad phase only, not a reach threshold
# ASSUMED collision-mesh uncertainty at a fitted rail end. The support
# continuation is local (one sphere radius), not an owner-wide exemption.
RAIL_CONTACT_QUERY = .25
DEFAULT_ELASTICITY = .3  # ASSUMED for triangles without exported material
MAX_BOUNCES = 3        # ASSUMED search bound, not an engine limit
RAIL_BALANCED = .9      # opposing horizontal projections for two-point support
KEYS = [(a | b) for a in (0, ph.KEY_UP, ph.KEY_DOWN) for b in (0, ph.KEY_LEFT, ph.KEY_RIGHT) if a or b]


def cost(edge):
    return edge["ticks"] * ph.DT * FOLLOW_SPEED + STOP_COST


def runway_speed(axis, distance):
    """Speed from rest after a supported run, using the positive Lambert root."""
    if not axis.climbs_from_rest() or distance <= EPS:
        return 0.
    ticks = max(cf._w_roots(0., axis.impulse, axis.loss, distance), default=0.)
    return -axis.terminal_speed * math.expm1(ticks * math.log1p(-axis.loss))


def variants(g, u, v):
    """All ball-specific ways of the pair; allow old/synthetic graphs without variants."""
    edge = g.info.get("launches", {}).get((u, v))
    return () if edge is None else edge.get("variants", (edge,))


def valid_in_sector(edge, landing, trafos, radius):
    """A single-kind witness cannot switch kinds and keep flying unchanged.

    Only the sector's active transformers are relevant. Transformation on
    arrival is still handled by StateGraph's destination kind.
    """
    if edge.get('flight_ticks', math.inf) <= EPS:
        return False
    if not trafos:
        return True
    path=motion.path_for(edge)
    for center, kind in trafos:
        if kind == edge['ball'] or np.linalg.norm(landing-center)<radius:
            continue
        if any(motion.sphere_hit(p,center,radius) for p in path):
            return False
    return True


def way_for(g, u, v, ball):
    return next((e for e in variants(g, u, v) if e.get("ball") == ball), g.info["launches"][u, v])


def _bands(P):
    levels = np.floor(P[:, 1] / BAND).astype(int)
    for k in np.unique(levels):
        ids = np.flatnonzero(levels == k)
        yield P[ids, 1].min(), P[ids, 1].max(), ids, cKDTree(P[ids][:, XZ])


def roll_labels(g, transformer_kind=None, ball=None, inside=None):
    """Strong components, not weak connectivity across one-way rolls."""
    k = transformer_kind or {}
    n = len(g.pos)
    keep = np.array([k.get(i, ball) == ball for i in range(n)])
    if inside is not None:
        keep &= inside
    terrain=g.info.get('terrain_ball_masks',{})
    bit=1<<KIND_INDEX[ball] if ball in KIND_INDEX else 7
    pairs = [(u, v) for u, adj in enumerate(g.info.get('launch_component_adj', g.adj)) if keep[u]
             for v, _, how in adj if how in ("roll", "rail", "rail end") and keep[v]
             and terrain.get((u,v,how),7)&bit]
    a, b = zip(*pairs) if pairs else ([], [])
    mat = csr_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    return connected_components(mat, directed=True, connection="strong")[1]


def witness_labels(g, info, transformer_kind, ball):
    """Refine rolling equivalence wherever a sector clips the rolling graph.

    The global component alone can join two ends by going outside the sector
    that consumes the cache. In every sector a missing point is its own
    sentinel, not the component of a substitute point inside that sector.
    """
    masks = g.info.get('launch_sector_masks', ())
    base = roll_labels(g, transformer_kind, ball)
    if not len(masks):
        return base
    signatures = [base]
    for inside in masks:
        signatures.append(np.where(inside, roll_labels(g, transformer_kind, ball, inside), -1))
    _, labels = np.unique(np.stack(signatures, axis=1), axis=0, return_inverse=True)
    return np.ravel(labels)


def _coeff(t, loss):
    """A, B on the actual within-PSI chord (not the smooth extension)."""
    k = np.floor(t)
    f = t - k
    power, s, q = cf._coeff(k, loss)
    return ph.DT * (s + f * power), ph.DT * (q + f * s)


def _time_at_b(b, loss):
    """Invert B, with the Lambert W root followed by exact chord interpolation."""
    a = -math.log1p(-loss) / loss
    c = 1 + (loss * loss / ph.DT) * b
    if np.isscalar(b):
        t = max(1., (c + float(lambertw(-a * math.exp(-a * c)).real) / a) / loss)
        k = math.floor(t)
        _, S, Q = cf._coeff(k, loss)
        if Q * ph.DT > b and k > 1:
            k -= 1
            _, S, Q = cf._coeff(k, loss)
        if (Q + S) * ph.DT < b:
            k += 1
            _, S, Q = cf._coeff(k, loss)
        return k + (b / ph.DT - Q) / S
    t = np.maximum(1, (c + lambertw(-a * np.exp(-a * c)).real / a) / loss)
    k = np.floor(t)
    _, s, q = cf._coeff(k, loss)
    k -= (q * ph.DT > b) & (k > 1)
    _, s, q = cf._coeff(k, loss)
    k += (q + s) * ph.DT < b
    _, s, q = cf._coeff(k, loss)
    return k + (b / ph.DT - q) / s


def _vertex(v, b, loss):
    """Integer knot where the within-tick velocity changes sign."""
    u = b / loss
    return math.ceil(math.log(u / (u - v)) / math.log1p(-loss)) if v * b < 0 else 0


def _peak(vy, loss):
    return 0 if vy < 0 else max(1, _vertex(vy, -GRAVITY_IMPULSE, loss))


def _landing_drift(ball, slope, speed, drops):
    """Horizontal drift at the descending height crossing, or NaN if too high."""
    loss = ball.lin_damp * ph.DT
    vy = slope * speed
    peak = _peak(vy, loss)
    a0, b0 = _coeff(peak, loss)
    g = GRAVITY_IMPULSE
    J = 1 + vy * loss / g
    a = -math.log1p(-loss) / loss
    if np.isscalar(drops):
        if vy * a0 - g * b0 < -drops:
            return math.nan
        c = drops * loss**2 / (ph.DT * g) + J
        root = (c + float(lambertw(max(-a * J * math.exp(-a * c), -1 / math.e + 1e-16)).real) / a) / loss
        k = max(peak, math.floor(root))
        A, B = _coeff(k, loss)
        if g * B - vy * A > drops and k > peak:
            k -= 1
            A, B = _coeff(k, loss)
        A1, B1 = _coeff(k + 1, loss)
        if g * B1 - vy * A1 < drops:
            A, B = A1, B1
            A1, B1 = _coeff(k + 2, loss)
        delta = g * (B1 - B) - vy * (A1 - A)
        f = min(1., max(0., (drops - g * B + vy * A) / delta)) if delta > 0 else 0.
        return speed * (A + f * (A1 - A))
    c = np.maximum(drops, g * b0 - vy * a0) * loss**2 / (ph.DT * g) + J
    root = (c + lambertw(np.maximum(-a * J * np.exp(-a * c), -1 / math.e + 1e-16)).real / a) / loss
    k = np.maximum(peak, np.floor(root))
    A, B = _coeff(k, loss)
    k -= (g * B - vy * A > drops) & (k > peak)
    A, B = _coeff(k, loss)
    A1, B1 = _coeff(k + 1, loss)
    k += (g * B1 - vy * A1 < drops)
    A, B = _coeff(k, loss)
    A1, B1 = _coeff(k + 1, loss)
    delta = g * (B1 - B) - vy * (A1 - A)
    f = np.clip(np.divide(drops - g * B + vy * A, delta, out=np.zeros_like(A), where=delta > 0), 0, 1)
    return np.where(vy * a0 - g * b0 < -drops, np.nan, speed * (A + f * (A1 - A)))


def _clip(lo, hi, p, q):
    """Intersect [lo,hi] with p*s <= q, elementwise."""
    if abs(p) < EPS:
        return lo, np.where(q >= -EPS, hi, -np.inf)
    if p > 0:
        return lo, np.minimum(hi, (q + EPS) / p)
    return np.maximum(lo, (q + EPS) / p), hi


def reachable(ball, delta, direction, slope, vmax):
    """Return admissible drift intervals for a batch of destination displacements.

    Write s = v*A(n), h = -delta_y and B = (h+slope*s)/gravity_impulse.
    In the flat case this is a segment + a square. On a slope the half-width
    varies *linearly* with s, so the same four half-space clips work without
    enumerating speeds or mistakenly combining drift and time from different v.
    """
    Q = np.atleast_2d(delta)
    drop = -Q[:, 1]
    loss = ball.lin_damp * ph.DT
    ratio = ball.push / GRAVITY_IMPULSE
    lo = np.zeros(len(Q))
    hi = np.full(len(Q), vmax * ph.DT / loss)
    lo, hi = _clip(lo, hi, -slope, drop)
    for i, axis in enumerate(XZ):
        for sign in (-1, 1):
            lo, hi = _clip(lo, hi, -sign * direction[i] - ratio * slope,
                           ratio * drop - sign * Q[:, axis])
    possible = np.flatnonzero(lo <= hi)
    if vmax > EPS and len(possible):
        ends = _landing_drift(ball, slope, vmax, drop[possible])
        hi[possible] = np.minimum(hi[possible], np.nan_to_num(ends, nan=-1.))
        for i in possible[(drop[possible] < 0) & (lo[possible] <= hi[possible])]:
            h = drop[i]
            def above(v):
                a, b = _coeff(_peak(v * slope, loss), loss)
                return v * slope * a - GRAVITY_IMPULSE * b + h
            v = brentq(above, 0, vmax, xtol=EPS)
            lo[i] = max(lo[i], v * _coeff(_peak(v * slope, loss), loss)[0])
    return lo, hi


def _support(mesh, p, radius):
    return tuple(c for c in mesh.contacts(p, radius, .025)
                 if c.normal[1] >= WALK_NY and not mesh.placed[c.triangle])


@njit(cache=True)
def _box_triangles(lows, highs, tri_lo, tri_hi, grid):
    ox, oz, cell, nx, nz, cuts, items = grid
    seen = set()
    ordered = []
    for lo, hi in zip(lows, highs):
        ix = max(0, int(math.floor((lo[0] - ox) / cell)))
        iz = max(0, int(math.floor((lo[2] - oz) / cell)))
        jx = min(nx - 1, int(math.floor((hi[0] - ox) / cell)))
        jz = min(nz - 1, int(math.floor((hi[2] - oz) / cell)))
        for x in range(ix, jx + 1):
            for z in range(iz, jz + 1):
                k = x * nz + z
                for j in range(cuts[k], cuts[k + 1]):
                    t = items[j]
                    if (t not in seen and tri_hi[t, 0] >= lo[0] and tri_lo[t, 0] <= hi[0]
                            and tri_hi[t, 1] >= lo[1] and tri_lo[t, 1] <= hi[1]
                            and tri_hi[t, 2] >= lo[2] and tri_lo[t, 2] <= hi[2]):
                        seen.add(t)
                        ordered.append(t)
    return ordered


def _lip(p, d, normal, mesh, tris, lo, hi):
    """Intersect the tangent contact ray with coplanar triangles and union intervals."""
    q = p - (ph.WOOD.radius + FLOOR) * normal
    end = q + 2 * STEP * d
    reach = 2 * STEP
    ids = _box_triangles(np.minimum(q, end)[None] - .03, np.maximum(q, end)[None] + .03, lo, hi, mesh._grid)
    if not len(ids):
        return 0.0
    T = tris[ids]
    T = T[np.max(np.abs((T - q) @ normal), axis=1) < .03]
    intervals = []
    for tri in T:
        E = np.roll(tri, -1, axis=0) - tri
        n = np.cross(E[0], E[1])
        if np.linalg.norm(n) < EPS:
            continue
        a = np.cross(E, q - tri) @ n
        b = np.cross(E, d) @ n
        l, h = 0., reach
        for x, y in zip(a, b):
            if abs(y) < EPS:
                if x < -EPS:
                    h = -1.
            elif y > 0:
                l = max(l, -x / y)
            else:
                h = min(h, -x / y)
        if h >= l - EPS:
            intervals.append((l, h))
    last = 0.
    for l, h in sorted(intervals):
        if l > last + EPS:
            break
        last = max(last, h)
    return last


def _sites(g, mesh, tris, lo, hi):
    P = g.pos
    support_adj = g.info.get("launch_runup_adj",g.adj)
    n = g.info["rest_places"]
    for u in np.flatnonzero(g.info["edge_node"][:n]):
        contacts = _support(mesh, P[u], ph.WOOD.radius)
        if not contacts:
            continue
        normal = contacts[0].normal
        p = P[u].copy()
        p[1] += (FLOOR - contacts[0].gap) / normal[1] + EPS
        for key in KEYS:
            push = ph.key_push(key, 0)[XZ]
            d = push / np.linalg.norm(push)
            def along(w, direction):
                nearest, best = -1, .25 * STEP
                for v, _, k in support_adj[w]:
                    if k == "roll":
                        vq = (P[v] - P[w])[XZ]
                        length = vq @ direction
                        off = np.linalg.norm(vq - length * direction)
                        if length >= .5 * STEP and off < best:
                            nearest, best = v, off
                return nearest
            if along(u, d) >= 0:
                continue
            slope = -float(normal[XZ] @ d) / normal[1]
            tangent = np.array([d[0], slope, d[1]])
            lip = _lip(p, tangent, normal, mesh, tris, lo, hi)
            if lip >= 2 * STEP - EPS:
                continue
            q = p + tangent * (lip + EPS)
            run = [int(u)]
            while (v := along(run[-1], -d)) >= 0:
                run.append(v)
            yield int(u), key, q, d, slope, normal, run, np.r_[lip, lip + np.cumsum(np.linalg.norm(np.diff(P[run], axis=0)[:, XZ], axis=1))], None
    rail_geometry = rails.physical_lanes(g, mesh)
    for chain in _lane_chains(g):
        track = rail_geometry[int(g.lane[chain[0]])]
        if len(chain) < 2 or len(track) < 2:
            continue
        for run, points in ((chain,track),(chain[::-1],track[::-1])):
            u=int(run[0]); t=points[0]-points[1]
            L=np.linalg.norm(t[XZ])
            if L<EPS:continue
            d=t[XZ]/L; slope=t[1]/L
            normal=np.array([-slope*d[0],1.,-slope*d[1]])
            normal/=np.linalg.norm(normal)
            key=max(KEYS,key=lambda k: ph.key_push(k,0)[XZ]@d)
            points=points+np.array([0.,FLOOR+EPS,0.])
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0)[:,XZ],axis=1))]
            mapped=[]
            for p in P[run]:
                j,q=rails.project_pieces(points,p)
                mapped.append(arc[j]+np.linalg.norm((q-points[j])[XZ]))
            distances=np.maximum.accumulate(np.r_[0.,mapped[1:-1],arc[-1]])
            yield u,key,points[0],d,slope,normal,list(run),distances,(points,arc)


# Quadrature is in angle (with theta = release_angle * u**2 to remove the
# from-rest endpoint singularity), not physical steps or integration of state.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X = (_GL_X + 1.) * .5
_GL_W = _GL_W * .5


def lip_basis(obs, q, normal, tangent, r):
    """Find an actual edge direction at the contact ray's exit, when on a face."""
    pivot = q - r * normal
    ids = _box_triangles(pivot[None] - .03, pivot[None] + .03, obs.lo, obs.hi, obs.mesh._grid)
    found = None
    for i in ids:
        n = obs.mesh.N[i]
        if abs(n @ normal) < 1 - 1e-4:
            continue
        tri = obs.tris[i]
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            e = b - a
            length = np.linalg.norm(e)
            if length <= EPS:
                continue
            e /= length
            projection = (pivot - a) @ e
            if not -.03 <= projection <= length + .03:
                continue
            near = a + e * np.clip(projection, 0, length)
            distance = np.linalg.norm(pivot - near)
            outward = np.cross(e, n)
            outward -= (outward @ normal) * normal
            h = outward @ tangent
            if distance < .03 and h > EPS:
                rank = (distance, -h)
                if found is None or rank < found[0]:
                    found = rank, near, outward / np.linalg.norm(outward)
    if found is None:
        return pivot, tangent / np.linalg.norm(tangent)
    return found[1:]


def rail_lip(obs, q, normal, tangent, double, radius):
    """Find the cap witness of a rail end; do not use triangles along its shaft as the rim."""
    mesh = obs.mesh
    # The coarse lane is an analytic tube-axis fit, not a sphere/mesh contact.
    # Look at the actual end section in the exported collision mesh.
    forward = tangent.copy()
    forward[1] = 0.
    forward /= np.linalg.norm(forward)
    side = np.cross(normal, tangent)
    side /= np.linalg.norm(side)
    reach = radius + FLOOR
    cs = mesh.contacts(q, radius, .25)
    caps = [c for c in cs if mesh.rail[c.triangle] and c.feature in ('vertex', 'edge')
            and abs((c.point - q) @ forward) < .03 and c.point[1] <= q[1]]
    if double:
        sides = [[c for c in caps if sign * (c.point - q) @ side > .1] for sign in (-1, 1)]
        if not all(sides):
            return None
        p0, p1 = [min(rows, key=lambda c: c.distance).point for rows in sides]
        line = p1 - p0
        gauge = np.linalg.norm(line)
        if not EPS < gauge < 2 * reach:
            return None
        line /= gauge
        p = (p0 + p1) * .5
        n = normal - (normal @ line) * line
        n /= np.linalg.norm(n)
        out = np.cross(line, n)
        out *= np.sign(out @ tangent)
        rho = math.sqrt(reach**2 - (gauge / 2)**2)
        return p, n, out, rho
    if not caps:
        return None
    p = min(caps, key=lambda c: c.distance).point
    return p, normal, tangent / np.linalg.norm(tangent), reach


@dataclass(frozen=True)
class RailSupport:
    owner: int
    faces: np.ndarray
    axes: tuple = ()


def rail_support(obs, q, radius):
    """Identify the departing rail object from the original coarse lane centre."""
    m = obs.mesh
    nearest = next((c for c in m.contacts(q, radius, RAIL_CONTACT_QUERY)
                    if m.rail[c.triangle] and c.normal[1] > 0), None)
    if nearest is None:
        return None
    owner = int(m.owner[nearest.triangle])
    # The whole allowed prefix is within a radius of the release point. The
    # rim can itself move one radius from the lane contact. Keep a local box.
    reach = 4 * (radius + FLOOR)
    nearby = _box_triangles(q[None] - reach, q[None] + reach, obs.lo, obs.hi, m._grid)
    faces = np.array([i for i in nearby if m.rail[i] and m.owner[i] == owner], dtype=np.int64)
    return RailSupport(owner, faces)


@dataclass
class Rim:
    start: ph.State
    seconds: float
    pivot: np.ndarray
    normal: np.ndarray
    outward: np.ndarray
    axis: np.ndarray
    along: float
    angle: float
    radius: float
    energy: tuple = ()

    def motion(self):
        if not self.angle:
            return motion.line(self.start.pos,self.start.pos)
        return motion.arc(self.pivot,self.radius*self.normal,self.radius*self.outward,
                          0.,self.angle,self.along*self.axis,self.energy)

    def way(self):
        radius = self.radius
        k = max(1, math.ceil(radius * self.angle + abs(self.along * self.seconds)))
        u = np.linspace(0, 1, k + 1)
        th = self.angle * u
        return (self.pivot + radius * (np.cos(th)[:, None] * self.normal + np.sin(th)[:, None] * self.outward)
                + (self.along * self.seconds * u)[:, None] * self.axis)


def rim_release(ball, pivot, normal, outward, velocity, key, radius=None, *, inertia_ratio=None, push=None):
    """Pure rolling on a straight edge under gravity and one constant key force.

    The transverse problem is exactly conservative; along the edge we retain
    the incoming velocity, with no axial acceleration/rolling resistance.
    """
    r = ball.radius + FLOOR if radius is None else radius
    axis = np.cross(normal, outward)
    across, along = float(velocity @ outward), float(velocity @ axis)
    if push is None:
        a = ph.key_push(key,0) * (ball.push/ph.DT)
        a[1] -= ph.GRAVITY
    else:
        a = cf.air_impulse(ball,push=push) / ph.DT
    an, at = float(a @ normal), float(a @ outward)
    k = ball.inertia / (ball.mass * r**2) if inertia_ratio is None else inertia_ratio * (ball.radius/r)**2
    if an + across**2 / r >= 0:
        return Rim(ph.State(pivot + r * normal, velocity), 0., pivot, normal, outward, axis, along, 0., r)
    if across <= EPS and at <= 0:
        return None  # exactly balanced or pushing back onto the supported face
    # a.n(theta) = an*cos(theta) + at*sin(theta); choose the first rising root.
    target = (2 * an - (1 + k) * across**2 / r) / (3 + k)
    phase = math.atan2(at, an)
    angle = (phase - math.acos(np.clip(target / math.hypot(an, at), -1., 1.))) % (2 * math.pi)
    # The first energy barrier is an endpoint or the minimum of the
    # sinusoidal work. Quadrature nodes are not used to decide passage.
    bottom = (phase + math.pi) % (2 * math.pi)
    gain = 2 * r / (1 + k)
    if 0. < bottom < angle and across**2 + gain * (-math.hypot(an, at) - an) <= 0.:
        return None
    if across**2 + gain * (-2*an*math.sin(angle/2)**2 + at*math.sin(angle)) <= 0.:
        return None
    theta = angle * _GL_X**2
    work = -2 * an * np.sin(theta / 2)**2 + at * np.sin(theta)
    v2 = across**2 + gain * work
    seconds = float(r * np.dot(_GL_W, 2 * angle * _GL_X / np.sqrt(v2)))
    n = normal * math.cos(angle) + outward * math.sin(angle)
    t = outward * math.cos(angle) - normal * math.sin(angle)
    speed = math.sqrt(max(0., -r * float(a @ n)))
    p = pivot + axis * (along * seconds) + r * n
    return Rim(ph.State(p, along * axis + speed * t), seconds, pivot, normal, outward, axis, along, angle, r,(across**2,gain*an,gain*at))


def convex_release(ball, center, normal, next_normal, tangent, velocity, push, inertia_ratio):
    outward = _unit(next_normal - (next_normal @ normal)*normal)
    if outward @ tangent <= EPS:
        return None
    r = ball.radius + FLOOR
    rim = rim_release(ball, center-r*normal, normal, outward, velocity, 0,
                      inertia_ratio=inertia_ratio, push=push)
    bend = math.acos(float(np.clip(normal @ next_normal,-1,1)))
    return rim if rim is not None and rim.angle < bend-EPS else None


def _points(start, t, loss, impulse):
    a, b = _coeff(t, loss)
    return start.pos + a[..., None] * start.vel + b[..., None] * impulse


def _turns(start, impulse, loss, t):
    return sorted({0., float(t)} | {float(k) for v, b in zip(start.vel, impulse)
                  if v * b < 0 and 0 < (k := _vertex(v, b, loss)) < t})


def _samples(start, impulse, loss, t, spacing):
    """Output polyline only; collision uses the analytic curve."""
    # Each coordinate is monotone between knots. Bisect by whole subcurves
    # until the bounding box guarantees the requested maximum step in space.
    times = np.array(_turns(start, impulse, loss, t))
    P = _points(start, times, loss, impulse)
    while True:
        split = np.flatnonzero(np.linalg.norm(np.diff(P, axis=0), axis=1) > spacing)
        if not len(split):
            return times, P
        mids = (times[split] + times[split + 1]) * .5
        times = np.insert(times, split + 1, mids)
        P = np.insert(P, split + 1, _points(start, mids, loss, impulse), axis=0)


def _death_hit(start, impulse, loss, t, boxes, *, chords=False):
    if not len(boxes):
        return False
    if chords:
        logd = math.log1p(-loss)
        point = lambda z: _curve(start.pos,start.vel,impulse,z,loss,logd)
        ks = sorted({0.,t} | {float(k) for v,b in zip(start.vel,impulse) if 0 < (k := _vertex(v,b,loss)) < t})
        pts = np.array([point(k) for k in ks])
    else:
        point = lambda z: _points(start,z,loss,impulse)
        ks = _turns(start, impulse, loss, t)
        pts = _points(start, np.array(ks), loss, impulse)
    for a, b, p, q in zip(ks, ks[1:], pts, pts[1:]):
        for lo, hi in boxes[np.all((boxes[:, 0] <= np.maximum(p, q)) & (boxes[:, 1] >= np.minimum(p, q)), axis=1)]:
            left, right = a, b
            for i in range(3):
                sign = 1 if q[i] >= p[i] else -1
                x, y = (lo[i], hi[i]) if sign > 0 else (-hi[i], -lo[i])
                if x > sign * p[i]:
                    left = max(left, brentq(lambda z: sign * point(z)[i] - x, a, b))
                if y < sign * q[i]:
                    right = min(right, brentq(lambda z: sign * point(z)[i] - y, a, b))
            if left <= right:
                return True
    return False


@njit(cache=True)
def _face_candidate(tri, normal, p, v, impulse, end, loss, logd, r, start_at=0., incoming_only=False, side_only=False):
    # Project onto the plane normal. Its derivative can change sign at
    # only one integer knot; solve each incoming offset-plane crossing.
    vv, bb = np.dot(v, normal), np.dot(impulse, normal)
    side = np.dot(p - tri[0], normal)
    turn = 0.
    if vv * bb < 0:
        u = bb / loss
        turn = math.ceil(math.log(u / (u - vv)) / logd)
    ta = start_at
    for tb in (min(turn, end), end):
        if tb <= ta:
            continue
        aa, ba = _chord(ta, loss, logd)
        ab, b_end = _chord(tb, loss, logd)
        da = side + aa * vv + ba * bb
        db = side + ab * vv + b_end * bb
        sign = 1. if db < da else -1.
        if side_only and sign * normal[1] >= WALK_NY:
            ta = tb
            continue
        if sign * da >= r - EPS and sign * db <= r + EPS:
            # Find the PSI chord by monotone binary search in integer knot
            # indices, then intersect that chord exactly (no time stepping).
            l, h = math.floor(ta), math.ceil(tb)
            while h - l > 1:
                mid = (l+h)//2
                a, b = _chord(mid, loss, logd)
                if sign * (side + a * vv + b * bb) > r:
                    l = mid
                else:
                    h = mid
            a0, b0 = _chord(l, loss, logd)
            a1, b1 = _chord(h, loss, logd)
            left = sign * (side + a0 * vv + b0 * bb) - r
            right = sign * (side + a1 * vv + b1 * bb) - r
            h = l + min(1., max(0., left / (left - right))) if left > right else tb
            if start_at + EPS < h <= end + EPS:
                x = _curve(p, v, impulse, h, loss, logd) - sign * r * normal
                inside = True
                for j in range(3):
                    e = tri[(j + 1) % 3] - tri[j]
                    q = x - tri[j]
                    if np.dot(np.cross(e, q), normal) < -EPS:
                        inside = False
                        break
                if inside:
                    return h
        elif ta == 0 and abs(da) < r - EPS and (not incoming_only or da * vv < 0) and (
                not side_only or math.copysign(1., da) * normal[1] < WALK_NY):
            x = p - da * normal
            inside = True
            for j in range(3):
                if np.dot(np.cross(tri[(j + 1) % 3] - tri[j], x - tri[j]), normal) < -EPS:
                    inside = False
                    break
            if inside:
                return 0.
        ta = tb
    return -1.


@njit(cache=True)
def _intersects(lo, hi, a, b, r):
    for i in range(3):
        if lo[i] > max(a[i], b[i]) + r or hi[i] < min(a[i], b[i]) - r:
            return False
    return True


@njit(cache=True)
def _rail_contact(p, tris, faces, r):
    """Single upward support or opposing upward reactions on the departure rail."""
    horizontal = []
    for i in faces:
        a, b, c = tris[i]
        x, y, z, _ = _closest(p[0], p[1], p[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])
        dx, dy, dz = p[0] - x, p[1] - y, p[2] - z
        sq = dx*dx + dy*dy + dz*dz
        if sq > (r + EPS)**2 or dy <= EPS:
            continue
        if dy >= WALK_NY * math.sqrt(sq):
            return True
        h = math.sqrt(dx*dx + dz*dz)
        if h > EPS:
            for u, v in horizontal:
                if dx*u + dz*v < -RAIL_BALANCED*h:
                    return True
            horizontal.append((dx/h, dz/h))
    return False


@njit(cache=True)
def _rail_prefix(p, v, impulse, end, loss, r, tris, faces):
    """End of the connected upward-support interval, bounded to a local sphere."""
    if not len(faces) or not _rail_contact(p, tris, faces, r):
        return 0.
    logd = math.log1p(-loss)
    # Traverse interval bounds in chronological order. At an engine chord
    # solve the support union on its feature cells.
    work = [(0., end)]
    while work:
        a, b = work.pop()
        lo, hi = sweep.bounds(p, v, impulse, a, b, loss, logd)
        center = (lo + hi) * .5
        width = np.linalg.norm(hi - lo) * .5
        drift = np.linalg.norm(center - p)
        if drift + width <= r and sweep.rail_span(p,v,impulse,a,b,loss,logd,r,tris,faces,center,width):
            continue
        k, h = math.floor(a), math.ceil(b)
        if h-k<=1:
            x,y=_curve(p,v,impulse,a,loss,logd),_curve(p,v,impulse,b,loss,logd)
            s,t=sweep.sphere_span(x-p,y-x,r,0.,1.)
            if s>0. or t<0.:
                return -1.
            u=sweep.chord_rail_end(x,y,r,tris,faces)
            if u < 1. and u<=t:
                return a+(b-a)*u
            if t < 1.:
                return -1.
            continue
        m=float((k+h)//2)
        work.append((m,b)); work.append((a,m))
    return -1.


@njit(cache=True)
def _screen(p, v, impulse, end, loss, r, tris, normals, lo, hi, grid, pivot, axis, owner, rail,
            void_owner, void_until, incoming_only, support_owner=-1, side_only=False, void_faces=None):
    logd = math.log1p(-loss)
    knots = [0., end]
    for j in range(3):
        k = sweep.turn(v[j], impulse[j], loss, logd)
        if 0 < k < end:
            knots.append(float(k))
    knots.sort()
    # Each box encloses the entire subcurve between coordinate extrema.
    # These knots are geometry events, not uniformly spaced flight probes.
    P = np.empty((len(knots), 3))
    for k in range(len(knots)):
        P[k] = _curve(p, v, impulse, knots[k], loss, logd)
    ids = _box_triangles(np.minimum(P[:-1], P[1:]) - r, np.maximum(P[:-1], P[1:]) + r, lo, hi, grid)
    kind, tri, first = 0, -1, end + EPS
    front_lo, front_hi = sweep.bounds(p,v,impulse,0.,end,loss,logd)
    screen_t = first
    for i in ids:
        if side_only and owner[i]==support_owner:
            continue
        if first < screen_t:
            front_lo, front_hi = sweep.bounds(p,v,impulse,0.,first,loss,logd)
            screen_t = first
        if not _intersects(lo[i], hi[i], front_lo, front_hi, r):
            continue
        normal = normals[i]
        if not normal.any():
            continue
        ignore = 0.
        if void_until>0. and rail[i] and owner[i] == void_owner:
            if void_faces is None:
                ignore=void_until
            else:
                idx=np.searchsorted(void_faces,i)
                if idx<len(void_faces) and void_faces[idx]==i:ignore=void_until
        t = _face_candidate(tris[i], normal, p, v, impulse, min(end, first), loss, logd, r, ignore, incoming_only, side_only)
        if t >= 0 and (t < first or kind == 2 and t <= first + EPS):
            kind, tri, first = 1, i, t
        after = np.nextafter(ignore, math.inf) if ignore > 0. else 0.
        t = sweep.edge_time(tris[i], normal, lo[i], hi[i], p, v, impulse, loss, logd, r,
                            after, min(end,first), pivot, axis, incoming_only, side_only)
        if 0 <= t < first - EPS:
            kind, tri, first = 2, i, t
    return kind, tri, first


class Obstacles:
    def __init__(self, mesh, boxes):
        self.mesh = mesh
        self.tris = np.stack([mesh.A, mesh.B, mesh.C], axis=1)
        self.lo, self.hi = self.tris.min(1), self.tris.max(1)
        self.bottom = min(float(self.lo[:, 1].min()), 0.) if len(self.lo) else 0.
        self.boxes = np.array([[b["lo"], b["hi"]] for b in boxes], float).reshape(-1, 2, 3)

    def support_clear(self, start, impulse, loss, end, owner, radius):
        """Continuous wall/ceiling and death test for a supported curve."""
        if end <= EPS:
            return True
        if _death_hit(start, impulse, loss, end, self.boxes, chords=True):
            return False
        kind, _, _ = _screen(start.pos,start.vel,impulse,end,loss,radius,self.tris,self.mesh.N,
                             self.lo,self.hi,self.mesh._grid,np.empty(0),np.empty(0),self.mesh.owner,
                             self.mesh.rail,-1,0.,True,owner,True)
        return kind == 0

    def first_contact(self, ball, start, push, t, rim=None, departure=None, clearance=0., *, incoming_only=False):
        """First solid hit or death, not whether the commanded endpoint was hit."""
        loss, r = ball.lin_damp * ph.DT, ball.radius + FLOOR
        impulse = cf.air_impulse(ball, push=push)
        pivot = rim.pivot if rim is not None else np.empty(0)
        axis = rim.axis if rim is not None else np.empty(0)
        kind, i, hit_t = _screen(start.pos, start.vel, impulse, t, loss, r,
                                 self.tris, self.mesh.N, self.lo, self.hi, self.mesh._grid, pivot, axis,
                                 self.mesh.owner, self.mesh.rail, departure.owner if departure is not None else -1, clearance, incoming_only,
                                 void_faces=departure.faces if departure is not None and departure.axes else None)
        if _death_hit(start, impulse, loss, min(t, hit_t), self.boxes, chords=True):
            return 'death_box', -1, min(t, hit_t), None
        if kind == 0:
            return 'no_contact', -1, t, None
        center = _curve(start.pos,start.vel,impulse,hit_t,loss,math.log1p(-loss))
        if kind == 1:
            # Use the face API's exact chord contact when it is the selected
            # crossing. The projected broad phase also finds later entries
            # when the first crossing of the infinite plane misses the face.
            hit = cf.triangle_face_hit(ball, start, self.tris[i], radius=r, push=push)
            if hit is not None:
                when = hit.step - 1 + hit.fraction if hit.step else 0.
                if abs(when - hit_t) <= 1e-5:
                    hit_t, center = when, hit.center
            normal = self.mesh.N[i]
            normal = normal if (center - self.tris[i, 0]) @ normal >= 0 else -normal
            if normal[1] >= WALK_NY and not self.mesh.placed[i]:
                return 'land', i, hit_t, center
            return 'face', i, hit_t, center
        a, b, c = self.tris[i]
        x, y, z, _ = _closest(*center, *a, *b, *c)
        radial = center - (x, y, z)
        distance = np.linalg.norm(radial)
        if distance > EPS and radial[1] >= WALK_NY * distance and not self.mesh.placed[i]:
            return 'land', i, hit_t, center
        return 'edge', i, hit_t, center

    def rail_clearance(self, ball, start, push, end, departure):
        if departure is None:
            return 0.
        return _rail_prefix(start.pos, start.vel, cf.air_impulse(ball, push=push), end,
                            ball.lin_damp * ph.DT, ball.radius + FLOOR, self.tris, departure.faces)

    def blocked(self, ball, start, push, t):
        why, *_ = self.first_contact(ball, start, push, t)
        return why if why not in ('land', 'no_contact') else None


class LandingPlaces:
    def __init__(self, mesh, pos, contacts, surface=None):
        self.pos = pos
        self.mesh = mesh
        self.surface = surface
        self.by_face = defaultdict(list)
        self.by_rail = defaultdict(set)
        for v, cs in enumerate(contacts):
            for c in cs:
                self.by_face[c.triangle].append(v)
                if mesh.rail[c.triangle]:
                    self.by_rail[int(mesh.owner[c.triangle])].add(v)
        self.by_vertex = defaultdict(list)
        for i, corners in enumerate(zip(mesh.A, mesh.B, mesh.C)):
            if abs(mesh.N[i, 1]) < WALK_NY:
                continue
            for p in corners:
                self.by_vertex[tuple(np.round(p, 5))].append(i)
        self.patch_nodes = {}

    def near(self, face, center):
        if self.mesh.rail[face]:
            ids = self.by_rail.get(int(self.mesh.owner[face]))
            if ids:
                if not isinstance(ids, tuple):
                    points = np.array(sorted(ids), dtype=int)
                    ids = (points, cKDTree(self.pos[points]))
                    self.by_rail[int(self.mesh.owner[face])] = ids
                distance, j = ids[1].query(center)
                return int(ids[0][j]) if distance <= 3 * STEP else -1
        if face not in self.patch_nodes:
            mesh = self.mesh
            corners = (mesh.A[face], mesh.B[face], mesh.C[face])
            neighbors = Counter(i for p in corners for i in self.by_vertex[tuple(np.round(p, 5))])
            # A shared edge connects the adjoining supporting facets; avoid
            # a different deck which happens to have a nearby coarse point.
            ids = set(self.by_face[face])
            for i, count in neighbors.items():
                if count >= 2 and abs(mesh.N[face] @ mesh.N[i]) >= WALK_NY:
                    ids.update(self.by_face[i])
            self.patch_nodes[face] = np.array(sorted(ids), dtype=int)
        ids = self.patch_nodes[face]
        if len(ids):
            D = np.linalg.norm(self.pos[ids] - center, axis=1)
            j=int(D.argmin())
            if D[j] <= 3*STEP:return int(ids[j])
        if self.surface is not None and (k:=self.surface.face_patch[face])>=0:
            ids={i for f in self.surface.patches[k].faces for i in self.by_face[f]}
            if ids:
                ids=np.array(sorted(ids)); D=np.linalg.norm(self.pos[ids]-center,axis=1); j=int(D.argmin())
                if D[j]<=3*STEP:return int(ids[j])
        return -1


def _unit(v):
    length = np.linalg.norm(v)
    return v / length if length > EPS else np.zeros_like(v)


def _impact(obs, kind, i, center):
    tri = obs.tris[i]
    if kind == 'face':
        normal = obs.mesh.N[i]
        h = (center - tri[0]) @ normal
        return center - h * normal, normal if h >= 0 else -normal
    point = np.array(_closest(*center, *tri[0], *tri[1], *tri[2])[:3])
    return point, _unit(center - point)


@dataclass
class AirTrace:
    # Retain the two-value unpacking API used by the graph and local probes.
    result: tuple
    parts: list
    contacts: list
    bounces: int

    def __iter__(self):
        return iter((self.result, self.parts))


def trace_air(obs, ball, start, schedule, *, rim=None, departure=None,
              clearance=0., cached_first=None, max_bounces=MAX_BOUNCES,
              follow_below=False, incoming_only=False):
    """Events, not ticks: one free curve up to a key change or first contact."""
    loss, radius = ball.lin_damp * ph.DT, ball.radius + FLOOR
    now, bounces, parts, contacts = 0., 0, [], []
    body, first = start, True
    episode_end, impacted_faces = -math.inf, set()
    for boundary, push in schedule:
        extend = follow_below and boundary == schedule[-1][0]
        while True:
            # The target horizon describes an unreflected representative only.
            # After an impact, continue until below all geometry, if necessary.
            remaining = boundary - now
            if remaining <= EPS:
                if not (extend and bounces):
                    break
                below = min(obs.bottom - radius, body.pos[1] - radius)
                remaining = float(cf.landing_step(ball, body, below) or 0)
                extend = False
                boundary = now + remaining
            # Unlike the legacy lip test, the finite joint prefix can span
            # several immutable input runs, but never a bounce or re-entry.
            finite=departure is not None and bool(departure.axes) and not contacts
            local_clear=max(0.,clearance-now) if finite else (clearance if first else 0.)
            void_owner=departure.owner if departure is not None and local_clear>0. else -1
            void_faces=departure.faces if finite else None
            if first or (finite and local_clear>0.):
                result=(cached_first if first and cached_first is not None else
                        obs.first_contact(ball,body,push,remaining,rim if first else None,departure,local_clear,
                                          incoming_only=incoming_only))
                first=False
            else:
                result=obs.first_contact(ball,body,push,remaining,incoming_only=(incoming_only or bool(bounces)))
            kind, i, dt, center = result
            parts.append((body, push, dt))
            now += dt
            if kind == 'no_contact':
                if (extend and bounces) or boundary != schedule[-1][0]:
                    impulse = cf.air_impulse(ball, push=push)
                    body = cf.flight(ball, body, int(dt), push=push)
                    # Normally key events are at integer PSI times. A bounce
                    # can leave a fractional interval up to a fixed key event.
                    previous_body = parts[-1][0]
                    end = _curve(previous_body.pos,previous_body.vel,impulse,dt,loss,math.log1p(-loss))
                    body = ph.State(end, body.vel, body.omega)
                continue
            if kind == 'death_box':
                return AirTrace((kind, i, now, center), parts, contacts, bounces)
            entity = obs.mesh.entities[obs.mesh.owner[i]]
            point, normal = _impact(obs, kind, i, center)
            if kind == 'land' and np.linalg.norm(normal) <= EPS:
                normal = obs.mesh.N[i]
            same_psi = bool(contacts) and now <= episode_end+EPS
            step = max(1, math.ceil(dt))
            at_hit = body if same_psi else cf.flight(ball, body, step, push=push)
            if not same_psi:
                episode_end = now+step-dt
                impacted_faces = set()
            if not same_psi and dt<=EPS and (center-point)@normal < radius-FLOOR:
                # ASSUMED recovery for a deep initial overlap (fitted lane or
                # lip release). It has no common entry root at this centre.
                # Resolve the first selected feature, then collect any
                # connected co-active geometry in the same PSI episode.
                rows = [(int(i),point,normal,float((center-point)@normal))]
            else:
                rows = contactset.contact_set(obs,center,radius,i,point,normal,void_owner if dt<=local_clear else -1,void_faces)
            ns = np.array([r[2] for r in rows]); gaps = np.array([r[3] for r in rows])
            # A supporting upward reaction, or a combination of two opposing
            # rail reactions at this same centre. No history-based pairing.
            supports = [(f,n) for f,_,n,_ in rows if n[1]>=WALK_NY and not obs.mesh.placed[f]]
            dual = None
            if not supports:
                rails = [(f,q,n) for f,q,n,_ in rows if obs.mesh.rail[f] and not obs.mesh.placed[f] and n[1]>EPS]
                if rails:
                    # V9 allowed one additional floor-skin width for a paired
                    # rail reaction. Evaluate both features at this centre;
                    # no previously-hit triangle is needed.
                    known = {f for f,_,_ in rails}
                    rails.extend((c.triangle,c.point,c.normal) for c in obs.mesh.contacts(center,radius,FLOOR)
                                 if c.triangle not in known and obs.mesh.rail[c.triangle] and not obs.mesh.placed[c.triangle]
                                 and c.normal[1]>EPS and not (dt<=local_clear and c.entity==void_owner and (void_faces is None or c.triangle in void_faces)))
                for j,(f,q,n) in enumerate(rails):
                    for g,q2,m in rails[:j]:
                        combined = contactset.opposed_support(n,m)
                        if combined is not None and np.linalg.norm(q-q2)>EPS and combined[1]>=WALK_NY:
                            dual = (f,combined, (dict(triangle=int(f),point=q,normal=n),
                                                dict(triangle=int(g),point=q2,normal=m)))
                            break
                    if dual is not None:
                        break
            event = dict(kind=kind, triangle=int(i), object=entity.get('name', ''), tick=float(now),
                         position=center.copy(), normal=normal, incoming=at_hit.vel.copy(), point=point, incoming_omega=at_hit.omega.copy())
            if len(rows)>1:
                event['manifold']=tuple(dict(triangle=int(f),normal=n,point=q) for f,q,n,_ in rows)
            if same_psi:
                prior=contacts[-1]
                prior.setdefault('subcontacts',[]).append(event)
                event['episode_start']=prior['tick']
            else:
                contacts.append(event)
            if supports or dual is not None:
                f,n = (max(supports,key=lambda r:r[1][1]) if supports else dual[:2])
                if dual is not None:event['manifold']=dual[2]
                event['support_normal']=n
                event['point'], event['normal'] = _impact(obs,'edge',f,center)
                event['triangle']=int(f)
                event['object']=obs.mesh.entities[obs.mesh.owner[f]].get('name','')
                event['kind'] = 'dual_rail' if dual is not None else 'land'
                if same_psi:
                    event['subcontacts']=[prior,*prior['subcontacts'][:-1]]
                    prior['subcontacts'].pop()  # keep history acyclic
                    contacts[-1]=event
                return AirTrace(('land', f, now, center), parts, contacts, bounces)
            if same_psi and i in impacted_faces:
                return AirTrace(('contact_jam',i,now,center),parts,contacts,bounces)
            impacted_faces.update(f for f,*_ in rows)
            if not same_psi and bounces >= max_bounces:
                return AirTrace(('bounce_limit', i, now, center), parts, contacts, bounces)
            if not normal.any():
                return AirTrace(('undefined_contact_normal', i, now, center), parts, contacts, bounces)
            if len(rows)==1 and not same_psi:
                offset = point + (radius + EPS * 8) * normal
                body, regime = cf.collision_impulse(ball, ph.State(offset, at_hit.vel, at_hit.omega), normal,
                              float(entity.get('elasticity', DEFAULT_ELASTICITY)))
                event['impulse_regime']=regime
            else:
                # A compressive combination of concurrent normal impulses.
                e=np.array([cf.collision_restitution(ball,obs.mesh.entities[obs.mesh.owner[f]].get('elasticity',DEFAULT_ELASTICITY))
                            for f,_,_,_ in rows])
                vn=ns@at_hit.vel
                goal=np.zeros(len(ns)) if same_psi else np.maximum(0.,-e*vn)
                dv=contactset.minimum_correction(ns,goal-vn)
                if dv is None:dv=contactset.minimum_correction(ns,-vn)
                offset=contactset.minimum_correction(ns,radius+EPS*8-gaps)
                if dv is None or offset is None:
                    return AirTrace(('contact_jam',i,now,center),parts,contacts,bounces)
                event['reaction_normal']=_unit(dv)
                event['impulse_regime']='normal_manifold'
                body=ph.State(center+offset,at_hit.vel+dv,at_hit.omega.copy())
            event['outgoing_omega']=body.omega.copy()
            event['outgoing'] = body.vel.copy()
            if same_psi:
                contacts[-1]['outgoing']=body.vel.copy()
            else:
                bounces += 1
    return AirTrace(('no_contact', -1, now, None), parts, contacts, bounces)




def trajectory(obs, ball, start, t, push, switch, rim, departure=None, clearance=0., cached_first=None):
    schedule = ((switch, push[0]), (t, push[1])) if switch else ((t, push),)
    return trace_air(obs, ball, start, schedule, rim=rim, departure=departure,
                     clearance=clearance, cached_first=cached_first, follow_below=True)


def densify_way(points):
    """Split long polyline chords without changing their geometry."""
    P = np.asarray(points, float)
    n = np.maximum(1, np.ceil(np.linalg.norm(np.diff(P, axis=0), axis=1) - EPS).astype(int))
    if not np.any(n > 1):
        return np.asarray(points, dtype=np.float32)
    edge = np.repeat(np.arange(len(n)), n)
    f = (np.arange(n.sum()) + 1 - np.repeat(np.cumsum(n) - n, n)) / n[edge]
    return np.vstack([P[0], P[edge] + f[:, None] * (P[edge + 1] - P[edge])]).astype(np.float32)


def air_geometry(ball, a, push, ticks):
    return motion.flight(a,cf.air_impulse(ball,push=push),ball.lin_damp*ph.DT,ticks)


def _run_path(P,lengths,chain,lip,track):
    if track is None:
        return np.vstack([lip,P[chain]]), np.r_[0.,lengths]
    points,arc=track
    return np.vstack([lip,points]),np.r_[0.,lengths[0]+arc]


def route_geometry(P,lengths,chain,L,lip,rim,ball,parts,center,source,target,track=None):
    before,lens=_run_path(P,lengths,chain,lip,track)
    t = np.r_[L,lens[(lens>0.) & (lens<L)][::-1],0.]
    support = np.column_stack([np.interp(t,lens,x) for x in before.T])
    pre = motion.lines(np.vstack([P[source],support]))
    arc = (rim.motion(),) if rim is not None and rim.seconds else ()
    return (*pre,*arc,*(air_geometry(ball,a,p,n) for a,p,n in parts),motion.line(center,P[target]))


def continued_motion(edge, pieces, center, landing):
    prefix = motion.path_for(edge)
    return (*prefix[:-1],*(x for p in pieces for x in p['motion']),motion.line(center,landing))


def _way(P, lengths, chain, L, lip, rim, ball, parts, center, source, target, track=None):
    before,lens=_run_path(P,lengths,chain,lip,track)
    steps = np.r_[L, np.arange(math.ceil(L) - 1, 0, -1), 0.]
    support = np.column_stack([np.interp(steps, lens, v) for v in before.T])
    arc = rim.way() if rim is not None and rim.seconds else np.empty((0, 3))
    loss = ball.lin_damp * ph.DT
    flight = []
    for start, push, ticks in parts:
        _, Q = _samples(start, cf.air_impulse(ball, push=push), loss, ticks, 1.)
        flight.append(Q)
    return densify_way(np.vstack([P[source], support, arc, *flight, center, P[target]]))


def _target_coeff(ball, start, heights):
    """A and B at the descending crossing of the requested heights."""
    loss = ball.lin_damp * ph.DT
    h = start.pos[1] - heights
    vy = start.vel[1]
    a = _landing_drift(ball, vy, 1., h)
    b = np.maximum(0., (h + vy * a) / GRAVITY_IMPULSE)
    return a, b


def rail_seed_key(site, source, target, ball, capture):
    # ASSUMED quantization: retain one speed interval per local floor-skin
    # escape speed, separately for the transverse component. This is a search
    # resolution, not a claim of dominance of different states in a bin.
    if getattr(capture,'crown',None) is not None:
        return (site,source,ball.name,capture.lane,capture.direction,capture.j,
                capture.body.pos.tobytes(),capture.body.vel.tobytes(),
                capture.rebound.vel.tobytes() if capture.rebound is not None else None)
    dv = math.sqrt(2*ph.GRAVITY*FLOOR)
    return (site, source, ball.name, capture.lane, capture.direction, capture.j,
            target, math.floor(np.linalg.norm(capture.body.vel)/dv), math.floor(capture.lateral/dv))


def road_targets(targets, ball, origin, direction, slope, vmax, top, rim, stats, node_targets=None):
    if not targets.surface.patches:return
    radius = ball.radius+FLOOR
    ratio = ball.push/GRAVITY_IMPULSE
    loss = ball.lin_damp*ph.DT
    patches=targets.surface.patches

    def floor_height(p):
        return p.heights(radius)[0]

    def cap(y):
        if vmax<=EPS:return 0.
        return float(np.nan_to_num(_landing_drift(ball,slope,vmax,origin[1]-y),nan=vmax*ph.DT/loss))

    y=targets.lowest
    # Conservative minimum (without skin) can be shared by all ball kinds.
    drift=cap(y)
    width=ratio*max(0.,origin[1]-y+max(0.,slope*drift))
    end=origin[XZ]+drift*direction
    lo,hi=np.minimum(origin[XZ],end)-width,np.maximum(origin[XZ],end)+width
    fixed=None
    if rim is not None:
        st=rim.start
        fixed=rail_inverse.ReleaseLaw(st.pos,st.vel,np.zeros(3),np.zeros(3),np.zeros(3),-1.)
        outer=rail_inverse.reach_planes(sys.modules[__name__],ball,fixed,0.,0.,y,math.inf,allow_rise=True)
        if outer is not None:lo,hi=np.minimum(lo,outer[0]),np.maximum(hi,outer[1])

    def bounds(p):
        bottom=floor_height(p)
        result=[]
        if bottom<=top+EPS:
            n,h=conformal.drift_planes(p,radius,origin,direction,slope,ratio,cap(bottom))
            # Reject the part above even the fastest rise.
            result.append((np.r_[n,[-p.normal[XZ]/p.normal[1]]],np.r_[h,top-(p.offset+radius)/p.normal[1]]))
        if fixed is not None:
            x=rail_inverse.reach_planes(sys.modules[__name__],ball,fixed,0.,0.,bottom,p.heights(radius)[1],allow_rise=True)
            if x is not None: result.append(x[2])
        return result

    def valid(p):
        stats['road_exact_checks']+=1
        a,b=reachable(ball,p-origin,direction,slope,vmax)
        if a[0]<=b[0]:return True
        if rim is None:return False
        A,B=_target_coeff(ball,rim.start,p[1])
        return np.max(abs(p[XZ]-rim.start.pos[XZ]-A*rim.start.vel[XZ]))<=B*ball.push+EPS

    def exact(patch,pieces):
        return conformal.speed_witnesses(patch,pieces,radius,origin,direction,slope,ratio,
              cap(floor_height(patch)),vmax,loss,ph.DT,GRAVITY_IMPULSE,_coeff)

    yield from targets.representatives(ball.name,radius,lo,hi,bounds,valid,stats=stats,prefix='road',
                                       anchor=origin[XZ],exact=exact,alternatives=True,spine=(origin[XZ],direction),
                                       node_targets=node_targets)


def add_launches(g, mesh, info=None, *, log=lambda *_: None, transformer_kind=None, include_carry=True):
    """Build proposals without calling a physics rollout. Mutate the coarse graph."""
    info = info or {}
    t0 = perf_counter()
    P = g.pos
    obs = Obstacles(mesh, info.get("death_boxes", ()))
    # Correct centre heights for the measured contact skin, including slopes.
    land = P.copy()
    contacts = [_support(mesh, p, ph.WOOD.radius) if v < g.info['rest_places']
                else tuple(c for c in mesh.contacts(p, ph.WOOD.radius, RAIL_CONTACT_QUERY)
                           if mesh.rail[c.triangle] and c.normal[1] > .05)
                for v, p in enumerate(P)]
    for v, c in enumerate(contacts):
        land[v, 1] += ((FLOOR - c[0].gap) / c[0].normal[1] if c and c[0].normal[1] >= WALK_NY else FLOOR)
    surface = g.info.get('conformal')
    if surface is None:
        surface = g.info['conformal'] = conformal.build(mesh, rails.physical_lanes(g, mesh))
    landing_places = LandingPlaces(mesh, P, contacts, surface)
    near_goal = np.zeros(len(P), bool)
    for p in [*info.get("checkpoints", ()), info.get("airship")]:
        if p is not None:
            near_goal |= np.linalg.norm(P - p, axis=1) <= GOAL_NEAR
    labels = {b.name: witness_labels(g, info, transformer_kind, b.name) for b in ph.BALLS.values()}
    targets = conformal.Targets(surface, P, landing_places.by_face, labels, g.info['rest_places'])
    landing_places.conformal_targets = targets
    residual = np.r_[targets.unmapped, np.arange(g.info['rest_places'],len(P))]
    bands = [(a,b,residual[ids],tree) for a,b,ids,tree in _bands(land[residual])]
    band_bottom = np.array([b[0] for b in bands])
    stats = Counter()
    candidates = []
    from .railride import RailPaths
    from . import supportcarry as sc
    floor_seeds = {}
    seed_paths = RailPaths(g,obs,landing_places) if np.any(mesh.rail) else None
    rail_seeds = {}
    best = {}
    directions = list(_sites(g, mesh, obs.tris, obs.lo, obs.hi))
    log("takeoff: %d places, %d sites, %d outgoing directions, 3 balls" %
        (len(P), len({e[0] for e in directions}), len(directions)))
    # Finish the constant-input pass for every direction at a site before
    # spending time on switch schedules. Never use the aimed-at component
    # to prune that pass: the first support can be a different destination.
    directions.sort(key=lambda e: e[0])
    for _, rows in groupby(directions, key=lambda e: e[0]):
        rows = list(rows)
        prefix_cache = {}  # scoped to this site; exact equality, no quantization
        for fallback in (False, True):
            for site, key, q, d, slope, normal, chain, lengths, track in rows:
                tangent = np.array([d[0], slope, d[1]])
                scale = np.linalg.norm(tangent)
                floor = site < g.info['rest_places']
                departure = None if floor else rail_support(obs, q, ph.WOOD.radius)
                if floor:
                    pivot, outward = lip_basis(obs, q, normal, tangent, ph.WOOD.radius + FLOOR)
                    support_n, rho = normal, ph.WOOD.radius + FLOOR
                else:
                    cap = rail_lip(obs, q, normal, tangent, g.lanes[g.lane[site]]['double'], ph.WOOD.radius)
                    if cap is not None:
                        pivot, support_n, outward, rho = cap
                        new_q = pivot + rho * support_n
                        lengths = lengths + max(0., float((new_q - q)[XZ] @ d))
                        q = new_q
                rimmed = floor or cap is not None
                for ball in ph.BALLS.values():
                    ball_labels = labels[ball.name]
                    source_components = tuple(map(int, np.unique(ball_labels[chain])))
                    loss = ball.lin_damp * ph.DT
                    axis = cf.rolling(ball, normal, key).along(tangent)
                    terminal = axis.terminal_speed
                    vmax = runway_speed(axis, lengths[-1] * scale) / scale
                    rim_zero = rim_release(ball, pivot, support_n, outward, np.zeros(3), key, rho) if rimmed else None
                    vy = vmax * slope
                    peak_a, peak_b = _coeff(_peak(vy, loss), loss)
                    top = q[1] + max(0., vy * peak_a - GRAVITY_IMPULSE * peak_b)
                    count = int(np.searchsorted(band_bottom, top + EPS, side="right"))
                    depths = q[1] - band_bottom[:count]
                    drifts = _landing_drift(ball, slope, vmax, depths) if vmax > EPS else np.zeros(count)
                    drifts = np.nan_to_num(drifts, nan=vmax * ph.DT / loss)
                    halves = ball.push / GRAVITY_IMPULSE * np.maximum(0., depths + np.maximum(0., slope * drifts))
                    if rim_zero is not None:
                        # Broad-phase union with the independently aimed from-rest rim profile.
                        zA, zB = _target_coeff(ball, rim_zero.start, band_bottom[:count])
                        zC = (rim_zero.start.pos - q)[XZ] + zA[:, None] * rim_zero.start.vel[XZ]
                    def batches():
                        for k_band, ((_, _, ids, tree), drift, half) in enumerate(zip(bands, drifts, halves)):
                            center = q[XZ] + d * (.5 * drift)
                            reach = half + .5 * drift + EPS
                            if rim_zero is not None and np.isfinite(zB[k_band]):
                                reach = max(reach, zB[k_band] * ball.push + np.linalg.norm(q[XZ] + zC[k_band] - center, np.inf) + EPS)
                            local = tree.query_ball_point(center, reach, p=np.inf)
                            if local:
                                selected = ids[np.asarray(local)]
                                selected = selected[(selected != site) & (land[selected,1] <= top+EPS)]
                                if len(selected):yield selected,land[selected]
                        local_stats = stats if not fallback else Counter()
                        reps=list(road_targets(targets,ball,q,d,slope,vmax,top,rim_zero,local_stats,land))
                        if not reps:return
                        if not fallback:
                            candidates.extend((site,key,ball.name,k,label,v,*map(float,p)) for v,p,k,label in reps)
                        yield np.array([v for v,_,_,_ in reps]),np.array([p for _,p,_,_ in reps])

                    for ids, points in batches():
                        if not fallback:
                            stats['candidates'] += len(ids)
                        D = points - q
                        low, high = reachable(ball, D, d, slope, vmax)
                        nominal = low <= high
                        if rim_zero is not None:
                            rA, rB = _target_coeff(ball, rim_zero.start, points[:, 1])
                            rDelta = points[:, XZ] - rim_zero.start.pos[XZ] - rA[:, None] * rim_zero.start.vel[XZ]
                            zero = np.max(np.abs(rDelta), axis=1) <= rB * ball.push + EPS
                        else:
                            zero = np.zeros(len(ids), bool)
                        yes = np.flatnonzero(nominal | zero)
                        if not fallback:
                            stats['reachable'] += len(yes)
                        if not len(yes):
                            continue
                        nominal_drift = np.where(nominal, low, 0.)
                        times = _time_at_b(np.maximum(0., (-D[:, 1] + slope * nominal_drift) / GRAVITY_IMPULSE), loss)
                        rank = np.argsort(times[yes], kind='stable')
                        for j in yes[rank]:
                            target = int(ids[j])
                            group = int(ball_labels[target])
                            prefix = (site, ball.name)
                            if fallback and all(
                                    (*prefix, source, 0, group) in best
                                    and (not near_goal[target] or (*prefix, source, 1, target) in best)
                                    for source in source_components):
                                stats['fallback_group_already_witnessed'] += 1
                                continue
                            target_p = points[j]
                            bases = []
                            tried_speeds = []

                            def representatives(fractions, from_rest=False):
                                if from_rest and zero[j]:
                                    tried_speeds.append(0.)
                                    yield (0., rim_zero, rim_zero.start, float(rA[j]), float(rB[j]), rDelta[j], 'rim_zero')
                                if not nominal[j]:
                                    return
                                for f in fractions:
                                    if f and high[j] - low[j] <= EPS:
                                        continue
                                    s = (1 - f) * low[j] + f * high[j]
                                    B0 = max(0., (-D[j, 1] + slope * s) / GRAVITY_IMPULSE)
                                    n0 = _time_at_b(B0, loss)
                                    A0, _ = _coeff(n0, loss)
                                    speed = float(s / A0)
                                    if any(abs(speed - tried) <= EPS for tried in tried_speeds):
                                        continue
                                    tried_speeds.append(speed)
                                    end = rim_release(ball, pivot, support_n, outward, tangent * speed, key, rho) if rimmed else None
                                    if rimmed and end is None:
                                        continue
                                    start = end.start if end is not None and end.seconds else ph.State(q, tangent * speed)
                                    if end is not None and end.seconds:
                                        a, b = _target_coeff(ball, start, target_p[1])
                                    else:
                                        a, b = A0, B0
                                    dx = target_p[XZ] - start.pos[XZ] - a * start.vel[XZ]
                                    if np.max(np.abs(dx)) <= b * ball.push + EPS:
                                        yield (speed, end, start, a, b, dx, 'primary' if f in (0., 1.) else 'extra_drift')

                            for phase in ((0, 1, 2) if fallback else (0,)):
                                reached_group = False
                                switched = phase == 2
                                options = bases if switched else representatives((0., 1.) if phase == 0 else (.5, .25, .75), phase == 0)
                                for option in options:
                                    if not switched:
                                        bases.append(option)
                                    speed, rim, start, _, b, dx, stage = option
                                    n = float(_time_at_b(b, loss))
                                    if n <= 1 + EPS and b <= EPS:
                                        continue
                                    n1 = (math.log1p(-min(speed * scale / terminal, 1 - 1e-15)) / math.log1p(-axis.loss)
                                          if terminal > 0 and speed > 0 else 0.)
                                    L = max(0., float(axis.at(0., n1)[0]) / scale) if n1 else 0.
                                    w = chain[min(int(np.searchsorted(lengths, max(0., L - EPS))), len(chain) - 1)]
                                    rim_ticks = rim.seconds / ph.DT if rim is not None else 0.
                                    if fallback and phase == 0:
                                        continue
                                    w_component = int(ball_labels[w])
                                    if fallback and (*prefix, w_component, 0, group) in best and (
                                            not near_goal[target] or (*prefix, w_component, 1, target) in best):
                                        continue
                                    if switched:
                                        switches = [(p, late, mask) for f in (.25, .5, .75) if 0 < (p := int(n * f)) < n
                                                    for late in (True, False) for mask in (3, 1, 2)]
                                    else:
                                        switches = [(0, False, 0)]
                                    for switch, late, mask in switches:
                                        if switch:
                                            after = float(_coeff(n - switch, loss)[1])
                                            gain = after if late else b - after
                                            active = np.array([bool(mask & 1), bool(mask & 2)])
                                            if gain <= EPS or np.any(np.abs(dx[active]) > ball.push * gain + EPS):
                                                continue
                                            push = dx / (np.where(active, gain, b) * ball.push)
                                            push = np.clip(push, -1, 1)
                                            off = np.where(active, 0., push)
                                            inputs = (off, push) if late else (push, off)
                                        else:
                                            push = np.clip(dx / (b * ball.push), -1, 1)
                                            inputs = push
                                        shape = 'extra_switch' if switch else stage
                                        cached_first = None
                                        if switch:
                                            if np.array_equal(inputs[0], inputs[1]):
                                                continue  # exactly the constant representative already tried
                                            cache_key = (ball.name, key, switch, start.pos.tobytes(), start.vel.tobytes(),
                                                         inputs[0].tobytes())
                                            if cache_key not in prefix_cache:
                                                rail_clear = obs.rail_clearance(ball, start, inputs[0], switch, departure)
                                                cached_first = (obs.first_contact(ball, start, inputs[0], switch, rim, departure, rail_clear)
                                                                if rail_clear >= 0 else None)
                                                prefix_cache[cache_key] = rail_clear, cached_first
                                            else:
                                                stats['switch_cached_prefix'] += 1
                                                rail_clear, cached_first = prefix_cache[cache_key]
                                        else:
                                            rail_clear = obs.rail_clearance(ball, start, push, n+1., departure)
                                        if rail_clear < 0:
                                            stats['rail_not_released'] += 1
                                            continue
                                        stats['trial_' + shape] += 1
                                        # The final height is a surface sample, not a time
                                        # limit. Allow its next PSI chord for contact skin.
                                        trace = trajectory(
                                            obs, ball, start, n + 1., inputs, switch, rim, departure, rail_clear, cached_first)
                                        (why, face, flight_ticks, hit), parts = trace
                                        if trace.bounces:
                                            stats['bounce_representatives'] += 1
                                            stats['bounce_hits'] += trace.bounces
                                        if why != 'land':
                                            stats[why] += 1
                                            continue
                                        if flight_ticks <= rail_clear + EPS:
                                            stats['never_airborne'] += 1
                                            continue
                                        v = landing_places.near(face, hit)
                                        if v < 0:
                                            stats['no_landing_node'] += 1
                                            continue
                                        if v == w:
                                            stats['returned_to_source'] += 1
                                            continue
                                        stats['clear_representative'] += 1
                                        if trace.bounces:
                                            stats['bounce_landings'] += 1
                                        if trace.contacts[-1]['kind'] == 'dual_rail':
                                            stats['dual_rail_landings'] += 1
                                        actual = int(ball_labels[v])
                                        if actual != group:
                                            stats['redirect_other_component'] += 1
                                        elif v != target:
                                            stats['redirect_same_component'] += 1
                                        actual_keys = [(*prefix, w_component, 0, actual)]
                                        patch = surface.face_patch[face]
                                        if patch>=0:
                                            actual_keys.append((*prefix,w_component,2,(actual,int(patch))))
                                        if near_goal[v]:
                                            actual_keys.append((*prefix, w_component, 1, v))
                                        duration = n1 + rim_ticks + flight_ticks
                                        needed = [k for k in actual_keys if k not in best or best[k][2]['ticks'] > duration]
                                        seed_key = None
                                        captured = None
                                        if seed_paths is not None and mesh.rail[face]:
                                            captured = seed_paths.capture(ball,trace.contacts[-1],v)
                                            if captured is not None:
                                                k = rail_seed_key(site,int(w),v,ball,captured)
                                                if k not in rail_seeds or duration < rail_seeds[k][2]['ticks']:
                                                    seed_key = k
                                        floor_key = None
                                        if include_carry and not mesh.rail[face]:
                                            h=trace.contacts[-1]; n_hit=h['normal']; incoming=h['incoming']
                                            # rebound changes the normal only: the tangential slip
                                            # gate is exactly the first test in SurfacePaths.impact.
                                            if sc.has_carry_speed(ball,n_hit,sc.rolling_entry(ball,incoming,n_hit)):
                                                k=(site,int(w),ball.name,int(face),hit.tobytes(),incoming.tobytes(),
                                                   switch,tuple(p.tobytes() for p in inputs))
                                                if k not in floor_seeds or duration < floor_seeds[k][2]['ticks']:
                                                    floor_key=k
                                        if needed or seed_key is not None or floor_key is not None:
                                            first_push = inputs[0] if switch else push
                                            release_pos = (_points(start, rail_clear, loss, cf.air_impulse(ball, push=first_push))
                                                           if rail_clear else start.pos)
                                            release_vel = (cf.flight(ball, start, int(rail_clear), push=first_push).vel
                                                           if rail_clear else start.vel)
                                            e = dict(site=site, ball=ball.name, ticks=float(duration), time=float(duration / 132.),
                                                     flight_ticks=float(max(0., flight_ticks - rail_clear)),
                                                     rail_support_prefix_ticks=float(rail_clear), rim_ticks=float(rim_ticks), runup=float(L),
                                                     lip_speed=float(speed * scale), launch_speed=float(np.linalg.norm(release_vel)),
                                                     launch_position=release_pos.copy(), launch_velocity=release_vel.copy(),
                                                     keys_in_air=push.copy(), switch_tick=switch,
                                                     air_inputs=tuple(p.copy() for p in inputs) if switch else (push.copy(),),
                                                     landing_position=hit.copy(), landing_face=int(face), representative=shape,
                                                     bounce_count=trace.bounces, contacts=tuple(trace.contacts),
                                                     across=float(np.linalg.norm((hit - release_pos)[XZ])), down=float(release_pos[1] - hit[1]))
                                            e['way'] = _way(P, lengths, chain, L, q, rim, ball, parts, hit, w, v, track)
                                            e['motion'] = route_geometry(P,lengths,chain,L,q,rim,ball,parts,hit,w,v,track)
                                            for k in needed:
                                                best[k] = (w, v, e)
                                            if seed_key is not None:
                                                rail_seeds[seed_key] = (int(w),v,e,captured)
                                            if floor_key is not None:
                                                floor_seeds[floor_key] = (int(w),v,e)
                                        if fallback and actual == group and (not near_goal[target] or v == target):
                                            reached_group = True
                                            break
                                    if reached_group:
                                        break
                                if reached_group:
                                    break
    pairs = {}
    for w, v, e in best.values():
        per_ball = pairs.setdefault((int(w), v), {})
        b = e['ball']
        if b not in per_ball or e['ticks'] < per_ball[b]['ticks']:
            per_ball[b] = e
    launches, counts, shapes = {}, Counter(), Counter()
    for (w, v), per_ball in sorted(pairs.items()):
        es = tuple(sorted(per_ball.values(), key=lambda e: (e['ticks'], e['ball'])))
        launches[w, v] = dict(es[0], variants=es)
        g.adj[w].append((v, cost(es[0]), 'launch'))
        counts.update(per_ball.keys())
        shapes.update(e['representative'] for e in es)
    g.info['launches'] = launches
    g.info['conformal_launch_candidates'] = candidates
    g.info['_rail_seeds'] = tuple(rail_seeds.values())
    g.info['_floor_seeds'] = tuple(floor_seeds.values())
    stats['floor_seeds']=len(floor_seeds)
    g.info['takeoff_stats'] = dict(stats, edges=dict(counts), retained_by_representative=dict(shapes), groups=len(best), pairs=len(launches), seconds=perf_counter() - t0)
    if include_carry:
        add_continuations(g, mesh, info, obs=obs, landing=landing_places)
        add_rail_departures(g,mesh,info,obs=obs,landing=landing_places,transformer_kind=transformer_kind)
        g.info['takeoff_stats']['seconds'] = perf_counter() - t0
    log('takeoff: %s' % g.info['takeoff_stats'])
    return len(launches)


def carry_contact(paths, ball, contact, node, *, coast=False, controls=None, clock=0., initial_omega=None, aim=None):
    """Follow supported/air events after a known impact, without time integration.

    controls optionally provides recorded input for replay audits. Normal
    graph proposals hold forward along support, with a floor-coast witness.
    """
    from . import supportcarry as sc
    import sys
    tf = sys.modules[__name__]
    obs = paths.obs
    pieces, way, contacts = [], [], []
    air_ticks = bounces = elapsed = 0.
    omega = initial_omega
    bearings={}
    def finish(**out):
        if bearings:
            out['bearings']=tuple(bearings.values())
            if out.get('reason')!='handoff':out['bearing']=next(reversed(bearings.values()))
        return out
    for count in range(sc.MAX_CARRIES + 1):
        if controls is not None:
            chosen = controls(clock+elapsed)[0][1]
        elif coast and not obs.mesh.rail[contact['triangle']]:
            chosen = np.zeros(2)
        else:
            chosen = None
        support = paths.impact(tf, ball, contact, node, omega, chosen, controls, clock+elapsed)
        # A later failed continuation does not invalidate an earlier
        # mapped road contact with an actual supported rolling segment.
        if (not obs.mesh.rail[contact['triangle']] and node>=0 and contacts and
                support.parts and support.parts[0]['kind']=='floor' and
                support.parts[0]['ticks']>EPS):
            bearings.setdefault(node,dict(reason='handoff',node=node,pieces=tuple(pieces),way=tuple(way),contacts=tuple(contacts),
                         ticks=elapsed,air_ticks=air_ticks,bounce_count=int(bounces),
                         state=ph.State(contact['position'],contact['incoming'],omega)))
        pieces.extend(support.parts); way.extend(support.way)
        elapsed += sum(s['ticks'] for s in support.parts)
        if support.reason == 'handoff':
            return finish(reason='handoff' if support.node >= 0 else 'no_landing_node', node=support.node, pieces=pieces, way=way, contacts=contacts,
                        ticks=elapsed, air_ticks=air_ticks, bounce_count=int(bounces), state=support.body)
        if support.reason not in ('chain_end','floor_end','side_release','convex','rebound') or count == sc.MAX_CARRIES:
            return finish(reason=support.reason if count < sc.MAX_CARRIES else 'carry_budget', pieces=pieces,
                        contacts=contacts, ticks=elapsed)
        start = support.body
        obj = paths.name(contact['triangle'])
        if support.reason == 'rebound':
            bounces += 1
        if not support.parts:
            pieces.append(sc.segment('release',obj,ph.State(contact['position'],contact['incoming']),start,0.,
                                     np.zeros(2),support.reason))
        below = min(obs.bottom-ball.radius-FLOOR, start.pos[1]-ball.radius)
        if controls is None:
            push = sc.keys_along(start.vel) if support.push is None else support.push
            t = cf.landing_step(ball,start,below)
            if aim is not None:
                from .rail_inverse import objective
                desired=objective(tf,ball,start,aim)
                if desired is not None and np.max(np.abs(desired[0]))<=1.+EPS:
                    push=np.clip(desired[0],-1.,1.)
            schedule = ((float(t),push),) if t else ()
        else:
            schedule = controls(clock+elapsed)
        if not schedule or schedule[-1][0] <= EPS:
            return finish(reason='no_air_window', pieces=pieces, contacts=contacts, ticks=elapsed)
        clearance = obs.rail_clearance(ball,start,schedule[0][1],schedule[0][0],support.departure)
        if clearance < 0:
            # An actual collision, not a fictitious supported prefix, decides
            # the outcome of a sharp upward bend / return to the same rail.
            clearance = 0.
        tr = trace_air(obs,ball,start,schedule,rim=support.rim,departure=support.departure,
                       clearance=clearance, incoming_only=True)
        why, face, dt, hit = tr.result
        bounces += tr.bounces
        bstates = {i['tick']:i for i in tr.contacts}
        part_start = 0.
        for a, push, n in tr.parts:
            event = bstates.get(part_start+n)
            b = cf.flight(ball,a,int(n),push=push)
            b.pos = event['position'] if event else _points(a,n,ball.lin_damp*ph.DT,cf.air_impulse(ball,push=push))
            if event:
                b.vel = event['incoming']
            pieces.append(sc.segment('flight',event['object'] if event else '',a,b,n,push,event['kind'] if event else 'key_change' if part_start+n < dt-EPS else why, geometry=(air_geometry(ball,a,push,n),)))
            _, path = _samples(a, cf.air_impulse(ball,push=push),ball.lin_damp*ph.DT,n,1.)
            way.extend(path)
            part_start += n
        contacts.extend(dict(i,tick=i['tick']+elapsed) for i in tr.contacts)
        elapsed += dt
        air_ticks += dt
        if why != 'land' or dt <= clearance+EPS:
            return finish(reason=why if why != 'land' else 'never_airborne', pieces=pieces, contacts=contacts,ticks=elapsed)
        node = paths.landing.near(face,hit)
        contact = tr.contacts[-1]
        last, push, n = tr.parts[-1]
        omega = cf.flight(ball,last,math.ceil(n),push=push).omega
    raise AssertionError('bounded carry search')


def add_continuations(g, mesh, info, *, obs=None, landing=None):
    from . import supportcarry as sc
    t0 = perf_counter()
    obs = obs if obs is not None else Obstacles(mesh,info.get('death_boxes',[]))
    if landing is None:
        contacts = [_support(mesh,p,ph.WOOD.radius) or tuple(c for c in mesh.contacts(p,ph.WOOD.radius,RAIL_CONTACT_QUERY)
                    if mesh.rail[c.triangle] and c.normal[1]>.05) for p in g.pos]
        landing = LandingPlaces(mesh,g.pos,contacts)
    paths = sc.SurfacePaths(g,obs,landing)
    stats = Counter()
    proposals = defaultdict(list)
    launches = g.info['launches']
    balls = {ball.name:ball for ball in ph.BALLS.values()}
    work={(u,v,id(e)):(u,v,e) for (u,v),entry in launches.items() for e in entry.get('variants',(entry,))}
    for u,v,e in g.info.pop('_floor_seeds',()):
        work[u,v,id(e)]=u,v,e
    for u,v,e in work.values():
        if not e.get('contacts') or e.get('segments'):
            continue
        ball = balls[e['ball']]
        last = e['contacts'][-1]
        if last['kind'] not in ('land','dual_rail'):
            continue
        for coast in (False,True):
            if coast and mesh.rail[last['triangle']] and not any(not mesh.rail[c['triangle']] and c['kind'] in ('land','dual_rail') for c in outcome['contacts']):
                break  # coasting would make exactly the same pre-floor trajectory
            outcome = carry_contact(paths,ball,last,v,coast=coast)
            stats['carry_' + outcome['reason']] += 1
            outcomes=(*outcome.get('bearings',()),outcome)
            for outcome0 in outcomes:
                if outcome0['reason'] != 'handoff' or not outcome0['pieces']:
                    continue
                w = outcome0['node']
                if w < 0 or w == u:
                    continue
                first = sc.segment('flight',last['object'],ph.State(e['launch_position'],e['launch_velocity']),
                                   ph.State(e['landing_position'],last['incoming']),e['flight_ticks'],e['keys_in_air'],'first_support')
                ticks = e['ticks'] + outcome0['ticks']
                target = outcome0['state'].pos
                edge = dict(e,ticks=ticks,time=ticks/132.,flight_ticks=e['flight_ticks']+outcome0['air_ticks'],
                            landing_position=target.copy(),down=float(e['launch_position'][1]-target[1]),
                            across=float(np.linalg.norm((target-e['launch_position'])[XZ])),
                            bounce_count=e['bounce_count']+outcome0['bounce_count'],
                            contacts=e['contacts']+tuple(dict(c,tick=c['tick']+last['tick']) for c in outcome0['contacts']),
                            segments=(first,*outcome0['pieces']), carry_coast=coast,
                            motion=continued_motion(e,outcome0['pieces'],target,g.pos[w]),
                            way=densify_way(np.vstack([e['way'][:-1],*outcome0['way'],target,g.pos[w]])),
                            representative='support_carry',first_landing_node=v)
                if outcome0['contacts']:
                    edge['landing_face'] = outcome0['contacts'][-1]['triangle']
                # The coast flag can yield an identical chain if it never
                # reaches a floor. Keep exact distinct witnesses, not quotas.
                same = proposals[u,w]
                if not any(a['ball']==edge['ball'] and motion.signature(motion.path_for(a))==motion.signature(edge['motion']) for a in same):
                    same.append(edge)
                    stats['carry_kept'] += 1
    for (u,v), extra in proposals.items():
        original = variants(g,u,v)
        entries = tuple(sorted((*original,*extra), key=lambda e:(e['ticks'],e['ball'])))
        launches[u,v] = dict(entries[0],variants=entries)
        if original:
            g.adj[u] = [(w,cost(entries[0]) if w==v and k=='launch' else c,k) for w,c,k in g.adj[u]]
        else:
            g.adj[u].append((v,cost(entries[0]),'launch'))
    stats['carry_seconds'] = perf_counter()-t0
    stats['carry_pairs'] = len(proposals)
    g.info.setdefault('takeoff_stats',{}).update(stats)
    return stats


def paper_cone_connections(tf, paths, edge, event, u, stats):
    """A measured-direction outer envelope, not a swept flight witness."""
    from . import paperimpact
    targets=getattr(paths.landing,'conformal_targets',None)
    if not targets:return
    cone=paperimpact.for_event(paths.obs.mesh,event,FLOOR)
    if cone is None:return
    _,_,vlo,vhi=cone.ranges(); radius=ph.PAPER.radius+FLOOR
    for w,q,patch,label in paperimpact.road_candidates(tf,cone,targets,radius,stats):
        if w==u:continue
        # An ordering estimate, not a trajectory or an exact time bound.
        fall=cf.landing_step(ph.PAPER,cone.center,min(q[1],cone.position[1]-EPS)) or 0.
        ticks=edge['ticks']+fall
        yield w,patch,label,dict(edge,ticks=ticks,time=ticks/132.,flight_ticks=edge['flight_ticks']+fall,
                                 representative='paper_impact_cone',paper_impact_cone=True,verified=False,
                                 impact_cone=dict(category=cone.category,half_angle_degrees=math.degrees(cone.angle),
                                                  origin=cone.position.tolist(),normal=cone.normal.tolist(),
                                                  velocity_low=vlo.tolist(),velocity_high=vhi.tolist(),
                                                  velocity_center=cone.center.vel.tolist(),target=q.tolist(),
                                                  patch=int(patch),rolling_component=int(label)),
                                 landing_position=q.copy(),way=np.vstack((edge['way'],q)),
                                 landing_face=int(targets.surface.patches[patch].faces[0]))


def add_rail_departures(g, mesh, info, *, obs=None, landing=None, transformer_kind=None):
    """Supplement restored v6 witnesses with target-inverted rail exits."""
    from .railride import RailPaths, support_normal
    from . import supportcarry as sc
    import sys
    tf = sys.modules[__name__]
    t0=perf_counter(); stats=Counter(); records=[]; best={}; hypotheses={}; confirmed=defaultdict(set)
    obs=obs if obs is not None else Obstacles(mesh,info.get('death_boxes',[]))
    if landing is None:
        contacts=[_support(mesh,p,ph.WOOD.radius) or tuple(c for c in mesh.contacts(p,ph.WOOD.radius,RAIL_CONTACT_QUERY)
                  if mesh.rail[c.triangle] and c.normal[1]>.05) for p in g.pos]
        landing=LandingPlaces(mesh,g.pos,contacts)
    paths=RailPaths(g,obs,landing)
    by_name={b.name:b for b in ph.BALLS.values()}
    labels={k:witness_labels(g,info,transformer_kind,k) for k in by_name}
    launches=g.info['launches']
    seeds = g.info.pop('_rail_seeds',None)
    if seeds is None:  # old archived graphs used by diagnostic tools
        seeds = []
        for (u,v), data in launches.items():
            for e in data.get('variants',(data,)):
                if not e.get('segments') and e.get('contacts') and mesh.rail[e['contacts'][-1]['triangle']]:
                    seeds.append((u,v,e,None))
    for u,v,e,c in seeds:
        b=by_name[e['ball']]
        last=e['contacts'][-1]
        if b.name=='paper':
            for w,patch,label,candidate in paper_cone_connections(tf,paths,e,last,u,stats):
                key=(u,e['site'],patch,label)
                if key not in hypotheses or candidate['ticks']<hypotheses[key][2]['ticks']:
                    hypotheses[key]=(u,w,candidate)
        if c is None:
            c=paths.capture(b,last,v)
        if c is None:
            continue
        l=labels[e['ball']]; paths.labels=l
        t=sc.unit(c.body.vel)
        # The additional search represents landing momentum, not a new
        # rollout from a rest node already in the graph.
        if np.linalg.norm(c.body.vel) <= sc.local_speed(sc.law(b,support_normal(t),sc.keys_along(t)).along(t)):
            continue
        stats['rail_capture_seeds']+=1
        for r in paths.search(tf,b,c,stats=stats,confirmed=confirmed[u,b.name]):
            w=r['node']
            if u==w:continue
            dep=r['departures'][-1]
            itinerary=[]
            for name in (last['object'],*(d['object'] for d in r['departures'])):
                if not itinerary or itinerary[-1]!=name:itinerary.append(name)
            record=dict(site=int(e['site']),source=int(u),first_rail=int(v),target=int(w),ball=e['ball'],
                        dest_component=int(l[w]),mode=dep['mode'],rail=dep['object'],target_node=dep['target_node'],
                        rail_itinerary=itinerary,joint=dep.get('joint'),
                        rail_lines=tuple(dict.fromkeys((paths.line_for(c.lane),*(a['rail_line'] for a in r['departures'])))),
                        launch=e['launch_position'].tolist(), depart=dep['position'].tolist(),
                        depart_velocity=dep['velocity'].tolist(), landing=r['state'].pos.tolist(),
                        push=dep['push'].tolist(),interval=list(dep['interval']),
                        ticks=float(e['ticks']+r['ticks']), seconds=float((e['ticks']+r['ticks'])/132.))
            records.append(record)
            surface=g.info.get('conformal')
            patch=int(surface.face_patch[r['face']]) if surface is not None else -1
            record['target_patch']=patch
            # A verified relay is a distinct route even if a direct flight
            # reaches the same rolling component faster.
            key=(e['site'],u,e['ball'],int(l[w]),patch,tuple(itinerary),record['rail_lines'] if any('crown' in x for x in record['rail_lines']) else ())
            if key not in best or record['ticks']<best[key][2]['ticks']:
                best[key]=(u,w,record,e,r)
    pairs=defaultdict(list)
    for u,w,record,e,r in best.values():
        last=e['contacts'][-1]; target=r['state'].pos
        first=sc.segment('flight',last['object'],ph.State(e['launch_position'],e['launch_velocity']),
                         ph.State(e['landing_position'],last['incoming']),e['flight_ticks'],e['keys_in_air'],'first_support')
        flight=e['flight_ticks']+sum(s['ticks'] for s in r['segments'] if s['kind'] in ('flight','rail_gap'))
        init=np.asarray(e['way']); k=len(init)-1-int(np.argmin(np.linalg.norm(init[::-1]-e['landing_position'],axis=1)))
        edge=dict(e,ticks=record['ticks'],time=record['seconds'],flight_ticks=flight,landing_position=target.copy(),
                  landing_face=int(r['face']),down=float(e['launch_position'][1]-target[1]),
                  across=float(np.linalg.norm((target-e['launch_position'])[XZ])),
                  bounce_count=e['bounce_count']+r['bounces'],contacts=e['contacts']+tuple(dict(h,tick=h['tick']+last['tick']) for h in r['contacts']),
                  segments=(first,*r['segments']),motion=continued_motion(e,r['segments'],target,g.pos[w]),way=densify_way(np.vstack([init[:k+1],*r['way'],target,g.pos[w]])),
                  representative='rail_inverse',first_landing_node=record['first_rail'],rail_departures=r['departures'],
                  rail_itinerary=tuple(record['rail_itinerary']),rail_lines=record['rail_lines'])
        pairs[u,w].append(edge)
    for u,w,e in hypotheses.values():pairs[u,w].append(e)
    for (u,v), extra in pairs.items():
        old=variants(g,u,v)
        es=tuple(sorted((*old,*extra),key=lambda e:(not e.get('verified',True),e['ticks'],e['ball'])))
        launches[u,v]=dict(es[0],variants=es)
        if old:
            g.adj[u]=[(w,cost(es[0]) if w==v and k=='launch' else p,k) for w,p,k in g.adj[u]]
        else:
            g.adj[u].append((v,cost(es[0]),'launch'))
    g.info['rail_departure_candidates']=records
    stats.update(paths.counts)
    stats['rail_inverse_seconds']=perf_counter()-t0
    stats['rail_mapped_candidates']=len(records)
    stats['rail_retained']=len(best)
    stats['paper_impact_cone_provisional']=len(hypotheses)
    g.info.setdefault('takeoff_stats',{}).update(stats)
    return stats
