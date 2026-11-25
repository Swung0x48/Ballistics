"""Closed-form stationary contact impulse; fits exclude both SR03 long records."""
from math import sqrt

import numpy as np
from numba import njit

from .physics import State

DEFAULT_ELASTICITY = .3  # ASSUMED missing entry, matches catalog floor/rail
DEFAULT_FRICTION = .7    # ASSUMED missing entry, matches catalog floor/rail


def inertia_ratio(ball):
    # MEASURED: balls/README, sections 2 and 6. Shape proxy for paper.
    return .277 if ball.name == 'paper' else .4


def coefficients(ball, surface, *, rail, support, cap_sample=False):
    paper = ball.name == 'paper'
    # MEASURED fits by shape/surface, leave-one-record-out, documented in physics.md.
    scale, cutoff = (.9, 0.) if paper else (1., .5)
    e = scale * sqrt(ball.elasticity * float(surface.get('elasticity', DEFAULT_ELASTICITY)))
    # Walls are too sparse to establish a tangential law. Rail side contacts
    # retain their family; only non-rail non-support contacts use this fallback.
    factor = (.25 if rail else .2 if support else 0.) if paper else (.1 if rail or support else 0.)
    if cap_sample:
        factor = 1.
    mu = factor * ball.friction * float(surface.get('friction', DEFAULT_FRICTION))
    return e, cutoff, mu


@njit(cache=True)
def _impact(v, omega, normal, r, k, e, cutoff, mu):
    vn = np.dot(v, normal)
    if vn >= 0.:
        return v.copy(), omega.copy()
    normal_change = -(1. + (e if -vn >= cutoff else 0.))*vn
    out = v + normal_change*normal
    if mu <= 0.:
        return out, omega.copy()
    # r_contact = -r*N, so surface slip = v_t - r*(omega cross N).
    slip = v - vn*normal - r*np.cross(omega,normal)
    speed = np.sqrt(np.dot(slip,slip))
    if speed <= 1.e-10:
        return out, omega.copy()
    tangential_change = -min(speed * k/(1.+k), mu*normal_change) / speed * slip
    return out+tangential_change, omega+np.cross(-normal,tangential_change)/(k*r)


def resolve(ball, state, normal, surface, *, rail=False, support=True, cap_sample=False):
    """Independent sphere impulse [DERIVED], with fitted damping of friction cap.

    ``normal`` is the oriented unit world contact normal (or dual-rail resultant).
    ``cap_sample`` evaluates the catalog Coulomb envelope, not a fitted prediction.
    """
    e, cutoff, mu = coefficients(ball,surface,rail=rail,support=support,cap_sample=cap_sample)
    v, w = _impact(state.vel,state.omega,normal,ball.radius,inertia_ratio(ball),e,cutoff,mu)
    return State(state.pos,v,w)
