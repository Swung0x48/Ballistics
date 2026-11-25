"""Small rigid-body contact manifold solver for the closed-form flight events."""
from itertools import combinations

import numpy as np


def minimum_correction(normals, rhs):
    """Minimize |delta|^2 subject to N @ delta >= rhs, or return None.

    KKT: delta is a positive linear combination of active normals. In 3-D
    at most three independent active normals are needed. Parallel mesh
    triangles do not give that reaction extra weight.
    """
    n = np.asarray(normals)
    rhs = np.asarray(rhs)
    tol = 64 * np.finfo(float).eps * max(1., float(np.max(abs(rhs), initial=0.)))
    if np.all(rhs <= tol):
        return np.zeros(3)
    best, cost = None, np.inf
    for k in range(1, min(len(n),3)+1):
        for active in combinations(range(len(n)),k):
            a = n[list(active)]
            b = rhs[list(active)]
            try:
                lam = np.linalg.solve(a@a.T,b)
            except np.linalg.LinAlgError:
                continue
            if np.min(lam)<-tol:
                continue
            delta=lam@a
            size=float(delta@delta)
            if size<cost and np.all(n@delta >= rhs-tol):
                best,cost=delta,size
    return best


def contact_set(obs, center, radius, i, point, normal, void_owner=-1, void_faces=None):
    """Co-active features at the *same centre*, not nearby future impacts."""
    eps=1.e-7  # same world-space envelope tolerance as swept feature tests
    rows=[(int(i),point,normal,float((center-point)@normal))]
    for c in obs.mesh.contacts(center,radius,eps):
        if c.triangle==i or (c.entity==void_owner and obs.mesh.rail[c.triangle] and
                             (void_faces is None or c.triangle in void_faces)):
            continue
        if c.distance>radius+eps or any(np.linalg.norm(c.normal-n)<=eps and abs(c.distance-g)<=eps for _,_,n,g in rows):
            continue
        rows.append((c.triangle,c.point,c.normal,c.distance))
    return rows


def opposed_support(a,b):
    """Upward reaction in the plane normal to the local two-contact tangent.

    Both coefficients must be compressive. On a sloping rail the normals can
    share a downhill projection; testing antiparallel world-XZ directions
    confuses this shared component with a lack of transverse enclosure.
    """
    c=float(a@b)
    if c>=0. or 1.-abs(c)<1.e-10:
        return None
    x=(a[1]-c*b[1])/(1.-c*c)
    y=(b[1]-c*a[1])/(1.-c*c)
    if min(x,y)<=0.:
        return None
    u=x*a+y*b
    return u/np.linalg.norm(u)
