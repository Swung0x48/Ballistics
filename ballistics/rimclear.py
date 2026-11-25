"""Feature-level clearance for a finite rolling rim (not a route polyline).

DERIVED: for each triangle Voronoi feature, its orthogonal projection and
validity constraints are affine in the centre. With axial drift we enclose
its actual displacement by the two end displacements. Minimizing over that
interval gives a rigorous lower bound (exact for a circular rim). Each
active axial coordinate is affine in sin/cos, so the remaining stationary
equation has degree at most four in tan(theta/2).
"""
import math

import numpy as np

from . import conformal

EPS = 1.e-8  # numerical tolerance, not the mesh export tolerance


def trig_zeros(c, a, b, lo, hi):
    h = math.hypot(a, b)
    if h <= EPS or abs(c) > h + EPS:
        return ()
    p = math.atan2(b, a)
    q = math.acos(np.clip(-c/h, -1., 1.))
    return tuple(t + k*math.tau for t in (p-q,p+q)
                 for k in range(math.ceil((lo-t)/math.tau), math.floor((hi-t)/math.tau)+1))


def stationary(d, lo, hi):
    c,a,b = d
    # d/dtheta |c+a*cos(theta)+b*sin(theta)|^2 / 2
    u,v,w,z = c@b, -c@a, a@b, (b@b-a@a)/2
    poly = [u+w, 2*v+4*z, -6*w, 2*v-4*z, -u+w]
    zeros = [math.pi]  # chart boundary, including any root at infinity
    for x in np.polynomial.polynomial.polyroots(poly):
        if abs(x.imag) <= EPS*(1+abs(x.real)):
            zeros.append(2*math.atan(x.real))
    return tuple(t+k*math.tau for t in zeros if math.isfinite(t)
                 for k in range(math.ceil((lo-t)/math.tau), math.floor((hi-t)/math.tau)+1))


def features(tri):
    """(kind, vertices, orthogonal projector, Voronoi halfspaces, strict)."""
    I = np.eye(3)
    a,b,c = tri
    n = np.cross(b-a,c-a)
    if np.linalg.norm(n) <= EPS:
        return
    n /= np.linalg.norm(n)
    sides = []
    for i in range(3):
        j = (i+1)%3
        edge = tri[j]-tri[i]; unit = edge/np.linalg.norm(edge)
        out = np.cross(unit,n)
        sides.append((out,out@tri[i]))
        H = np.array([unit,-unit,-out]); Y = np.array([unit@tri[j],-unit@tri[i],-out@tri[i]])
        yield 'edge',(i,j),np.outer(unit,unit),H,Y,2
    H,Y = map(np.array,zip(*sides))
    yield 'face',(0,1,2),I-np.outer(n,n),H,Y,3
    for i in range(3):
        H = np.array([tri[j]-tri[i] for j in range(3) if j!=i]); H /= np.linalg.norm(H,axis=1)[:,None]
        yield 'vertex',(i,),np.zeros((3,3)),H,H@tri[i],0


def feature_minimum(curve, tri, feature):
    kind, ids, proj, H, Y, strict = feature
    if curve.kind == 'line':
        # Rim.motion() uses a constant line for immediate release.
        c,s,d,axis,lo,hi,z0,z1 = curve.p,np.zeros(3),np.zeros(3),np.zeros(3),0.,0.,0.,0.
    else:
        lo,turn,axis,*_ = curve.arc
        hi = lo+turn
        c,s,d = curve.p,curve.v,curve.b
        z0,z1 = sorted((0.,curve.axial(curve.end)))
    C = np.stack((c,s,d)); A = H@axis
    constraints = np.stack((H@c-Y,H@s,H@d),axis=1)
    P = np.eye(3)-proj
    D = (C-np.stack((tri[ids[0]],np.zeros(3),np.zeros(3))))@P
    axial_residual = P@axis
    candidates = [np.array([z,0.,0.]) for z in {z0,z1}]
    if z1>z0:
        candidates.extend(-k/a for k,a in zip(constraints,A) if abs(a)>EPS)
        if axial_residual@axial_residual>EPS**2:
            candidates.append(-(D@axial_residual)/(axial_residual@axial_residual))
    best = None
    for k in candidates:
        limits = np.vstack((constraints+A[:,None]*k, k-[z1,0.,0.], [z0,0.,0.]-k))
        objective = D+k[:,None]*axial_residual
        cuts = sorted({lo,hi,*(t for row in limits for t in trig_zeros(*row,lo,hi))})
        roots = stationary(objective,lo,hi)
        def basis(t):
            return np.array([1.,math.cos(t),math.sin(t)])
        def valid(t):
            return np.all(limits@basis(t)<=EPS)
        def consider(t):
            nonlocal best
            w = basis(t); z = float(k@w)
            centre = w@C+axis*z
            pt = tri[ids[0]]+proj@(centre-tri[ids[0]])
            distance = float(np.linalg.norm(centre-pt))
            if best is None or distance<best['distance']:
                u = (t-lo)/(hi-lo) if hi>lo else 0.
                actual = curve.point(u)
                inside = H@actual-Y
                # A point on a Voronoi boundary belongs to its lower-dimensional
                # feature, not to the supporting face/edge interior.
                bnd = bool(np.any((limits@w)[:strict]>=-EPS))
                attained = (np.all(inside<=EPS) and np.all(inside[:strict]<-EPS))
                best = dict(distance=distance, point=pt, center=centre, parameter=u,
                            actual_distance=float(np.linalg.norm(P@(actual-tri[ids[0]]))) if attained else None,
                            boundary_limit=bnd)
        for a,b in zip(cuts,cuts[1:]):
            mid = (a+b)*.5
            # These are cells delimited by exact trigonometric roots, not probes.
            if valid(mid) and np.all((limits@basis(mid))[:strict]<-EPS):
                for t in (a,b,*(x for x in roots if a<x<b)):
                    consider(t)
        for t in cuts:
            if valid(t) and np.all((limits@basis(t))[:strict]<-EPS):
                consider(t)
    return best


def support_edge(vertices, support):
    if support is None:
        return False
    a,b = map(np.asarray,support)
    d = b-a; length = np.linalg.norm(d)
    if length<=EPS:
        return False
    d /= length
    q = vertices-a; s = q@d
    # READ: raw mesh span identity, not the conformal height fit tolerance.
    return (np.all(np.linalg.norm(q-s[:,None]*d,axis=1)<=EPS)
            and min(s)>=-EPS and max(s)<=length+EPS)


def check(tf, ball, curve, obs, support=None, *, details=False):
    """Return clear, touching, penetrating or unproved_axial_envelope."""
    from . import motion, supportcarry as sc

    r = ball.radius+sc.SKIN
    if support is not None and curve.kind == 'arc':
        a,b = map(np.asarray,support)
        d = b-a; length = np.linalg.norm(d)
        if length<=EPS:
            return 'outside_support', []
        q = (np.array([curve.p,curve.p+curve.arc[2]*curve.axial(curve.end)])-a)@(d/length)
        if min(q)<-EPS or max(q)>length+EPS:
            return 'outside_support', []
    if any(motion.box_hit(curve,lo,hi) is not None for lo,hi in obs.boxes):
        return 'box', []
    lower,upper = curve.bounds(0.,curve.end)
    pad = r+conformal.PLANE_ERROR
    fs = np.asarray(tf._box_triangles((lower-pad)[None],(upper+pad)[None],obs.lo,obs.hi,obs.mesh._grid),dtype=int)
    if not len(fs):
        return 'clear', []
    normals = obs.mesh.N[fs]; lo,hi = curve.bounds(0.,curve.end,normals)
    off = np.einsum('ij,ij->i',obs.mesh.A[fs],normals)
    rows = []; status = 'clear'
    order = ['clear','touching','unproved_axial_envelope','penetrating']
    for face in np.unique(fs[np.maximum(lo-off,off-hi)<=pad]):
        tri = obs.tris[face]
        for feature in features(tri):
            kind, ids, *_ = feature; vertices = tri[list(ids)]
            carried = kind=='edge' and support_edge(vertices,support)
            q = feature_minimum(curve,tri,feature) if details or not carried else None
            if q is None:
                continue
            distance = q['distance']; actual = q['actual_distance']
            # A limit at the boundary of an incident face can occur exactly at
            # the supported launch pose. It is not itself another contact.
            at_support = kind=='face' and q['boundary_limit'] and support_edge(q['point'][None],support)
            if carried:
                result = 'support'
            elif actual is not None and actual<r-conformal.PLANE_ERROR:
                result = 'penetrating'
            elif distance<r-conformal.PLANE_ERROR:
                result = 'unproved_axial_envelope'
            elif distance<=pad and not at_support:
                result = 'touching'
            else:
                result = 'clear'
            if not carried and order.index(result)>order.index(status):
                status = result
            if details or result not in ('clear','support'):
                rows.append(dict(q,feature=kind,vertices=vertices,face=int(face),result=result,
                                 object=obs.mesh.entities[obs.mesh.owner[face]]['name']))
    return status, rows
