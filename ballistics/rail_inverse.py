"""Continuous rail-to-target envelope inversion (no grid of release points)."""
from dataclasses import dataclass
import math

import numpy as np

from . import physics as ph

# Numerical interval root tolerance, in arc length relative to the fitted mesh.
# Does not control a uniform spatial sample or the number of air checks.
ROOT_SPACE = 1.e-4
EPS = 1.e-7
XZ = [0, 2]


def product(lo, hi, a, b):
    x = np.array([lo*a, lo*b, hi*a, hi*b])
    return x.min(axis=0),x.max(axis=0)


@dataclass(frozen=True)
class ReleaseLaw:
    """p=p0+linear*t+pe*expm1(logd*t), v=v0+ve*expm1(logd*t)."""
    p0: np.ndarray
    v0: np.ndarray
    linear: np.ndarray
    pe: np.ndarray
    ve: np.ndarray
    logd: float

    def at(self, t):
        e=math.expm1(self.logd*t)
        return ph.State(self.p0+self.linear*t+self.pe*e,self.v0+self.ve*e)

    def turning_points(self, lo, hi):
        for k in range(3):
            q=self.logd*self.pe[k]
            if q!=0.:
                d=-self.linear[k]/q
                if d>0.:
                    t=math.log(d)/self.logd
                    if lo<t<hi:yield t

    def ranges(self, lo, hi):
        a,b=self.at(lo),self.at(hi)
        p=np.array([a.pos,b.pos,*(self.at(t).pos for t in self.turning_points(lo,hi))]);
        return p.min(0),p.max(0),np.minimum(a.vel,b.vel),np.maximum(a.vel,b.vel)


def contact_time(tf, ball, state, target):
    if state.pos[1] < target[1]-EPS:return None
    a,b=tf._target_coeff(ball,state,target[1]); b=float(b)
    if not math.isfinite(b) or b<=EPS:return None
    t=float(tf._time_at_b(b,ball.lin_damp*ph.DT))
    if not math.isfinite(t) or t<=EPS:return None
    return float(a),b,t


def objective(tf,ball,state,target):
    abt=contact_time(tf,ball,state,target)
    if abt is None:return None
    a,b,t=abt
    push=(target[XZ]-state.pos[XZ]-a*state.vel[XZ])/(b*ball.push)
    return push,t


def bound(tf,ball,ranges,y):
    pl,pu,vl,vu=ranges
    hl,hu=pl[1]-y,pu[1]-y
    if hu < -EPS:return None
    # The descending root (and A and B) increases with starting height
    # and initial vertical velocity. Use the two opposite corners of that
    # box, even when height and velocity are linked on the actual rail.
    a0,b0=tf._target_coeff(ball,ph.State(np.array([0.,max(0.,hl),0.]),np.array([0.,vl[1],0.])),0.)
    a1,b1=tf._target_coeff(ball,ph.State(np.array([0.,max(0.,hu),0.]),np.array([0.,vu[1],0.])),0.)
    d0,d1=product(float(a0),float(a1),vl[XZ],vu[XZ]); lo,hi=pl[XZ]+d0,pu[XZ]+d1
    return lo,hi,float(b0)*ball.push,float(b1)*ball.push,hl


def invert(tf, ball, law, lo, hi, q, *, initial=None):
    """Enclose sign changes with interval arithmetic; keep only certified intervals.

    Position components are exponential-linear, so their only extrema are
    analytic. The air time corners above are monotone bounds. No trajectory
    is evaluated at the subdivisions; ambiguous root brackets are refined
    to a numerical arc-length tolerance. The unresolved strips are not
    counted as feasible (near-tangencies can therefore be missed).
    """
    if hi<lo:return []
    work=[(lo,hi,initial)]; found=[]
    while work:
        u,v,b=work.pop()
        bounds=law.ranges(u,v)
        if b is None:b=bound(tf,ball,bounds,q[1])
        if b is None:continue
        low,high,small,large,h=b
        x=q[XZ]
        if np.any(x<low-large-EPS) or np.any(x>high+large+EPS):continue
        if h>=-EPS and np.all(x>=high-small-EPS) and np.all(x<=low+small+EPS):
            found.append((u,v));continue
        if v-u<=EPS or np.linalg.norm(bounds[1]-bounds[0])<=ROOT_SPACE:
            # A zero-width endpoint may be a genuine fixed geometric exit.
            if hi==lo:
                obj=objective(tf,ball,law.at(lo),q)
                if obj is not None and np.max(abs(obj[0]))<=1+EPS:found.append((lo,lo))
            continue
        c=(u+v)*.5
        work.append((c,v,None)); work.append((u,c,None))
    out=[]
    for x,y in found: # ordered by processing the left bracket first
        if out and x<=out[-1][1]+EPS:out[-1]=(out[-1][0],y)
        else:out.append((x,y))
    return out


def candidates(tf,ball,law,lo,hi,bands):
    box=law.ranges(lo,hi)
    for h0,h1,ids,tree in bands:
        b=bound(tf,ball,box,h0)
        if b is None:continue
        low,high,_,large,_=b
        a=(low+high)*.5; reach=float(np.max((high-low)*.5))+large+EPS
        for j in tree.query_ball_point(a,reach,p=np.inf):
            yield int(ids[j])


def reach_planes(tf, ball, law, lo, hi, height, height_high=None, *, allow_rise=False):
    """Hull of world key squares of the release band at a fixed height.

    With a fixed A, each of the four square-corner laws is affine in
    t and exp(logd*t). Straight-track release has parallel coefficients;
    their correlated extrema form a segment, not an axis-aligned box.
    If vertical state varies, hull the two bounds for A as an outer band.
    """
    from shapely.geometry import MultiPoint
    pl,pu,vl,vu=law.ranges(lo,hi)
    if height_high is None:height_high=height
    if pu[1]<height-EPS and not allow_rise:return None
    def drift(y,vy,h):
        a,_=tf._target_coeff(ball,ph.State(np.array([0.,y-h,0.]),np.array([0.,vy,0.])),0.)
        return float(a)
    a0=drift(pl[1],vl[1],height_high) if pl[1]>=height_high or allow_rise else 0.
    a1=drift(pu[1],vu[1],height)
    if not math.isfinite(a0):a0=0.
    if not math.isfinite(a1):return None
    ratio=ball.push/tf.GRAVITY_IMPULSE
    points=[]
    for A in {a0,a1}:
        if not math.isfinite(A):continue
        base=law.p0+A*law.v0; wanted=law.pe+A*law.ve
        for sx in (-1.,1.):
            for sz in (-1.,1.):
                side=ratio*np.array([sx,sz])
                q=base[XZ]+side*(base[1]-height)
                v=law.linear[XZ]+side*law.linear[1]
                e=wanted[XZ]+side*wanted[1]
                d=np.linalg.norm(v)
                if d>EPS and abs(v[0]*e[1]-v[1]*e[0])<=EPS*d:
                    arg=-float(v@v)/(law.logd*float(v@e)) if v@e else -1.
                    u=math.log(arg)/law.logd if arg>0 else lo
                    for t in (lo,hi,np.clip(u,lo,hi)):
                        points.append(q+v*t+e*math.expm1(law.logd*t))
                else:
                    box=[q+v*t+e*math.expm1(law.logd*t) for t in (lo,hi)]
                    for x in range(2):
                        arg=-v[x]/(law.logd*e[x]) if e[x] else -1.
                        t=math.log(arg)/law.logd if arg>0 else lo
                        if lo<t<hi:box.append(q+v*t+e*math.expm1(law.logd*t))
                    l,h=np.min(box,axis=0),np.max(box,axis=0)
                    points.extend((i,j) for i in (l[0],h[0]) for j in (l[1],h[1]))
    if not points:return None
    hull=MultiPoint(points).convex_hull.buffer(16*EPS,quad_segs=1,join_style=2)
    vertices=np.asarray(hull.exterior.coords)[:-1]
    edges=np.roll(vertices,-1,axis=0)-vertices
    normals=np.c_[-edges[:,1],edges[:,0]] # GEOS buffer exterior is clockwise
    sign=1. if np.sum(normals*(vertices-vertices.mean(0)))>=0. else -1.
    normals*=sign
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    heights=np.sum(normals*vertices,axis=1)
    x,z,X,Z=hull.bounds
    return np.array([x,z]),np.array([X,Z]),(normals,heights)
