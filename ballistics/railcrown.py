"""Analytic cross-section of a ball on a single crown of a paired rail."""
from dataclasses import dataclass
import math
import numpy as np

EPS=1.e-8  # floating-point root tolerance, not a capture band


def roots(a,b,c,lo,hi):
    """Roots of a*cos(x)+b*sin(x)+c on a finite angle interval."""
    r=math.hypot(a,b)
    if r<=EPS or abs(c)>r+EPS:return []
    phi=math.atan2(b,a); d=math.acos(float(np.clip(-c/r,-1.,1.)))
    return sorted({float(np.clip(x,lo,hi)) for branch in (phi-d,phi+d)
                   for j in range(math.ceil((lo-branch-EPS)/math.tau),math.floor((hi-branch+EPS)/math.tau)+1)
                   if lo-EPS<=(x:=branch+j*math.tau)<=hi+EPS})


@dataclass(frozen=True)
class Sweep:
    start: float
    end: float
    speed: float
    end_speed: float
    energy: tuple
    event: str
    barrier: float


def swing(theta,v,g,f,r,k,inner):
    """First events on a cylindrical support, angle positive away from twin.

    No-slip, independent axial/cross-sectional motion and no dissipation are
    ASSUMED. This solves energy, normal load and the inner contact boundary,
    not a positional tolerance around the crown.
    """
    B=2*r/(1+k); C=-B*g; D=B*f
    e=v*v-C*math.cos(theta)-D*math.sin(theta)
    v2=lambda x:e+C*math.cos(x)+D*math.sin(x)
    load=lambda x:g*math.cos(x)-f*math.sin(x)-v2(x)/r
    saddle=math.atan2(-f,g)
    barrier=max(0.,(1+k)/2*(v*v-v2(saddle)))
    d=1 if v>EPS else -1 if v<-EPS else int(np.sign(g*math.sin(theta)+f*math.cos(theta)))
    if abs(v)<=EPS and abs(g*math.sin(theta)+f*math.cos(theta))<=EPS:d=0
    if load(theta)<=EPS:
        return (Sweep(theta,theta,v,v,(v*v,C,D),'pressure',barrier),)
    if d==0:return ()
    out=[]; x=theta; u=v
    for _ in range(2):  # one turn at most in the upper cross-section
        edge=math.pi/2 if d>0 else inner
        if (edge-x)*d<=EPS:
            return (*out,Sweep(x,x,u,u,(u*u,C,D),'open' if d>0 else 'twin',barrier))
        lo,hi=sorted((x,edge))
        events=[(edge,'open' if d>0 else 'twin')]
        events.extend((q,'pressure') for q in roots(g-C/r,-f-D/r,-e/r,lo,hi) if (q-x)*d>EPS)
        events.extend((q,'turn') for q in roots(C,D,e,lo,hi) if (q-x)*d>EPS)
        y,event=min(events,key=lambda item:(item[0]-x)*d)
        if event=='turn' and abs(y-saddle)<=EPS:event='asymptote'
        ans=d*math.sqrt(max(0.,v2(y))) if event not in ('turn','asymptote') else 0.
        out.append(Sweep(x,y,u,ans,(max(0.,v2(x)),C,D),event,barrier))
        if event!='turn':break
        x=y;u=0.;d=-d
    return tuple(out)


def miter_root(piece, vertex, incoming, outgoing):
    """First forward intersection with the equal-radius inner seam.

    With q measured from the common vertex, equal distances from the
    two infinite lines mean (q.t0)^2=(q.t1)^2. The inner branch has
    q.t0 <= 0 <= q.t1, hence q.(t0+t1)=0. Restrict to finite pieces at
    the caller. Only admit a root if forward monotonicity is proved.
    """
    normal=incoming+outgoing
    if normal@incoming<=EPS:return None
    def value(x):return float((piece.point(x)-vertex)@normal)
    a,b=value(0.),value(piece.end)
    if a>EPS or b<-EPS:return None
    if piece.kind=='arc':
        from .physics import DT
        x,turn,axis,c,A,B,roll=piece.arc
        if not roll or axis@normal<=EPS:return None
        speed,beta,loss=roll
        ticks=piece.duration(piece.end)/DT
        # The derivative of the interpolated axial displacement is an
        # affine exponential (hence monotone). Bound the angular speed by
        # the extrema of its energy sinusoid over this finite sweep.
        derivative=lambda t:beta/loss-(speed-beta/loss)*math.log1p(-loss)/loss*math.exp(math.log1p(-loss)*t)
        lo,hi=sorted((x,x+turn*piece.end))
        e=c-A*math.cos(x)-B*math.sin(x)
        extrema=[lo,hi,*roots(B,-A,0.,lo,hi)]
        v2=max(0.,*(e+A*math.cos(t)+B*math.sin(t) for t in extrema))
        projections=math.hypot(piece.v@normal,piece.b@normal)/np.linalg.norm(piece.v)
        if min(derivative(0.),derivative(ticks))*(axis@normal)<=math.sqrt(v2)*projections+EPS:return None
    elif piece.kind=='rolling':
        from .motion import tangent
        if min(tangent(piece,t)@normal for t in (0.,piece.end))<=EPS:return None
    elif piece.kind=='line':
        if piece.v@normal<=EPS:return None
    else:return None
    from scipy.optimize import brentq
    z=0. if a>=-EPS else piece.end if b<=EPS else brentq(value,0.,piece.end)
    q=piece.point(z)-vertex
    return z if q@incoming<=EPS and q@outgoing>=-EPS else None


def exposed_load(radial, velocity, force, radius):
    """Instantaneous load per unit mass on the exposed vertex sphere.

    This is not the cylindrical-barrel load. Positive load at this one
    instant does not certify the continuation of a cap walk.
    """
    return -float(force@radial)-float(velocity@velocity)/radius
