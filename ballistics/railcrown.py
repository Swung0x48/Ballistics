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
