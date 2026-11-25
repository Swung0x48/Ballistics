"""Target-side end-of-track inversion for one supported straight-piece family."""
import math

import numpy as np
from shapely import LineString
from shapely.affinity import translate

from . import closedform as cf, physics as ph, supportcarry as sc

XZ=[0,2]
EPS=1.e-7


def band(ball,center,tangent,key,patch,speeds):
    """Inverse image of the finite support polygon under held-key flight.

    With A,B the closed-form air coefficients, centre = P+A*v*T+B*b.
    If n.T != 0, the plane equation eliminates A*v, so the footprint
    is affine in B. If n.T == 0, contact time is constant and the
    footprint is affine in v. This is a continuous proposal; the
    original PSI/triangle contact must still verify the representative.
    """
    n=patch.normal; r=ball.radius+sc.SKIN; loss=ball.lin_damp*ph.DT
    gap=float(center@n-patch.offset-r); direction=float(tangent@n)
    push=cf.air_impulse(ball,push=key); weight=float(push@n)
    if gap<=EPS or weight>=-EPS:return
    def time(v):
        speed=-direction*v
        lo=cf._turn(speed,-weight,loss) if speed<0. else 0.
        return cf._increasing_root(speed,-weight,loss,gap,lo)
    if abs(direction)<=EPS:
        tick=time(0.)
        if tick is None or tick<1.:return
        _,a,b=cf._coeff(tick,loss); a*=ph.DT;b*=ph.DT
        base=center+b*push; delta=a*tangent; interval=speeds
        speed_at=lambda x:x
    else:
        times=[time(v) for v in speeds]
        if any(t is None or t<1. for t in times):return
        interval=tuple(sorted(ph.DT*cf._coeff(t,loss)[2] for t in times))
        base=center-gap/direction*tangent; delta=push-weight/direction*tangent
        from scipy.optimize import brentq
        def speed_at(x):
            lo,hi=sorted(times)
            f=lambda t:ph.DT*cf._coeff(t,loss)[2]-x
            tick=lo if abs(f(lo))<=EPS else hi if abs(f(hi))<=EPS else brentq(f,lo,hi)
            return -(gap+x*weight)/(direction*ph.DT*cf._coeff(tick,loss)[1])
    low,high=interval
    if high-low<=EPS:return
    poly=translate(patch.polygon,*(r*n[XZ])); p=base[XZ]; d=delta[XZ]
    if np.linalg.norm(d)<=EPS:
        from shapely import Point
        spans=[(low,high)] if poly.covers(Point(p+d*(low+high)*.5)) else []
    else:
        from .sharedsearch import _lines
        geom=poly.intersection(LineString((p+low*d,p+high*d))); spans=[]
        for seg in _lines(geom):
            if seg.is_empty:continue
            a,b=sorted(float((np.asarray(y)-p)@d/(d@d)) for y in (seg.coords[0],seg.coords[-1]))
            if b-a>EPS:spans.append((max(a,low),min(b,high)))
    for a,b in spans:
        m=(a+b)*.5
        yield tuple(sorted((speed_at(a),speed_at(b)))),speed_at(m),(p+a*d,p+b*d)


def runway_interval(axis,distance,speeds):
    """Map a speed interval to distance before the outlet, from rest."""
    reach=axis.over(0.,distance)
    if reach is None:return None
    low=max(0.,speeds[0]); high=min(reach[0],speeds[1]); terminal=axis.terminal_speed
    if high-low<=EPS or terminal<=EPS:return None
    def back(v):
        ticks=math.log1p(-min(v,np.nextafter(terminal,0.))/terminal)/math.log1p(-axis.loss)
        return min(distance,float(axis.at(0.,ticks)[0]))
    return back(low),back(high),back((low+high)*.5)

def integer_runway(axis,speed_bounds,distance):
    """An integer-tick representative inside the analytic speed interval."""
    top=axis.over(0.,distance); terminal=axis.terminal_speed
    if top is None or terminal<=EPS:return None
    low=max(0.,speed_bounds[0]); high=min(speed_bounds[1],top[0],np.nextafter(terminal,0.))
    if high-low<=EPS:return None
    logd=math.log1p(-axis.loss)
    lo,hi=(math.log1p(-v/terminal)/logd for v in (low,high))
    first,last=max(1,math.ceil(lo+EPS)),math.floor(hi-EPS)
    if last<first:return None
    tick=(first+last)//2
    return float(axis.at(0.,tick)[0]),tick


def midrail_switch(first,second,initial,distance,speeds,clock):
    """Invert final speed for one switch on a finite, straight groove.

    Both runs have the same loss l. D = DT * (U*t - (v-v0)/l),
    U=terminal speed. At fixed total length, d(v_end)/d(switch)
    has the sign of U_first-U_second while both runs move forward.
    Bracket that monotone inverse; intersect it with absolute integer ticks.
    """
    from scipy.optimize import brentq
    if first.loss!=second.loss or min(first.impulse,second.impulse,initial,distance)<=EPS:return None
    def forward(axis):
        u=axis.terminal_speed
        return u-(initial-u)*math.log1p(-axis.loss)/axis.loss>EPS
    if not (forward(first) and forward(second)):return None
    max_time=first.over(initial,distance)[1]/ph.DT
    def outlet(t):
        travelled,v=first.at(initial,t)
        end=second.over(float(v),max(0.,distance-float(travelled)))
        return end[0]
    a,b=outlet(0.),outlet(max_time); u,v=max(min(a,b),speeds[0]),min(max(a,b),speeds[1])
    if v-u<=EPS:return None
    def invert(speed):
        if abs(speed-a)<=EPS:return 0.
        if abs(speed-b)<=EPS:return max_time
        return brentq(lambda t:outlet(t)-speed,0.,max_time)
    lo,hi=sorted((invert(u),invert(v)))
    first_tick,last_tick=math.ceil(clock+lo+EPS),math.floor(clock+hi-EPS)
    if last_tick<first_tick:return None
    tick=(first_tick+last_tick)//2
    span=tuple(float(first.at(initial,t)[0]) for t in (lo,hi))
    return tick,span,(u,v),float(outlet(tick-clock))
