"""Swept sphere against mesh features using bounded PSI chord root isolation.

The eight-bit keyboard and damping determine the positions at integer PSI.
Between two knots the engine moves along a straight chord. Bisection here is
only over bounds on a whole time interval: never advance the body to the next
PSI or look for a collision by point sampling.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from .geometry import _closest
from .physics import DT

EPS = 1.e-7


@njit(cache=True)
def chord(t, loss, logd):
    k = math.floor(t)
    f = t - k
    power = math.exp(logd * k)
    s = -math.expm1(logd * k) / loss
    return DT * (s + f * power), DT * ((k - s) / loss + f * s)


@njit(cache=True)
def curve(p, v, impulse, t, loss, logd):
    a, b = chord(t, loss, logd)
    return p + a * v + b * impulse


@njit(cache=True)
def turn(v, impulse, loss, logd):
    if v * impulse < 0:
        u = impulse / loss
        return math.ceil(math.log(u / (u - v)) / logd)
    return 0


@njit(cache=True)
def linear_clip(a, b, lo, hi):
    """Restrict a + b*f >= 0, using f in [lo, hi]."""
    if b > 0.:
        lo = max(lo, -a / b)
    elif b < 0.:
        hi = min(hi, -a / b)
    elif a < 0.:
        return 1., 0.
    return lo, hi


@njit(cache=True)
def sphere_span(w, d, r, lo, hi):
    """Clip by |w + f*d| <= r. Also used for projected cylinder roots."""
    speed2 = np.dot(d, d)
    if speed2 == 0.:
        if np.dot(w, w) <= r*r:
            return lo, hi
        return 1., 0.
    closest = -np.dot(w, d) / speed2
    perp = w + closest*d
    half2 = (r*r - np.dot(perp, perp)) / speed2
    if half2 < 0.:
        return 1., 0.
    half = math.sqrt(half2)
    return max(lo, closest - half), min(hi, closest + half)


@njit(cache=True)
def feature_span(tri, normal, i, p, d):
    """Nearest-feature halfspaces, then radial displacement w + f*dw.

    0..2: vertices; 3..5: edges leaving the corresponding vertex; 6: face.
    On a feature cell both the closest point and radial vector are affine.
    """
    j = i % 3
    a = tri[j]
    edge = tri[(j+1) % 3] - a
    w = p - a
    lo, hi = 0., 1.
    if i < 3:
        previous = tri[(j+2) % 3] - a
        lo, hi = linear_clip(-np.dot(w, edge), -np.dot(d, edge), lo, hi)
        lo, hi = linear_clip(-np.dot(w, previous), -np.dot(d, previous), lo, hi)
        return lo, hi, w, d
    if i < 6:
        e2 = np.dot(edge, edge)
        if e2 <= EPS*EPS:
            return 1., 0., w, d
        proj, dp = np.dot(w, edge), np.dot(d, edge)
        lo, hi = linear_clip(proj, dp, lo, hi)
        lo, hi = linear_clip(e2-proj, -dp, lo, hi)
        outward = np.cross(edge, normal)
        lo, hi = linear_clip(np.dot(w, outward), np.dot(d, outward), lo, hi)
        return lo, hi, w - (proj/e2)*edge, d - (dp/e2)*edge
    for k in range(3):
        e = tri[(k+1)%3] - tri[k]
        inside = np.cross(normal, e)
        lo, hi = linear_clip(np.dot(p-tri[k], inside), np.dot(d, inside), lo, hi)
    return lo, hi, np.dot(w, normal)*normal, np.dot(d, normal)*normal


@njit(cache=True)
def _chord_edge(tri, normal, p, q, r, origin, pivot, axis, incoming_only, side_only):
    d = q - p
    earliest = math.inf
    for j in range(6):
        lo, hi, w, dw = feature_span(tri, normal, j, p, d)
        if hi < lo:
            continue
        lo, hi = sphere_span(w, dw, r, lo, hi)
        if hi < lo or lo >= earliest:
            continue
        # The departure edge itself is waived only inside the local sphere;
        # the relief must not suppress a later approach to that same feature.
        if len(pivot):
            anchor = tri[j % 3] - pivot
            delta = anchor - np.dot(anchor, axis)*axis
            on_pivot = np.dot(delta, delta) < EPS*EPS
            if on_pivot and j >= 3:
                e = tri[(j+1)%3] - tri[j%3]
                delta = e - np.dot(e, axis)*axis
                on_pivot = np.dot(delta, delta) < EPS*EPS
            if on_pivot:
                low, high = sphere_span(p-origin, d, r, 0., 1.)
                if low <= lo <= high:
                    lo = np.nextafter(high, math.inf)
        if hi < lo:
            continue
        h = w + lo*dw
        if incoming_only and np.dot(h, dw) >= 0.:
            continue
        if side_only and h[1] >= .6*math.sqrt(np.dot(h, h)):
            # Solve the support cone boundary along the chord, rather than
            # discard a following side approach in this same feature cell.
            lo = cone_exit(w, dw, lo, hi, .6)
            if hi < lo:
                continue
            if incoming_only and np.dot(w+lo*dw, dw) >= 0.:
                continue
        earliest = lo
    return earliest


@njit(cache=True)
def cone_exit(w, d, lo, hi, ny):
    """End of a connected upward normal cone on a straight segment."""
    if hi < lo:
        return math.inf
    x = w + hi*d
    if x[1] >= ny*math.sqrt(np.dot(x,x)):
        return math.inf  # upward cone is convex
    a, b = lo, hi
    while b-a > 1.e-10:
        m = (a+b)*.5
        x = w+m*d
        if x[1] >= ny*math.sqrt(np.dot(x,x)):
            a = m
        else:
            b = m
    return b


@njit(cache=True)
def bounds(p, v, impulse, lo, hi, loss, logd):
    a, b = curve(p,v,impulse,lo,loss,logd), curve(p,v,impulse,hi,loss,logd)
    lower, upper = np.minimum(a,b), np.maximum(a,b)
    for i in range(3):
        k = turn(v[i],impulse[i],loss,logd)
        if lo < k < hi:
            ca, cb = chord(k,loss,logd)
            x = p[i]+ca*v[i]+cb*impulse[i]
            lower[i], upper[i] = min(lower[i],x), max(upper[i],x)
    return lower, upper


@njit(cache=True)
def closest(tri, p):
    a,b,c = tri
    x,y,z,_ = _closest(p[0],p[1],p[2],a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2])
    return np.array([x,y,z])


@njit(cache=True)
def edge_time(tri, normal, tlo, thi, p, v, impulse, loss, logd, r, start, end,
              pivot, axis, incoming_only=False, side_only=False, exempt_radius=0.):
    """First real edge/vertex contact; bracket subcurves, solve chord quadratics."""
    if end < start:
        return -1.
    work = [(start,end)]
    while work:
        a,b = work.pop()
        lo,hi = bounds(p,v,impulse,a,b,loss,logd)
        gap = np.maximum(0., np.maximum(lo-thi,tlo-hi))
        if np.dot(gap,gap) > r*r + EPS:
            continue
        center = (lo+hi)*.5
        extent = (hi-lo)*.5
        if exempt_radius > 0.:
            outer = np.maximum(np.abs(lo-p),np.abs(hi-p))
            if np.dot(outer,outer) < exempt_radius*exempt_radius:
                continue
        delta = center-closest(tri,center)
        # The distance to a convex set is 1-Lipschitz. The nearest point
        # supplies an additional separating plane for the entire box.
        dist = math.sqrt(np.dot(delta,delta))
        if dist*dist - np.dot(np.abs(delta),extent) > r*dist + EPS:
            continue
        l,h = math.floor(a),math.ceil(b)
        if h-l > 1:
            m = float((l+h)//2)
            work.append((m,b))
            work.append((a,m))
            continue
        x,y = curve(p,v,impulse,a,loss,logd),curve(p,v,impulse,b,loss,logd)
        f = _chord_edge(tri,normal,x,y,r,p,pivot,axis,incoming_only,side_only)
        if f <= 1.:
            return a + (b-a)*f
    return -1.


@njit(cache=True)
def sphere_time(center, r, p, v, impulse, loss, end):
    """First hit for a stationary trigger sphere; no triangle required."""
    logd = math.log1p(-loss)
    work = [(0.,end)]
    while work:
        a,b = work.pop()
        lo,hi = bounds(p,v,impulse,a,b,loss,logd)
        gap = np.maximum(0.,np.maximum(lo-center,center-hi))
        if np.dot(gap,gap) > r*r:
            continue
        l,h = math.floor(a),math.ceil(b)
        if h-l > 1:
            m = float((l+h)//2)
            work.append((m,b)); work.append((a,m))
            continue
        x,y = curve(p,v,impulse,a,loss,logd),curve(p,v,impulse,b,loss,logd)
        f,g = sphere_span(x-center,y-x,r,0.,1.)
        if g > f:
            return a + (b-a)*f
    return -1.



@njit(cache=True)
def rail_span(p,v,impulse,a,b,loss,logd,r,tris,faces,center,width):
    """Certify an entire interval of single- or two-reaction rail support."""
    reactions = []
    errors = []
    on_chord = math.ceil(b)-math.floor(a) <= 1
    if on_chord:
        x,y = curve(p,v,impulse,a,loss,logd),curve(p,v,impulse,b,loss,logd)
    else:
        x,y = center,center
    for i in faces:
        tri = tris[i]
        w = center-closest(tri,center)
        size = np.linalg.norm(w)
        if size-width > r+EPS or w[1]+width <= EPS:
            continue
        if size+width <= r+EPS and w[1]-width > EPS:
            if w[1]-width >= .6*(size+width):
                return True
            reactions.append(w)
            errors.append(width)
        if not on_chord:
            continue
        n = np.cross(tri[1]-tri[0],tri[2]-tri[0])
        nn = np.linalg.norm(n)
        if nn <= EPS:
            continue
        n /= nn
        d = y-x
        for j in range(7):
            lo,hi,wx,dx = feature_span(tri,n,j,x,d)
            if lo > 0. or hi < 1.:
                continue
            wy = wx+dx
            rx,ry = np.linalg.norm(wx),np.linalg.norm(wy)
            if max(rx,ry) > r+EPS or min(wx[1],wy[1]) <= EPS:
                continue
            # Balls and the upward cone are convex on this feature cell.
            if wx[1] >= .6*rx and wy[1] >= .6*ry:
                return True
            reactions.append((wx+wy)*.5)
            errors.append(np.linalg.norm(dx)*.5)
            break
    for i in range(len(reactions)):
        w = reactions[i]
        a0 = math.hypot(w[0],w[2])
        ea = errors[i]
        if a0 <= ea:
            continue
        for j in range(i):
            z = reactions[j]
            b0 = math.hypot(z[0],z[2])
            eb = errors[j]
            if b0 <= eb:
                continue
            # Maximal angular deviation of a horizontal vector in its disk.
            angle = math.acos(min(1.,max(-1.,(w[0]*z[0]+w[2]*z[2])/(a0*b0))))
            if angle-math.asin(ea/a0)-math.asin(eb/b0) > math.acos(-.9):
                return True
    return False


@njit(cache=True)
def _poly(coeff, degree, x):
    val = coeff[degree]
    for j in range(degree-1,-1,-1):
        val = val*x+coeff[j]
    return val


@njit(cache=True)
def _roots(coeff, degree, a, b):
    """All real sign-changing roots in (a,b), using derivative partitions."""
    while degree > 0 and coeff[degree] == 0.:
        degree -= 1
    # Build derivative coefficients down to the linear polynomial, then
    # walk back up: saved roots of the derivative are the monotonic cuts.
    # An iterative implementation also avoids Numba's recursive JIT linkage
    # when loading callers from a cache in a fresh process on Windows.
    polys = np.zeros((degree+1,degree+1))
    for j in range(degree+1):
        polys[degree,j] = coeff[j]
    for k in range(degree-1,0,-1):
        for j in range(k+1):
            polys[k,j] = (j+1)*polys[k+1,j+1]
    roots = np.empty(0,dtype=np.float64)
    for k in range(1,degree+1):
        q=polys[k]
        out=[]
        if k==1:
            z=-q[0]/q[1]
            if a<z<b:
                out.append(z)
        else:
            x=a
            left=_poly(q,k,x)
            for i in range(len(roots)+1):
                y=roots[i] if i<len(roots) else b
                right=_poly(q,k,y)
                if left*right<0.:
                    l,r=x,y
                    while r-l>2.e-12:
                        m=(l+r)*.5
                        if left*_poly(q,k,m)>0.:
                            l=m
                        else:
                            r=m
                    out.append((l+r)*.5)
                if i<len(roots) and right==0.:
                    out.append(y)
                x,left=y,right
        roots=np.asarray(out)
    return roots


@njit(cache=True)
def _nonnegative(coeff, degree, a, b):
    roots = _roots(coeff,degree,a,b)
    spans = []
    x = a
    for i in range(len(roots)+1):
        y = roots[i] if i < len(roots) else b
        # Root-free intervals have constant sign; this evaluation chooses
        # which side of an algebraic boundary satisfies the inequality.
        if _poly(coeff,degree,(x+y)*.5) >= 0.:
            spans.append((x,y))
        x = y
    return spans


@njit(cache=True)
def _dot_xz(a,b):
    return a[0]*b[0]+a[2]*b[2]


@njit(cache=True)
def chord_rail_end(a,b,r,tris,faces):
    """First gap of the upward-support union on one straight chord.

    Closest triangle features are affine on their Voronoi cells. Range and
    single-normal cone limits are quadratic; the opposing horizontal plane
    normals give one quartic and one sign test. Classify only root intervals.
    """
    d=b-a
    parts=[]
    possible=[]
    for i in faces:
        tri=tris[i]
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        length=np.linalg.norm(n)
        if length<=EPS:
            continue
        n /= length
        for j in range(7):
            lo,hi,w,dw=feature_span(tri,n,j,a,d)
            if hi<=lo:continue
            lo,hi=sphere_span(w,dw,r+EPS,lo,hi)
            lo,hi=linear_clip(w[1]-EPS,dw[1],lo,hi)
            if hi<=lo:continue
            # y^2 - cos(theta)^2 |radial|^2 >= 0 with y > 0.
            cos2=.6**2
            q=np.array([w[1]**2-cos2*np.dot(w,w),2*(w[1]*dw[1]-cos2*np.dot(w,dw)),dw[1]**2-cos2*np.dot(dw,dw)])
            if q[0] >= 0. and lo == 0. and hi == 1. and _poly(q,2,1.) >= 0.:
                return 1.  # convex cone on fixed nearest-feature cell
            parts.extend(_nonnegative(q,2,lo,hi))
            possible.append((lo,hi,w,dw,i))
    for j in range(len(possible)):
        a0,a1,w,dw,i=possible[j]
        for k in range(j):
            b0,b1,z,dz,m=possible[k]
            lo,hi=max(a0,b0),min(a1,b1)
            if i==m or hi<=lo:continue
            w2=np.array([_dot_xz(w,w),2*_dot_xz(w,dw),_dot_xz(dw,dw)])
            z2=np.array([_dot_xz(z,z),2*_dot_xz(z,dz),_dot_xz(dz,dz)])
            dot=np.array([-_dot_xz(w,z),-_dot_xz(w,dz)-_dot_xz(z,dw),-_dot_xz(dw,dz)])
            opp=np.zeros(5)
            for x in range(3):
                for y in range(3):
                    opp[x+y]+=dot[x]*dot[y]-.9**2*w2[x]*z2[y]
            for s,t in _nonnegative(dot,2,lo,hi):
                if _poly(dot,2,(s+t)*.5)>0.:
                    parts.extend(_nonnegative(opp,4,s,t))
    parts.sort()
    end=0.
    for lo,hi in parts:
        if lo>end+2.e-12:break
        end=max(end,hi)
    return end
