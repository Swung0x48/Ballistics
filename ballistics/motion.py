"""Geometry of a route, independent of its display polyline.

Each piece supplies extrema or bounds on a complete parameter interval.
Straight pieces clip exactly; curves isolate overlapping interval roots.
"""
from __future__ import annotations
from typing import NamedTuple
import math

import numpy as np

from . import closedform as cf, sweep
from .physics import DT

TOL=1.e-7  # geometric root enclosure in world units, not a probe spacing
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X, _GL_W = (_GL_X+1.)*.5, _GL_W*.5


class Piece(NamedTuple):
    kind: str
    p: np.ndarray
    v: np.ndarray
    b: np.ndarray
    loss: float
    end: float
    arc: tuple = ()

    def point(self,t):
        if self.kind=='line':return self.p+t*self.v
        if self.kind=='air':return sweep.curve(self.p,self.v,self.b,t,self.loss,math.log1p(-self.loss))
        if self.kind=='rolling':return self.p+cf.RollingLaw(self.b,self.loss).at(self.v,t)[0]
        start,turn,axis,c,a,b,initial = self.arc
        z=start+turn*t
        return self.p+self.v*math.cos(z)+self.b*math.sin(z)+axis*self.axial(t)

    def duration(self,t):
        start,turn,_,energy,a,b,_ = self.arc
        if t<=0. or turn==0.:return 0.
        q=abs(turn)*t
        diff=(turn*t)*_GL_X**2
        local=start+diff*.5
        v2=energy+2*np.sin(diff*.5)*(-a*np.sin(local)+b*np.cos(local))
        return float((2*q*np.linalg.norm(self.v)*_GL_X/np.sqrt(np.maximum(v2,np.finfo(float).tiny)))@_GL_W)

    def axial(self,t):
        duration=self.duration(t)
        roll=self.arc[-1]
        if not roll:return duration
        speed,beta,loss = roll
        return float(cf.RollingAxis(beta,loss).at(speed,duration/DT)[0])

    def bounds(self,a,b,n=None):
        n = np.eye(3) if n is None else np.asarray(n)
        if self.kind=='arc':
            start,turn,axis,*_ = self.arc
            x,y = sorted((start+turn*a,start+turn*b))
            u,w = n@self.v,n@self.b
            aa,bb = u*math.cos(x)+w*math.sin(x),u*math.cos(y)+w*math.sin(y)
            lo,hi=np.minimum(aa,bb),np.maximum(aa,bb)
            for j,(s,t) in enumerate(zip(u,w)):
                phase=math.atan2(t,s)
                for sign, peak in ((1.,phase),(-1.,phase+math.pi)):
                    if math.ceil((x-peak)/(2*math.pi))<=math.floor((y-peak)/(2*math.pi)):
                        if sign>0:hi[j]=math.hypot(s,t)
                        else:lo[j]=-math.hypot(s,t)
            coef=n@axis
            ya,yb = coef*self.axial(a),coef*self.axial(b)
            low_axis,high_axis=np.minimum(ya,yb),np.maximum(ya,yb)
            if self.arc[-1]:
                speed,beta,loss=self.arc[-1]
                turn_tick=cf._turn(speed,beta,loss)
                if self.duration(a)<turn_tick*DT<self.duration(b):
                    dx=cf.RollingAxis(beta,loss).at(speed,turn_tick)[0]
                    low_axis=np.minimum(low_axis,coef*dx)
                    high_axis=np.maximum(high_axis,coef*dx)
            base=n@self.p
            return base+lo+low_axis,base+hi+high_axis
        pa,pb = n@self.point(a),n@self.point(b)
        lo,hi = np.minimum(pa,pb),np.maximum(pa,pb)
        if self.kind=='line':return lo,hi
        v,d = n@self.v,n@self.b
        for j,(s,t) in enumerate(zip(v,d)):
            if self.kind=='air':
                z=sweep.turn(s,t,self.loss,math.log1p(-self.loss))
            else:
                z=cf._turn(s,t,self.loss)
            if a<z<b:
                q=float(n[j]@self.point(z))
                lo[j],hi[j]=min(lo[j],q),max(hi[j],q)
        return lo,hi


def line(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return Piece('line',a,b-a,np.zeros(3),0.,1.)


def flight(start,impulse,loss,t):
    return Piece('air',start.pos.copy(),start.vel.copy(),impulse.copy(),loss,float(t))


def rolling(start,impulse,loss,t):
    return Piece('rolling',start.pos.copy(),start.vel.copy(),impulse.copy(),loss,float(t))


def arc(p,c,s,start,turn,axis,energy,axial_roll=()):
    return Piece('arc',p,c,s,0.,1.,(start,turn,axis,*energy,axial_roll))


def lines(pts):
    return tuple(line(a,b) for a,b in zip(pts,pts[1:]))


def convex_hit(piece, normals, rhs):
    """First parameter whose position satisfies normals @ p <= rhs."""
    n=np.asarray(normals); rhs=np.asarray(rhs)
    if piece.kind=='line':
        lo,hi=0.,1.
        for a,b in zip(rhs-n@piece.p,-(n@piece.v)):
            lo,hi=sweep.linear_clip(a,b,lo,hi)
            if hi<lo:return None
        return lo
    stack=[(0.,piece.end)]
    while stack:
        a,b=stack.pop()
        lo,hi=piece.bounds(a,b,n)
        if np.any(lo>rhs+TOL):continue
        if np.all(hi<=rhs+TOL):return a
        if b==a or np.max(hi-lo,initial=0.)<=TOL:
            # Conservative only in the final numerical root enclosure.
            return a
        m=(a+b)*.5
        if m in (a,b):return a
        stack.extend(((m,b),(a,m)))
    return None


def box_hit(piece, lo, hi, origin=None, frame=None):
    a=np.eye(3) if frame is None else frame
    q=np.zeros(3) if origin is None else a@origin
    return convex_hit(piece,np.vstack((a,-a)),np.r_[hi+q,-lo-q])


def near_hit(piece, distance, radius, *, a=0., b=None, clip=None):
    """Signed/unsigned 1-Lipschitz distance below radius on a closed curve."""
    todo=[(a,piece.end if b is None else b)]
    while todo:
        u,v=todo.pop(); lo,hi=piece.bounds(u,v)
        if clip is not None:
            n, cut = clip
            lower, upper = piece.bounds(u,v,n[None])
            if upper[0]<=cut:continue
        mid=(lo+hi)*.5; half=np.linalg.norm(hi-lo)*.5
        d=distance(mid)
        if d-half>=radius:continue
        if d+half<radius and (clip is None or lower[0]>cut):return u
        if half<=TOL or u==v:
            q=piece.point((u+v)*.5)
            if distance(q)<radius and (clip is None or q@n>cut):return u
            continue
        m=(u+v)*.5
        if m in (u,v):continue
        todo.extend(((m,v),(u,m)))
    return None


def sphere_hit(piece, center, radius):
    if piece.kind=='line':
        a,b=sweep.sphere_span(piece.p-center,piece.v,radius,0.,1.)
        u=(a+b)*.5
        return b>=a and np.linalg.norm(piece.point(u)-center)<radius
    if piece.kind=='air':
        return sweep.sphere_time(center,radius,piece.p,piece.v,piece.b,piece.loss,piece.end)>=0.
    return near_hit(piece,lambda q:np.linalg.norm(q-center),radius) is not None


def path_for(e):
    """Old synthetic fixtures and old fan paths can be explicit straight paths."""
    return e['motion'] if 'motion' in e else lines(np.asarray(e.get('way',()),dtype=float))


def guard_path_for(edge):
    """Guard both the witness and its coarse attachment, not just one endpoint."""
    chords=edge.get('projection_chords')
    return path_for(edge) if not chords else (chords[0],*path_for(edge),chords[1])


def edge_paths(g,u,v,kind):
    if kind=='launch':
        from .takeoff import variants
        return tuple(guard_path_for(e) for e in variants(g,u,v))
    if kind=='lift':
        e=g.info.get('lifts',{}).get((u,v))
        return (path_for(e),) if e else ()
    return ((line(g.pos[u],g.pos[v]),),)


def line_box_distance(p,d,half,lo=0.,hi=1.):
    """Minimum distance from a clipped segment to a centred box."""
    cuts=[lo,hi]
    for x,y,h in zip(p,d,half):
        if y:
            cuts.extend(t for q in (-h,h) if lo<(t:=(q-x)/y)<hi)
    cuts.sort(); best=math.inf
    for a,b in zip(cuts,cuts[1:]):
        m=p+d*((a+b)*.5)
        above=m>half; below=m<-half; use=above|below
        w=(p-np.where(above,half,-half))[use]; v=d[use]
        vv=v@v
        t=np.clip(-(w@v)/vv,a,b) if vv else a
        best=min(best,float(np.linalg.norm(w+t*v)))
    return best


def tangent(piece,t):
    if piece.kind=='line':return piece.v
    if piece.kind=='air':
        power=math.exp(math.log1p(-piece.loss)*math.floor(t))
        return power*piece.v+(1-power)*piece.b/piece.loss
    if piece.kind=='rolling':
        u=piece.b/piece.loss; l=math.log1p(-piece.loss)
        return u+(piece.v-u)*(-l/piece.loss)*math.exp(l*t)
    start,turn,axis,c,a,b,roll=piece.arc
    z=start+turn*t
    angular=turn*(-piece.v*math.sin(z)+piece.b*math.cos(z))
    if turn==0 or not np.any(axis):return angular
    v2=c+2*math.sin((z-start)*.5)*(-a*math.sin((z+start)*.5)+b*math.cos((z+start)*.5))
    rate=abs(turn)*np.linalg.norm(piece.v)/math.sqrt(max(v2,np.finfo(float).eps))
    if roll:
        speed,beta,loss=roll
        ticks=piece.duration(t)/DT
        u=beta/loss; l=math.log1p(-loss)
        speed=u+(speed-u)*(-l/loss)*math.exp(l*ticks)
        return angular+axis*rate*speed
    return angular+axis*rate


def support_planes(tri,n,radius):
    """Centre prism above a triangle face, bounded by a support radius."""
    e=np.roll(tri,-1,axis=0)-tri
    sides=np.cross(e,n)
    h=np.r_[np.einsum('ij,ij->i',sides,tri), tri[0]@n+radius, -(tri[0]@n)]
    return np.vstack((sides,n,-n)),h


def segment_clip(p,d,n,h):
    a,b=0.,1.
    for w,z in zip(h-n@p,-n@d):
        a,b=sweep.linear_clip(w,z,a,b)
        if b<a:return None
    return a,b


def union_spans(spans,tol=1.e-10):
    out=[]
    for a,b in sorted(spans):
        if not out or a>out[-1][1]+tol:out.append((a,b))
        else:out[-1]=(out[-1][0],max(b,out[-1][1]))
    return out


def vertical_extrusion(n,h,drop):
    """Eliminate q-hat_y*s in a convex prism, 0<=s<=drop."""
    c=-n[:,1]
    rows=[]; rhs=[]
    for i in range(len(n)):
        if abs(c[i])<=1.e-14:
            rows.append(n[i]); rhs.append(h[i]);continue
        if c[i]<0:
            rows.append(n[i]);rhs.append(h[i]-c[i]*drop)
            for j in np.flatnonzero(c>1.e-14):
                # Lower bound (h_i-n_i*q)/c_i <= upper (h_j-n_j*q)/c_j.
                rows.append(n[j]/c[j]-n[i]/c[i]);rhs.append(h[j]/c[j]-h[i]/c[i])
        else:
            rows.append(n[i]);rhs.append(h[i])
    return np.asarray(rows),np.asarray(rhs)


def angular_spans(center,u,v,n,rhs):
    """Solve all convex halfspaces on a circle via their analytic angle roots."""
    a,b,c=n@u,n@v,rhs-n@center
    cuts=[0.,2*math.pi]
    for x,y,h in zip(a,b,c):
        r=math.hypot(x,y)
        if h < -r:return []
        if h>=r or r==0.:continue
        d=math.atan2(y,x); w=math.acos(np.clip(h/r,-1.,1.))
        cuts.extend(((d-w)%(2*math.pi),(d+w)%(2*math.pi)))
    cuts.sort()
    return [(l,r) for l,r in zip(cuts,cuts[1:]) if r>l and
            np.all(a*math.cos((l+r)*.5)+b*math.sin((l+r)*.5)<=c)]


def signature(path):
    """Exact geometry identity without consulting the display polyline."""
    def value(x):
        if isinstance(x,np.ndarray):return x.tobytes()
        if isinstance(x,tuple):return tuple(value(z) for z in x)
        return x
    return tuple(value(p) for p in path)
