"""Finite tube support at a rail departure, and finite downstream sleeves."""
from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

from . import conformal, rail_inverse as inverse, sweep

EPS=1.e-7
XZ=[0,2]


def tube_radial(points, a, b):
    v=b-a
    point=np.asarray(points)-a
    d=float(v@v)
    return point if d<=EPS**2 else point-np.clip(point@v/d,0.,1.)[...,None]*v


def _cap_faces(triangles, fitted, a, b, radius):
    """Close the fitted shell with attached faces in a terminal ring plane."""
    axis=b-a; length=np.linalg.norm(axis)
    if length<=EPS:return fitted
    u=axis/length
    axial=(triangles-a)@u
    # A mitred end need not be perpendicular to the tube axis. For an
    # end face tilted at most 45 degrees, the axial span is at most the
    # ring radius. The shared-edge test below distinguishes the actual cap.
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    width=radius+conformal.PLANE_ERROR
    cap=np.flatnonzero((abs(normals@u)>np.linalg.norm(normals,axis=1)/math.sqrt(2.))&
                       ((np.max(abs(axial),axis=1)<=width)|(np.max(abs(axial-length),axis=1)<=width)))
    if not len(cap):return fitted
    def edges(k):
        t=triangles[k]
        return {conformal.edge_key(t[i],t[(i+1)%3]) for i in range(3)}
    connected=set().union(*(edges(k) for k in np.flatnonzero(fitted)))
    remaining={k:edges(k) for k in cap if not fitted[k]}
    while attached:= [k for k,v in remaining.items() if v&connected]:
        for k in attached:
            fitted[k]=True
            connected.update(remaining.pop(k))
    return fitted


def tube_span(tubes, a, b):
    """Raw fitted cylinders with positive overlap on the departing axis.

    A same-owner pipe, a touching successor, or the other crown does not
    inherit the departing bearing's clearance.
    """
    tubes=np.asarray(tubes).reshape(-1,7)
    t=b-a; length=np.linalg.norm(t)
    if length<=EPS:return tubes[:0]
    t=t/length
    q=tubes[:,:6].reshape(-1,2,3)-a
    s=q@t
    off=q-s[:,:,None]*t
    overlap=np.minimum(s.max(1),length)-np.maximum(s.min(1),0.)
    return tubes[(overlap>EPS)&(np.linalg.norm(off,axis=2).max(1)<=conformal.PLANE_ERROR)]


def bind_axes(tubes, triangles, origin, radius):
    """Bind only initially active finite tubes whose skin fits the source mesh.

    Triangle ownership is known at the caller. The exporter does not store a
    tube owner, so verify that a triangle on that owner fits the tube, not
    just that the bearing centre happens to be near it.
    """
    matches=[]; tracks=[]
    if not len(triangles) or not len(tubes):return (),np.empty(0,dtype=int)
    a,b,rt=tubes[:,:3],tubes[:,3:6],tubes[:,6]
    d=b-a; D=np.einsum('ij,ij->i',d,d)
    w=origin-a
    radial=w-np.clip(np.sum(w*d,axis=1)/np.maximum(D,EPS**2),0,1)[:,None]*d
    near=np.flatnonzero((radial[:,1]>EPS)&(np.linalg.norm(radial,axis=1)<=radius+rt+EPS))
    for i in near:
        # Only used to associate the fitted tube to faces. This is not a
        # second, enlarged collision radius for the support prefix.
        dist=np.linalg.norm(tube_radial(triangles,a[i],b[i]),axis=2)
        tolerance=conformal.PLANE_ERROR+10.**(-conformal.WELD)
        on=abs(np.max(dist,axis=1)-rt[i])<=tolerance
        if np.any(on):
            matches.append(tubes[i]);tracks.extend(np.flatnonzero(_cap_faces(triangles,on,a[i],b[i],rt[i])))
    return tuple(matches),np.unique(tracks).astype(np.int64)


def _cells(axis):
    a,b,r=axis[:3],axis[3:6],axis[6]
    d=b-a; length=np.linalg.norm(d)
    I=np.eye(3)
    if length<=EPS:return ((a,I,None,0.,0.),)
    u=d/length
    # In each closest-feature region radial distance is affine in position.
    return ((a,I,u,-math.inf,0.),(a,I-np.outer(u,u),u,0.,length),(b,I,u,0.,math.inf))


def _covers(piece, start, end, cells):
    for a,Q,u,l,h,r in cells:
        if u is not None:
            mn,mx=piece.bounds(start,end,u[None,:]); offset=float(a@u)
            if mn[0]-offset<l-EPS or mx[0]-offset>h+EPS:continue
        mn,mx=piece.bounds(start,end,Q); origin=Q@a
        lo,hi=mn-origin,mx-origin
        if lo[1]>=EPS and np.linalg.norm(np.maximum(abs(lo),abs(hi)))<=r+EPS:
            return True
    return False


def chord_support_end(p,q,cells):
    """Connected prefix of finite-feature quadratic intervals on one chord."""
    d=q-p; spans=[]
    for a,Q,u,l,h,r in cells:
        lo,hi=0.,1.
        if u is not None:
            v=float((p-a)@u); v1=float(d@u)
            if math.isfinite(l):lo,hi=sweep.linear_clip(v-l,v1,lo,hi)
            if math.isfinite(h):lo,hi=sweep.linear_clip(h-v,-v1,lo,hi)
        x=Q@(p-a); dx=Q@d
        lo,hi=sweep.linear_clip(x[1]-EPS,dx[1],lo,hi)
        if hi<lo:continue
        lo,hi=sweep.sphere_span(x,dx,r,lo,hi)
        if hi>=lo:spans.append((lo,hi))
    until=0.
    for a,b in sorted(spans):
        if a>until+EPS:break
        until=max(until,b)
    return until


def support_end(piece, axes, radius):
    """First end of the connected, upward finite-tube support envelope.

    Interval extrema skip entire pieces of the exponential law. At an
    engine chord, affine feature slabs + quadratic sphere roots give the
    boundary without probing regularly spaced times. Re-entry does not
    extend this prefix.
    """
    cells=[(*c,radius+t[6]) for t in axes for c in _cells(t)]
    work=[(0.,piece.end)]
    while work:
        a,b=work.pop()
        if _covers(piece,a,b,cells):continue
        k,h=math.floor(a),math.ceil(b)
        if h-k<=1:
            f=chord_support_end(piece.point(a),piece.point(b),cells)
            if f<1.-EPS:return a+f*(b-a)
        else:
            cut=float((k+h)//2)
            work.extend(((cut,b),(a,cut)))
    return piece.end


@dataclass(frozen=True)
class Sleeve:
    lane: int
    j: int
    direction: int
    a: np.ndarray
    b: np.ndarray
    half: float
    node: int

    @property
    def polygon(self):
        d=(self.b-self.a)[XZ]
        side=np.array([-d[1],d[0]])*(self.half/np.linalg.norm(d))
        return MultiPoint([a[XZ]+s*side for a in (self.a,self.b) for s in (-1,1)]).convex_hull

    def point(self,xz):
        d=(self.b-self.a)[XZ]
        s=float(np.clip((np.asarray(xz)-self.a[XZ])@d/(d@d),0.,1.))
        return np.array([xz[0],self.a[1]+s*(self.b[1]-self.a[1]),xz[1]])


def invert_sleeve(tf, ball, law, lo, hi, sleeve):
    """Intersect an analytic reach band with the finite rail support strip.

    Representatives are the clipped strip's boundary/centre events. Each
    representative is inverted with the linked exponential release law;
    only the certified intervals for those aims are yielded. A rejected
    representative is not a proof that the entire continuous strip blocks.
    """
    ys=(sleeve.a[1],sleeve.b[1])
    bound=inverse.reach_planes(tf,ball,law,lo,hi,min(ys),max(ys))
    if bound is None:return
    geo=sleeve.polygon
    for overlap in conformal.footprint(geo,[bound[2]]):
        seen=set()
        for p in conformal.polygon_parts(overlap):
            targets=[p.representative_point(),p.centroid,*(nearest_points(Point(x[XZ]),p)[1] for x in (sleeve.a,sleeve.b))]
            if p.geom_type=='Polygon':targets.extend(Point(x) for x in p.exterior.coords[:-1])
            elif p.geom_type=='LineString':targets.extend(Point(x) for x in p.coords)
            for aim in targets:
                if not p.covers(aim):continue
                xz=tuple(aim.coords[0])
                if xz in seen:continue
                seen.add(xz)
                target=sleeve.point(xz)
                for ab in inverse.invert(tf,ball,law,lo,hi,target):
                    yield target,ab
