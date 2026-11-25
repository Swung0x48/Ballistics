from dataclasses import dataclass
from collections import defaultdict
import math

import numpy as np
from shapely import STRtree, LineString

from . import conformal, physics as ph, closedform as cf, motion

XZ=[0,2]
EPS=1.e-7


class Topology:
    """Finite shared XZ boundary spans, with a world-height agreement test."""
    def __init__(self,surface):
        self.surface=surface
        self.edges=[(k,f,a,b) for k,p in enumerate(surface.patches) for f,a,b in p.boundary
                    if np.linalg.norm((b-a)[XZ])>EPS]
        self.lines=[LineString([a[XZ],b[XZ]]) for _,_,a,b in self.edges]
        self.tree=STRtree(self.lines)
        self.welded=defaultdict(dict)
        for k,f,a,b in self.edges:
            self.welded[conformal.edge_key(a,b)].setdefault(k,f)

    def neighbours(self,k,a,b,point):
        """Unwelded endpoints/T seams are fine; overlap of positive length is required."""
        d=(b-a)[XZ]; size=float(np.linalg.norm(d))
        if size<=EPS:return ()
        t=d/size; q=point[XZ]; side=np.array([-t[1],t[0]])
        tol=conformal.PLANE_ERROR
        line=LineString([a[XZ],b[XZ]])
        # Exact shared mesh spans remain adjacent even if the two fitted
        # plane proxies differ by the sum of their export tolerances.
        out={j:f for j,f in self.welded[conformal.edge_key(a,b)].items() if j!=k}
        plane=self.surface.patches[k]
        for i in self.tree.query(line.buffer(tol,quad_segs=1)):
            j,f,x,y=self.edges[i]
            if j==k or (y-x)[XZ]@t>=-EPS:continue
            if max(abs((x-a)[XZ]@side),abs((y-a)[XZ]@side))>tol:continue
            # Compare fitted planes over the overlap, not along an unrelated
            # portion of a longer edge. The contact must share the finite span.
            lo,hi=sorted(((x-a)[XZ]@t,(y-a)[XZ]@t));lo=max(0.,lo);hi=min(size,hi)
            s=(q-a[XZ])@t
            if hi-lo<=EPS or not lo-tol<=s<=hi+tol:continue
            other=self.surface.patches[j]
            if max(abs(plane.height(a[XZ]+z*t)-other.height(a[XZ]+z*t)) for z in (lo,hi))>tol:continue
            out.setdefault(j,f)
        return tuple(sorted(out.items()))

    def ray_exit(self,k,q,d):
        """First outward finite edge on a polygon, including holes."""
        found=[]
        for f,a,b in self.surface.patches[k].boundary:
            edge=(b-a)[XZ]; outward=np.array([-edge[1],edge[0]])
            den=float(outward@d)
            if den<=EPS:continue
            s=float(outward@(a[XZ]-q)/den)
            at=q+max(s,0.)*d
            if s>=-EPS and -EPS<=(at-a[XZ])@edge<=edge@edge+EPS:
                found.append((max(s,0.),f,a,b))
        return min(found,key=lambda x:x[0],default=None)


@dataclass(frozen=True)
class Runway:
    """Contact-plane ray, stored from the lip backwards, not grid chords."""
    points: np.ndarray
    arc: np.ndarray
    patches: tuple
    normals: tuple
    owners: tuple
    lip: tuple
    next_normals: tuple = ()

    @classmethod
    def trace(cls,topo,mesh,k,p,d,reach):
        patch=topo.surface.patches[k]; q=p[XZ]
        row=topo.ray_exit(k,q,d)
        if row is None or row[0]>reach+EPS:return None
        _,f,a,b=row; q=q+row[0]*d
        def on_plane(xz,patch):return np.array([xz[0],patch.height(xz),xz[1]])
        lip=on_plane(q,patch)
        # Connected uphill/concave exits are not free lips. At a convex seam
        # the normal of the next block tilts in the direction of travel.
        following=topo.neighbours(k,a,b,lip)
        if following and all(topo.surface.patches[j].normal[XZ]@d<=patch.normal[XZ]@d+EPS for j,_ in following):return None
        points=[lip]; ids=[]; normals=[]; owners=[]; seen=set()
        while k not in seen:
            seen.add(k); patch=topo.surface.patches[k]; end=topo.ray_exit(k,q,-d)
            if end is None or end[0]<=EPS:break
            length,f0,a0,b0=end; p0=on_plane(q-length*d,patch)
            points.append(p0);ids.append(k);normals.append(patch.normal);owners.append(int(mesh.owner[f0]))
            candidates=topo.neighbours(k,a0,b0,p0)
            if not candidates:break
            # At a branched vertex no unique planar runway is certified.
            if len(candidates)!=1:break
            k=candidates[0][0];q=p0[XZ]
        if not ids:return None
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0)[:,XZ],axis=1))]
        return cls(np.asarray(points),arc,tuple(ids),tuple(normals),tuple(owners),(a,b,f),
                   tuple(topo.surface.patches[j].normal for j,_ in following))

    def allows_release(self,rim):
        if rim is None:return False
        return all(rim.angle<math.acos(float(np.clip(self.normals[0]@n,-1.,1.)))-EPS for n in self.next_normals)

    def roll(self,tf,ball,key,depth,obs=None):
        """Piecewise straight longitudinal rolling, no distance/time sampling.

        A heading-preserving runway is a model constraint. The lateral
        component of the key/gravity law is not advanced here. At an
        intermediate convex seam a release invalidates this candidate.
        """
        from . import supportcarry as sc
        speed=ticks=0.; parts=[]; keys=ph.key_push(key,0)[XZ]
        for i in reversed(range(len(self.patches))):
            span=self.arc[i+1]-self.arc[i]
            covered=min(span,depth-self.arc[i]); n=self.normals[i]
            if covered<=EPS:continue
            r=ball.radius+sc.SKIN
            raw=self.points[i]-self.points[i+1]; d=raw/np.linalg.norm(raw)
            p=self.points[i]-raw*(covered/span)+r*n
            law=sc.law(ball,n,keys).along(d)
            if parts:
                v0=parts[-1]['end_velocity']; previous=self.normals[i+1]
                if v0@n>EPS and tf.convex_release(ball,parts[-1]['end_position'],previous,n,v0,v0,keys,sc.inertia_ratio(ball)) is not None:return None
                speed=max(0.,float(v0@d))
            res=law.over(speed,covered*np.linalg.norm(raw)/span)
            if res is None:return None
            end,seconds=res; t=seconds/ph.DT
            a=ph.State(p,speed*d); b=ph.State(self.points[i]+r*n,end*d)
            curve=motion.rolling(a,law.impulse*d,law.loss,t)
            if obs is not None and not obs.support_clear(a,law.impulse*d,law.loss,t,self.owners[i],ball.radius-sc.RECAPTURE):return None
            parts.append(dict(sc.segment('runway','',a,b,t,keys,'patch_runway',geometry=(curve,)),patch=self.patches[i],normal=n.copy()))
            speed=end;ticks+=t
        return speed,ticks,tuple(parts)

    def profiles(self,tf,ball,key):
        """Intervals bracketed by starts at geometric seams or the lip."""
        out=[(0.,0.,0.)]
        for p in self.arc[1:]:
            hit=self.roll(tf,ball,key,p)
            if hit is None:break
            out.append((float(p),hit[0],hit[1]))
        return out

    def for_speed(self,tf,ball,key,speed,profiles,obs):
        if speed<=EPS:return 0.,0.,()
        from scipy.optimize import brentq
        for (a,s, _),(b,t,_) in zip(profiles,profiles[1:]):
            if not min(s,t)-EPS<=speed<=max(s,t)+EPS:continue
            if abs(t-speed)<=EPS:depth=b
            elif abs(s-speed)<=EPS:depth=a
            else:
                # Numeric root of the composed closed forms. A failed seam
                # event is not a length estimate or a valid speed witness.
                def residual(d):
                    trial=self.roll(tf,ball,key,d)
                    if trial is None:raise ValueError('unwitnessed runway branch')
                    return trial[0]-speed
                try:depth=brentq(residual,a,b,xtol=EPS)
                except ValueError:continue
            hit=self.roll(tf,ball,key,depth,obs)
            if hit is not None:return depth,hit[1],hit[2]
        return None
