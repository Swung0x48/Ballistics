"""Enclosures for uncertain paper-impact direction; never collision witnesses."""
from dataclasses import dataclass
import math

import numpy as np

from . import closedform as cf, physics as ph, rail_inverse as inverse

# Observed maxima of resolvable, outward changes in the 24 record audit.
# Coverage of those samples, not a physical hard cap or a fitted trajectory.
PAPER_ANGLES = dict(face=85.755, edge_or_vertex=75.270, rail_tube=84.015, rail_end=51.580)
XZ=[0,2]


def projection(n, angle, vector):
    """Exact extrema of vector . d for unit d in the circular cap about n."""
    size=float(np.linalg.norm(vector))
    if not size:return 0.,0.
    a=float(np.clip(vector@n/size,-1.,1.)); b=math.sqrt(max(0.,1.-a*a))
    c,s=math.cos(angle),math.sin(angle)
    return (size*(-1. if -a>=c else a*c-b*s),size*(1. if a>=c else a*c+b*s))


@dataclass(frozen=True)
class Cone:
    position: np.ndarray
    incoming: np.ndarray
    normal: np.ndarray
    elasticity: float
    category: str

    @property
    def angle(self):return math.radians(PAPER_ANGLES[self.category])

    @property
    def center(self):
        return cf.collision_impulse(ph.PAPER,ph.State(self.position,self.incoming),self.normal,self.elasticity)[0]

    def ranges(self):
        """Conservative: impulse magnitude and direction are correlated."""
        j0,j1=projection(self.normal,self.angle,-self.incoming)
        factor=1.+cf.collision_restitution(ph.PAPER,self.elasticity)
        lo,hi=np.array([projection(self.normal,self.angle,d) for d in np.eye(3)]).T
        a,b=inverse.product(*np.maximum(0.,factor*np.array([j0,j1])),lo,hi)
        return self.position,self.position,self.incoming+a,self.incoming+b

    def at_tick(self,t,impulse):
        """At fixed elapsed PSI and fixed push, rather than a grid of angles."""
        _,_,lo,hi=self.ranges(); loss=ph.PAPER.lin_damp*ph.DT
        a,_=cf._motion(t,lo,impulse,loss); b,_=cf._motion(t,hi,impulse,loss)
        mid,v=cf._motion(t,self.center.vel,impulse,loss)
        return self.position+a,self.position+b,self.position+mid,v


def category(mesh,face,point,normal):
    if not mesh.rail[face]:
        return 'face' if abs(float(mesh.N[face]@normal))>1.-1.e-7 else 'edge_or_vertex'
    axes=mesh.tubes
    if len(axes):
        a,b=axes[:,:3],axes[:,3:6];d=b-a; d2=np.einsum('ij,ij->i',d,d)
        if np.all(d2<=0.):return 'rail_tube'
        t=np.einsum('ij,ij->i',point-a,d)/np.maximum(d2,1.e-20)
        r=np.linalg.norm(point-a-np.clip(t,0.,1.)[:,None]*d,axis=1)
        score=np.where(d2>0.,abs(r-axes[:,6]),np.inf)
        # Same end-ring/mitre tolerance as the measurement. Use a
        # conformal axis of nonzero length; not a position rule for one map.
        j=int(np.argmin(score)); axis=d[j]/math.sqrt(d2[j])
        for end in (a[j],b[j]):
            near=(np.linalg.norm(a-end,axis=1)<.02)|(np.linalg.norm(b-end,axis=1)<.02)
            if near.sum()==1 and abs((point-end)@axis)<=axes[j,6]+.02:return 'rail_end'
    return 'rail_tube'


def for_event(mesh,event,floor):
    if event.get('kind')=='dual_rail' or len(event.get('manifold',()))>1:return None
    i=event['triangle']; normal=event['normal']; incoming=event['incoming']
    if np.linalg.norm(normal)<.99 or incoming@normal>=0.:return None
    e=float(mesh.entities[mesh.owner[i]].get('elasticity',.3))
    point=event['point']; mode=category(mesh,i,point,normal)
    return Cone(point+(ph.PAPER.radius+floor)*normal,incoming,normal,e,mode)


def road_candidates(tf,cone,targets,radius,stats):
    """Continuous outer band against patch-region polygons (no direction probes)."""
    ranges=cone.ranges(); bottom=targets.lowest
    if bottom is None:return
    def at_height(y,top):
        pl,pu,vl,vu=ranges
        # DESCENDING crossing, including the apex-height case. The drift
        # coefficient is increasing in vy and decreasing in target height.
        h0,h1=pl[1]-top,pu[1]-y
        a,b=tf._target_coeff(ph.PAPER,ph.State(np.array([0.,h1,0.]),np.array([0.,vu[1],0.])),0.)
        if not np.isfinite(a) or not np.isfinite(b):return None
        a0=0.
        if h0>=0.:
            a0,_=tf._target_coeff(ph.PAPER,ph.State(np.array([0.,h0,0.]),np.array([0.,vl[1],0.])),0.)
            if not np.isfinite(a0):a0=0.
        low,high=inverse.product(float(a0),float(a),vl[XZ],vu[XZ]); reach=float(b)*ph.PAPER.push
        return pl[XZ]+low-reach,pu[XZ]+high+reach
    window=at_height(bottom,bottom)
    if window is None:return
    # Include initial positions: at a higher target, less drift/reach can
    # put the footprint behind the low-height band.
    lo=np.minimum(window[0],cone.position[XZ]);hi=np.maximum(window[1],cone.position[XZ])
    normals=np.vstack((np.eye(2),-np.eye(2)))
    def cuts(p):
        h,H=p.heights(radius); r=at_height(h,H)
        return () if r is None else ((normals,np.r_[r[1],-r[0]]),)
    yield from targets.representatives('paper',radius,lo,hi,cuts,lambda q:True,
                                      stats=stats,prefix='paper_impact_cone',anchor=cone.position[XZ])


