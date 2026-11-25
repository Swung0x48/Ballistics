"""Joint location/velocity reachability on one conformal rolling plane.

Controls use the convex hull of the world keys. Bounds do not certify that
an intermediate state stays on the finite patch; witnesses certify that
separately. No enumeration of physical steps is used here.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from . import closedform as cf, physics as ph, motion, supportcarry as sc

EPS=1.e-8


def sums(n,loss):
    _,s,t=cf._coeff(n,loss)
    return float(ph.DT*t),float(s)


def absolute_sum(n,loss,x,y):
    """sum |x + y*d**j|, j=0..n-1. There is at most one sign change."""
    if not n:return 0.
    c=0
    if y and 0.<-x/y<1.:
        c=min(n,max(0,1+math.floor(math.log(-x/y)/math.log1p(-loss))))
    total=x*n+y*sums(n,loss)[1]
    prefix=x*c+y*sums(c,loss)[1]
    return abs(prefix)+abs(total-prefix)


def extreme(n,loss,want,sign):
    """Max/min terminal input speed, conditional on weighted position.

    From the baseline -1, allocate +2 first to the most recent input, in
    decreasing B_j/A_j order. The partial last allocation is the only
    fractional item. Binary search uses geometric PREFIX sums, never ticks.
    Return (terminal speed coefficient, chronological held-input runs).
    """
    A,B=sums(n,loss)
    if not n:return 0.,()
    if abs(want)>A+EPS:raise ValueError('outside positional support')
    need=np.clip((sign*want+A)*.5,0.,A); lo,hi=1,n
    while lo<hi:
        j=(lo+hi+1)//2
        if sums(j,loss)[0]<=need+EPS:lo=j
        else:hi=j-1
    count=lo; pos,vel=sums(count,loss); level=0.
    if count<n:
        size=sums(count+1,loss)[0]-pos
        level=float(np.clip((need-pos)/size,0.,1.))
    power=math.exp(count*math.log1p(-loss)); speed=sign*(-B+2*(vel+level*power))
    runs=[]
    if count<n:
        if n-count-1:runs.append((n-count-1,-sign))
        runs.append((n-count,sign*(-1.+2*level)))
    runs.append((n,sign))
    return float(speed),tuple(runs)


@dataclass(frozen=True)
class PlaneReach:
    origin: np.ndarray
    velocity: np.ndarray
    gravity: np.ndarray
    axes: np.ndarray
    inverse: np.ndarray
    loss: float
    normal: np.ndarray

    @classmethod
    def make(cls,ball,patch,position,velocity):
        n=np.asarray(patch.normal);r=ball.radius+sc.SKIN
        p=np.asarray(position)+n*(patch.offset+r-n@position)
        v=np.asarray(velocity)-n*(n@velocity)
        no=sc.law(ball,n,np.zeros(2)); I=np.eye(3); P=I-n[:,None]*n[None,:]
        axes=(P[:,[0,2]])*ball.push/(1.+sc.inertia_ratio(ball))
        return cls(p,v,no.impulse,axes,np.linalg.pinv(axes),no.loss,n)

    def center(self,n):
        a,v=cf._motion(n,self.velocity,self.gravity,self.loss)
        return self.origin+a,v

    def support(self,n,position_direction,velocity_direction):
        """Exact support of the joint 4D image at an integer PSI."""
        if int(n)!=n or n<0:raise ValueError('expected nonnegative integer horizon')
        p,v=self.center(n); w=position_direction@self.axes; z=velocity_direction@self.axes
        x=w*ph.DT/self.loss
        return float(position_direction@p+velocity_direction@v+sum(absolute_sum(n,self.loss,a,b) for a,b in zip(x,z-x)))

    def edge(self,a,b,n):
        p,_=self.center(n);A,_=sums(n,self.loss); q=self.inverse@(a-p); d=self.inverse@(b-a)
        lo,hi=0.,1.
        for base,delta in zip(q,d):
            if abs(delta)<=EPS:
                if abs(base)>A+EPS:return None
                continue
            x,y=sorted(((-A-base)/delta,(A-base)/delta));lo=max(lo,x);hi=min(hi,y)
        return (lo,hi) if lo<=hi+EPS else None

    def slice(self,n,position):
        """Exact velocity parallelogram at a fixed attainable point/time."""
        p,v=self.center(n); w=self.inverse@(position-p)
        pairs=[tuple(extreme(n,self.loss,d,sign) for sign in (-1,1)) for d in w]
        lo,hi=np.array([[pair[j][0] for pair in pairs] for j in (0,1)])
        return v,lo,hi,pairs

    def schedule(self,n,position,which=None):
        p,_=self.center(n); w=self.inverse@(position-p); A,_=sums(n,self.loss)
        if which is None:
            return ((n,w/A),) if A>EPS else ((n,np.zeros(2)),)
        _,_,_,pairs=self.slice(n,position)
        runs=[p[j][1] for p,j in zip(pairs,which)]
        stops=sorted({t for seq in runs for t,_ in seq})
        return tuple((t,np.array([next(value for end,value in seq if end>=t) for seq in runs])) for t in stops)

    def towards(self,n,position,heading):
        """Invert an attainable outgoing direction without a direction grid."""
        direction=np.asarray(heading); direction=direction-self.normal*(self.normal@direction)
        if np.linalg.norm(direction)<=EPS:return ()
        direction=direction/np.linalg.norm(direction)
        center,low,high,pairs=self.slice(n,position)
        m=self.inverse@direction; start=self.inverse@center; lo,hi=0.,math.inf
        for a,v,l,h in zip(m,start,low,high):
            if abs(a)<=EPS:
                if not l-EPS<=-v<=h+EPS:return ()
                continue
            s,t=sorted(((l+v)/a,(h+v)/a));lo=max(lo,s);hi=min(hi,t)
        if not lo<=hi or hi<=EPS:return ()
        wanted=hi*m-start
        fraction=np.divide(wanted-low,high-low,out=np.zeros_like(low),where=high-low>EPS)
        fraction=np.clip(fraction,0.,1.)
        stops=sorted({q for row in pairs for _,runs in row for q,_ in runs})
        result=[]
        for q in stops:
            push=[]
            for row,f in zip(pairs,fraction):
                l,h=[next(v for end,v in runs if end>=q) for _,runs in row]
                push.append(l+(h-l)*f)
            result.append((q,np.asarray(push)))
        return tuple(result)

    def time_windows(self,a,b):
        """Closed edge intersection windows for the continuous coefficient law.

        Eliminate edge parameter s via lower<=upper. Each inequality is
        c+linear*t+expcoef*expm1(log(d)*t)<=0. Split at its sole stationary
        point and solve roots on monotone intervals, including infinite tails.
        The set of actual full-step contacts is the integers in these windows.
        """
        q=self.inverse@(a-self.origin); direction=self.inverse@(b-a)
        v=self.inverse@self.velocity; g=self.inverse@self.gravity; loss=self.loss
        lower=[np.zeros(3)]; upper=[np.array([1.,0.,0.])]; inequalities=[]
        for base,m,speed,grav in zip(q,direction,v,g):
            for sign in (-1.,1.):
                # sign*(base + m*s - DT*S*v - DT*T*g) <= DT*T
                c=np.array([sign*base,-ph.DT*(1.+sign*grav)/loss,
                            ph.DT*(sign*speed-(1.+sign*grav)/loss)/loss])
                p=sign*m
                if p>EPS:upper.append(-c/p)
                elif p<-EPS:lower.append(-c/p)
                else:inequalities.append(c)
        inequalities.extend(lo-hi for lo in lower for hi in upper)
        windows=[(1.,math.inf)]
        for c in inequalities:
            good=nonpositive(c,math.log1p(-loss))
            windows=[(max(a,p),min(b,q)) for a,b in windows for p,q in good if max(a,p)<=min(b,q)]
            if not windows:break
        return tuple(windows)


def nonpositive(c,logd):
    a,b,h=map(float,c); value=lambda t:a+b*t+h*math.expm1(logd*t)
    last=(math.copysign(math.inf,b) if b else a-h)
    z=-b/(h*logd) if h else -1.
    turn=math.log(z)/logd if z>0. else -1.
    cuts=[1.]+([turn] if turn>1. else [])+[math.inf];out=[]
    for lo,hi in zip(cuts,cuts[1:]):
        left=value(lo);right=value(hi) if math.isfinite(hi) else last
        if left<=0. and right<=0.:
            out.append((lo,hi));continue
        if left>0. and right>=0.:
            if right==0. and math.isfinite(hi):out.append((hi,hi))
            continue
        if left==0. and right>0.:
            out.append((lo,lo));continue
        end=hi
        if not math.isfinite(end):
            # A monotone branch with only an asymptotic zero has no finite root.
            if right==0. or left*right>=0.:continue
            end=max(2.,2*lo)
            while value(end)*left>0.:end*=2.
        root=brentq(value,lo,end,xtol=EPS)
        out.append((lo,root) if left<=0. else (root,hi))
    return tuple(out)


def trace(ball,reach,schedule,patch,obs,owner):
    """Certify piecewise held-input representatives against the finite patch."""
    from shapely import Point
    start=ph.State(reach.origin,reach.velocity);pieces=[];t=0.
    xz=[0,2];r=ball.radius+sc.SKIN
    poly=patch.polygon.buffer(EPS)
    for end,push in schedule:
        dur=end-t; law=sc.law(ball,reach.normal,push)
        item=motion.rolling(start,law.impulse,law.loss,dur); splits={0.,dur}
        for _,a,b in patch.boundary:
            e=(b-a)[xz]; normal=np.array([-e[1],0.,e[0]])
            delta=float((a+r*reach.normal-start.pos)@normal)
            for x in cf._w_roots(start.vel@normal,law.impulse@normal,law.loss,delta):
                if 0.<x<dur:splits.add(x)
        cuts=sorted(splits)
        # Containment is constant between all crossings of the perimeter's
        # supporting lines. Midpoints classify those analytic cells only.
        if any(not poly.covers(Point((item.point((a+b)*.5)-r*reach.normal)[xz])) for a,b in zip(cuts,cuts[1:])):return None
        if obs is not None and not obs.support_clear(start,law.impulse,law.loss,dur,owner,ball.radius-sc.RECAPTURE):return None
        p,v=law.at(start.vel,dur); final=ph.State(start.pos+p,v,ph.rolling_spin(v,reach.normal,ball.radius))
        pieces.append(sc.segment('runway','',start,final,dur,push,'joint_key_witness',geometry=(item,)))
        start=final;t=end
    return start,tuple(pieces)


def rim_clear(tf,ball,topology,patch_id,rim,obs):
    """Check the lip arc and its release centre against other solids.

    A new runway can approach a lip from a new bearing, including one next
    to a wall. An initially overlapped wall is not an airborne edge.
    """
    if obs is None:return True
    path=rim.motion(); r=ball.radius+sc.SKIN
    if any(motion.box_hit(path,lo,hi) is not None for lo,hi in obs.boxes):return False
    lower,upper=path.bounds(0.,1.)
    fs=np.asarray(tf._box_triangles((lower-r)[None],(upper+r)[None],obs.lo,obs.hi,obs.mesh._grid),dtype=int)
    fs=fs[topology.surface.face_patch[fs]!=patch_id]
    if not len(fs):return True
    # Exact rim extrema in each face-normal direction. In particular a
    # supported/tangent face cannot force a huge Lipschitz subdivision if
    # its entire plane is on the nonpenetrating side of the threshold.
    normals=obs.mesh.N[fs]; a,b=path.bounds(0.,1.,normals)
    offsets=np.einsum('ij,ij->i',obs.mesh.A[fs],normals)
    fs=fs[np.maximum(a-offsets,offsets-b)<r-8*EPS]
    if not len(fs):return True
    mask=np.zeros(len(obs.tris),bool);mask[fs]=True; mesh=obs.mesh.subset(mask)
    # This is a proof of clearance, not a search for a contact point.
    # An ambiguous cell within the existing conformal export tolerance
    # is unknown: do not spend many subintervals declaring it clear, or
    # use it to discard a previously witnessed departure.
    from .conformal import PLANE_ERROR
    work=[(0.,path.end)]
    while work:
        t0,t1=work.pop(); lo,hi=path.bounds(t0,t1)
        radius=float(np.linalg.norm(hi-lo))*.5
        d=mesh.distance((lo+hi)*.5,r+1.)[0]
        if d-radius>=r-8*EPS:continue
        if d+radius<r-8*EPS or radius<=PLANE_ERROR:return False
        mid=(t0+t1)*.5
        work.extend(((t0,mid),(mid,t1)))
    return True

def departures(tf,ball,topology,patch_id,position,velocity,obs=None,*,edges=None,allow_transfer=False,guide=None):
    """Yield collision-checked witnesses at analytic edge/time cells.

    The window itself is an unconstrained envelope. At its integer ends we
    invert first the finite line, then choose the positional least aim and
    joint conditional velocity extrema. Only contained paths are witnesses.
    """
    patch=topology.surface.patches[patch_id]; n=patch.normal; r=ball.radius+sc.SKIN
    reach=PlaneReach.make(ball,patch,position,velocity)
    for f,raw_a,raw_b in (patch.boundary if edges is None else edges):
        # Reproject both endpoints to this fitted support plane.
        # Use the block's XZ footprint. Orthogonal projection of a raw
        # vertex moves that footprint horizontally by the fit residual.
        def center(q):
            return np.array([q[0],patch.height(q[[0,2]]),q[2]])+r*n
        a=center(raw_a); b=center(raw_b)
        e=b-a
        if np.linalg.norm(e)<EPS:continue
        outward=sc.unit(np.cross(e,n)); tried=set()
        for low,high in reach.time_windows(a,b):
            left=max(2,math.ceil(low-EPS)); right=math.floor(high+EPS) if math.isfinite(high) else None
            if right is not None and right<left:continue
            for ticks in dict.fromkeys((left,right)):
                if ticks is None or ticks in tried:continue
                tried.add(ticks); span=reach.edge(a,b,ticks)
                if span is None:continue
                l,h=span; pin=np.clip((reach.origin-a)@e/(e@e),l,h)
                for s in dict.fromkeys((float(pin),l,h,(l+h)*.5)):
                    dest=a+np.clip(s,0.,1.)*e
                    schedules=[reach.schedule(ticks,dest,c) for c in (None,(0,0),(1,1),(1,0),(0,1))]
                    if guide is not None:
                        schedule=reach.towards(ticks,dest,guide-dest)
                        if schedule:schedules.insert(0,schedule)
                    seen_schedules=set()
                    for schedule in schedules:
                        tag=tuple((t,tuple(push)) for t,push in schedule)
                        if tag in seen_schedules:continue
                        seen_schedules.add(tag)
                        hit=trace(ball,reach,schedule,patch,obs,int(obs.mesh.owner[f]) if obs is not None else -1)
                        if hit is None:continue
                        state,parts=hit
                        if np.linalg.norm(state.pos-dest)>1.e-5 or state.vel@outward<=EPS:continue
                        push=schedule[-1][1]; foot=state.pos-r*n
                        rim=tf.rim_release(ball,foot,n,outward,state.vel,0,push=push,inertia_ratio=sc.inertia_ratio(ball))
                        others=topology.neighbours(patch_id,raw_a,raw_b,foot)
                        accept_air=rim is not None and not any(
                            state.vel@topology.surface.patches[j].normal<=EPS or rim.angle>=math.acos(float(np.clip(n@topology.surface.patches[j].normal,-1.,1.)))-EPS for j,_ in others)
                        if not (accept_air or allow_transfer and others):continue
                        center,lo,hi,_=reach.slice(ticks,dest)
                        lows=[]; highs=[]
                        for u in span:
                            _,ll,hh,_=reach.slice(ticks,a+np.clip(u,0.,1.)*e); lows.append(ll);highs.append(hh)
                        record=dict(face=f,patch=patch_id,ticks=ticks,schedule=schedule,parts=parts,rim=rim,edge=(a,b),
                                    time_window=(low,high if math.isfinite(high) else None),edge_range=span,
                                    velocity_center=center,velocity_low=lo,velocity_high=hi,velocity_axes=reach.axes,
                                    edge_velocity_low=np.min(lows,axis=0),edge_velocity_high=np.max(highs,axis=0),boundary_state=state,
                                    entry_position=reach.origin,entry_velocity=reach.velocity,normal=n,radius=r)
                        if accept_air:
                            if rim_clear(tf,ball,topology,patch_id,rim,obs):yield record
                        elif allow_transfer:
                            for j,_ in others:
                                other=topology.surface.patches[j]; new=other.normal
                                if state.vel@new>EPS:
                                    # Supported convex transition, using the same
                                    # one-axis conservative rim law as floor carry.
                                    direction=sc.unit(new-n*(new@n))
                                    arc=tf.rim_release(ball,foot,n,direction,state.vel,0,push=push,inertia_ratio=sc.inertia_ratio(ball))
                                    angle=math.acos(float(np.clip(n@new,-1.,1.)))
                                    if arc is None or arc.angle<angle-EPS:continue
                                    curve=motion.arc(foot,r*n,r*direction,0.,angle,arc.along*arc.axis,arc.energy)
                                    seconds=curve.duration(1.)
                                    lo,hi=sorted(float((x-foot)@arc.axis) for x in (raw_a,raw_b))
                                    if not lo-EPS<=seconds*arc.along<=hi+EPS:continue
                                    if obs is not None:
                                        if any(motion.box_hit(curve,l,h) is not None for l,h in obs.boxes):continue
                                        lower,upper=curve.bounds(0.,1.)
                                        fs=tf._box_triangles((lower-r)[None],(upper+r)[None],obs.lo,obs.hi,obs.mesh._grid)
                                        exclude=(topology.surface.face_patch==j)|(topology.surface.face_patch==patch_id)
                                        mask=np.zeros(len(obs.tris),bool); mask[np.asarray(fs,dtype=int)]=True; mask&=~exclude
                                        if mask.any():
                                            solids=obs.mesh.subset(mask)
                                            if motion.near_hit(curve,lambda p:solids.distance(p,r+1.)[0],ball.radius-sc.RECAPTURE) is not None:continue
                                    e0,c0,d0=arc.energy
                                    v=math.sqrt(max(0.,e0+c0*(math.cos(angle)-1.)+d0*math.sin(angle)))
                                    vel=arc.along*arc.axis+v*(direction*math.cos(angle)-n*math.sin(angle))
                                    q=curve.point(1.); q+=new*(other.offset+r-q@new)
                                    entry=ph.State(q,vel,ph.rolling_spin(vel,new,ball.radius)); arc_ticks=seconds/ph.DT
                                    add=sc.segment('rim','',state,entry,arc_ticks,push,'runway_convex_seam',geometry=(curve,))
                                    yield dict(record,next_patch=j,entry=entry,rim=None,parts=(*parts,add),ticks=ticks+arc_ticks)
                                    continue
                                pivot=foot+new*(other.offset-foot@new)
                                vel=state.vel-new*(state.vel@new)
                                yield dict(record,next_patch=j,entry=ph.State(pivot+r*new,vel,ph.rolling_spin(vel,new,ball.radius)),rim=None)


def seam_outer_box(outcome,neighbour):
    """Fixed-time seam outer image in next plane (not an attainable union)."""
    a,b=outcome['edge']; l,h=outcome['edge_range']; n=neighbour.normal; P=np.eye(3)-np.outer(n,n)
    r=outcome['radius']; offset=n*neighbour.offset
    a=P@(a-r*outcome['normal'])+offset+r*n; b=P@(b-r*outcome['normal'])+offset+r*n
    B=P@outcome['velocity_axes']; lo=outcome['edge_velocity_low']; hi=outcome['edge_velocity_high']
    return (np.minimum(a+l*(b-a),a+h*(b-a)),np.maximum(a+l*(b-a),a+h*(b-a)),
            P@outcome['velocity_center']+np.minimum(B*lo,B*hi).sum(1),
            P@outcome['velocity_center']+np.maximum(B*lo,B*hi).sum(1))

def routes(tf,ball,topology,patch_id,position,velocity,obs=None,*,guide=None):
    """Compose checked representatives through nonconvex finite seams.

    The boxes attached to transitions are extrapolation/audit enclosures,
    not permission to restart at arbitrary states inside them. Recurrent
    visits to one patch are not enumerated by this directed witness family.
    """
    import heapq
    from itertools import count
    serial=count(); work=[]; used=set()
    heapq.heappush(work,(0.,next(serial),patch_id,position,velocity,(),frozenset()))
    while work:
        _,_,k,p,v,path,visited=heapq.heappop(work)
        for q in departures(tf,ball,topology,k,p,v,obs,allow_transfer=True,guide=guide):
            j=q.get('next_patch')
            if j is not None:
                if any(p['reason']=='runway_convex_seam' for p in q['parts']):
                    q['successor_outer']=None  # rim transport is nonlinear
                else:
                    q['successor_outer']=seam_outer_box(q,topology.surface.patches[j])
            trail=(*path,q)
            if j is None:
                yield dict(q,parts=tuple(p for x in trail for p in x['parts']),ticks=sum(x['ticks'] for x in trail),
                           blocks=tuple({a:b for a,b in x.items() if a not in ('entry','parts','rim','boundary_state')} for x in trail))
                continue
            if j in visited or j==k:continue
            st=q['entry']; ident=(j,st.pos.tobytes(),st.vel.tobytes(),visited|{k})
            if ident in used:continue
            used.add(ident)
            d=float(np.linalg.norm(st.pos-guide)) if guide is not None else float(len(path))
            heapq.heappush(work,(d,next(serial),j,st.pos,st.vel,trail,visited|{k}))


@dataclass(frozen=True)
class ControlledTrack:
    """A single verified member of a joint runway envelope for takeoff."""
    source: int
    patches: tuple
    rim: object
    parts: tuple
    ticks: float
    audit: dict

    @property
    def points(self):
        return np.asarray([self.parts[-1]['end_position'],*(p['start_position'] for p in reversed(self.parts))])

    @property
    def arc(self):
        return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.points[:,[0,2]],axis=0),axis=1))]


def takeoff_rows(tf,g,obs,rows):
    """Generate key-controlled alternatives to geometric lip proposals."""
    from .roadway import Topology
    from .conformal import edge_key
    topo=Topology(g.info['conformal']); seen=set()
    for site,key,q,d,slope,n,chain,lengths,track in rows:
        if not hasattr(track,'patches'):continue
        a,b,f=track.lip; patch_id=track.patches[0];patch=topo.surface.patches[patch_id]
        for source in chain:
            for ball in ph.BALLS.values():
                token=(site,source,patch_id,edge_key(a,b),ball.name)
                if token in seen:continue
                seen.add(token)
                ground=g.pos[source]-n*(g.pos[source]@n-patch.offset)
                from shapely import Point
                if not patch.polygon.buffer(EPS).covers(Point(ground[[0,2]])):continue
                for case in departures(tf,ball,topo,patch_id,g.pos[source],np.zeros(3),obs,edges=((f,a,b),)):
                    rim=case['rim']; vel=case['boundary_state'].vel; h=np.linalg.norm(vel[[0,2]])
                    if h<=EPS:continue
                    pos=case['boundary_state'].pos; direction=vel[[0,2]]/h
                    if not rim.angle:
                        from dataclasses import replace
                        rim=replace(rim,start=ph.State(rim.start.pos+8*EPS*(rim.normal+rim.outward),rim.start.vel,rim.start.omega))
                    audit={k:v for k,v in case.items() if k not in ('rim','boundary_state','parts')}
                    audit['ball']=ball.name
                    controlled=ControlledTrack(source,(patch_id,),rim,case['parts'],case['ticks'],audit)
                    yield site,key,pos,direction,float(vel[1]/h),n,[source],np.r_[controlled.arc[-1]],controlled
