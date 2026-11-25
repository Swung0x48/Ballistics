"""Supported rail corridors and closed-form departure opportunities."""
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from . import closedform as cf, physics as ph, supportcarry as sc, rail_inverse as inverse, motion, rails, railjoint, railcrown
from .coarse import WALK_NY, _lane_chains

UP, XZ, EPS = sc.UP, sc.XZ, sc.EPS
MAX_CAPTURES = 4    # ASSUMED; distinct collision objects in a rail itinerary, not ticks
MIN_ALIGN = math.sqrt(.5)  # ASSUMED: compatible directions at a fitted seam
FIT = .25          # ASSUMED: existing coarse rail fitting/contact tolerance


@dataclass
class Lane:
    ids: np.ndarray
    p: np.ndarray
    length: np.ndarray
    tangent: np.ndarray
    double: bool
    tube: float
    gauge: float
    line: str = "single"
    parent: int = -1
    axis: np.ndarray | None = None
    twin: np.ndarray | None = None


@dataclass
class Catch:
    lane: int
    j: int
    direction: int
    body: ph.State
    lateral: float               # signed in support cross-plane
    triangle: int
    rebound: object = None
    crown: dict | None = None


@dataclass
class Departure:
    body: ph.State
    tick: float
    kind: str
    source: Catch
    pieces: tuple
    way: tuple
    push: np.ndarray
    rim: object = None
    intervals: tuple = ()
    aimed: int = -1
    horizon: float | None = None
    fallback: bool = False
    joint: tuple | None = None


def support_normal(tangent):
    return sc.unit(UP - (UP @ tangent)*tangent)


def crest(ball, lane, normal, side, speed, push, *, geometry=False):
    """Work to the top of the outer tube, including loss of contact en route.

    The analytically chosen critical angles test the whole potential barrier.
    Time is the conservative-work quadrature, not integration of motion steps.
    """
    R = ball.radius + lane.tube + sc.SKIN
    half = lane.gauge/2
    if not EPS < half < R:
        return None
    phi = math.asin(half/R)
    a = cf.air_impulse(ball,push=push)/ph.DT
    up, across = -float(a @ normal), float(a @ side)
    k = sc.inertia_ratio(ball)*(ball.radius/R)**2
    # Angle measured from the tube crown toward the inner valley.
    def v2(theta):
        return speed**2 + 2*R/(1+k)*(across*(math.sin(phi)-np.sin(theta)) - up*(np.cos(theta)-math.cos(phi)))
    critical = math.atan2(across,up)
    if min(v2(0.),v2(phi),v2(critical) if 0<critical<phi else math.inf) < -EPS:
        return None
    if speed <= EPS and across*math.cos(phi) <= up*math.sin(phi)+EPS:
        return None
    # Loss of pressure can occur on the inner flank, before the crest.
    # Normal force is proportional to up*cos(theta)+across*sin(theta)-v^2/R.
    def load(theta):
        return up*math.cos(theta)+across*math.sin(theta)-v2(theta)/R
    if load(phi) <= 0.:
        end = phi
    elif load(0.) >= 0.:
        end = 0.
    else:
        from scipy.optimize import brentq
        end = brentq(load,0.,phi)
    from .takeoff import _GL_X, _GL_W
    th = phi -(phi-end)*_GL_X**2
    ticks = float(R/ph.DT * (2*(phi-end)*_GL_X/np.sqrt(np.maximum(v2(th),1.e-16))) @ _GL_W)
    displacement = R*((math.cos(end)-math.cos(phi))*normal + (math.sin(phi)-math.sin(end))*side)
    v = math.sqrt(max(0.,float(v2(end))))*(math.cos(end)*side+math.sin(end)*normal)
    values = R,end,displacement,v,ticks
    if not geometry:
        return values
    k2=2*R/(1+k)
    energy=(speed**2,-k2*up,-k2*across)
    return (*values,phi,energy)


class ArrivalFrontier:
    """Antichain of certified arrival cells, never the convex hull of witnesses.

    Position and discrete continuations (including input phase) must agree.
    The caller may supply a velocity interval only when the whole interval
    is a witnessed family; point trajectories supply a singleton interval.
    """
    def __init__(self):
        self.cells=defaultdict(list)

    def accept(self,key,tick,lo,hi,visited,objects):
        items=self.cells[key]
        def covers(a,b):
            ta,la,ha,va,oa=a; tb,lb,hb,vb,ob=b
            return ta<=tb and va==vb and oa<=ob and np.all(la<=lb) and np.all(hb<=ha)
        new=(tick,lo,hi,visited,objects)
        if any(covers(a,new) for a in items):return False
        items[:]=[a for a in items if not covers(new,a)];items.append(new)
        return True


class RailPaths:
    def __init__(self, g, obs, landing):
        self.g,self.obs,self.landing = g,obs,landing
        self.surface = sc.SurfacePaths(g,obs,landing)
        self.lanes={}
        self.endpoint=[]
        lo,hi,owners=[],[],[]
        metadata=g.lanes
        if any(m['double'] and 'axes' not in m for m in metadata) and len(obs.mesh.tubes):
            rebuilt=rails.lanes(obs.mesh.tubes)
            if len(rebuilt)==len(metadata) and all(np.array_equal(a['points'],b['points']) for a,b in zip(metadata,rebuilt)):
                metadata=rebuilt
        self.pairs={}
        for ids in _lane_chains(g):
            if len(ids)<2: continue
            li=int(g.lane[ids[0]]); m=metadata[li]; p=self.surface.rail_geometry[li]
            if len(p)<2:continue
            delta=np.diff(p,axis=0)
            L=np.linalg.norm(delta,axis=1)
            if np.any(L<=EPS): continue
            self.lanes[li]=Lane(ids,p,L,delta/L[:,None],m['double'],m['radius'],m['gauge'],'groove' if m['double'] else 'single',li)
            self.endpoint.extend(((li,1,p[0]),(li,-1,p[-1])))
            lo.extend(np.minimum(p[:-1],p[1:])); hi.extend(np.maximum(p[:-1],p[1:]))
            owners.extend([li]*len(L))
            if not m.get('axes') or not all(len(a)>=2 for a in m['axes']):continue
            a,b=(rails.straight_pieces(x) for x in m['axes'])
            # Left/right is in the oriented mesh-chain frame, not world Z.
            t=sc.unit(p[-1]-p[0]); side=sc.unit(np.cross(support_normal(t),t))
            if (a[0]-b[0])@side<0.:a,b=b,a
            pair=[]
            for index,(line,ax,twin) in enumerate((('left_crown',a,b),('right_crown',b,a))):
                h=len(metadata)+2*li+index
                d=np.diff(ax,axis=0); length=np.linalg.norm(d,axis=1); tangents=d/length[:,None]
                tops=ax+(ph.WOOD.radius+m['radius']+sc.SKIN)*np.vstack([support_normal(t) for t in (*tangents,tangents[-1])])
                self.lanes[h]=Lane(ids,tops,length,tangents,False,m['radius'],m['gauge'],line,li,ax,twin)
                self.endpoint.extend(((h,1,tops[0]),(h,-1,tops[-1])))
                lo.extend(np.minimum(tops[:-1],tops[1:]));hi.extend(np.maximum(tops[:-1],tops[1:]));owners.extend([h]*len(length))
                pair.append(h)
            self.pairs[li]=tuple(pair)
        self.tree=cKDTree([x[2][XZ] for x in self.endpoint]) if self.endpoint else None
        self.seam_cache={}
        self.piece_lo=np.asarray(lo).reshape(-1,3); self.piece_hi=np.asarray(hi).reshape(-1,3)
        self.piece_lanes=np.asarray(owners,dtype=int)
        self.bands=None
        self.inverse_targets=None
        self.counts=defaultdict(int)
        self.labels=None

    def geometry_support(self, center, ball, double, tangent=None):
        """Use real mesh reactions, not just the existence of a fitted line."""
        m=self.obs.mesh
        cs=[c for c in m.contacts(center,ball.radius,FIT) if m.rail[c.triangle] and c.normal[1]>EPS]
        if double:
            up=support_normal(tangent) if tangent is not None else UP
            for i,a in enumerate(cs):
                for b in cs[i+1:]:
                    n,t=a.normal,b.normal
                    if np.linalg.norm(a.point-b.point)<=FIT: continue
                    nh,th=n-(n@up)*up,t-(t@up)*up
                    if sc.unit(nh) @ sc.unit(th) < -.9 and sc.unit(n+t)[1]>=WALK_NY:
                        return a.triangle
            return -1
        return next((c.triangle for c in cs if c.normal[1]>=WALK_NY),-1)

    def fit(self, lane, pos):
        return rails.project_pieces(lane.p,pos)

    def line_for(self, lane):
        return self.lanes[lane].line

    def cross_section(self,lane,j,p,ball):
        t=lane.tangent[j]; up=support_normal(t)
        a,b=lane.axis[j:j+2]
        center=a+np.clip((p-a)@t,0.,lane.length[j])*t
        _,other=rails.project_pieces(lane.twin,center)
        w=other-center; w=w-(w@t)*t
        side=sc.unit(np.cross(up,t)); side*=1 if side@w<0. else -1
        g=-float(w@side); dy=float(w@up); r=ball.radius+lane.tube+sc.SKIN
        # Equal expanded tube radii: the cross-sectional intersection
        # satisfies w . radial = |w|^2/2. Pick the upper inner solution.
        length=math.hypot(g,dy)
        inner=-math.pi/2
        if EPS<length<2*r:
            aim=math.atan2(-g,dy)
            inner=aim+math.acos(length/(2*r))
        return center,t,up,side,r,inner

    def rail_contacts(self,hit,ball,owner):
        """Actual manifold; coincident faces on one tube do not form a pair."""
        m=self.obs.mesh
        contacts=list(hit.get('manifold',()))
        default=hit['position']-hit['normal']*ball.radius
        contacts.append(dict(triangle=hit['triangle'],point=hit.get('point',default),normal=hit['normal']))
        return [x for x in contacts if m.rail[x['triangle']] and m.owner[x['triangle']]==owner and x['normal'][1]>EPS]

    def paired_capture(self,ball,hit,omega,li,contacts):
        p=hit['position']; omega=hit.get('incoming_omega',omega); ids=self.pairs[li]; groups=[[],[]]; sections=[]
        for h in ids:
            l=self.lanes[h]; j,_=rails.project_pieces(l.axis,p)
            sections.append((j,self.cross_section(l,j,p,ball)))
        for contact in contacts:
            q=contact['point']
            scores=[]
            for h in ids:
                l=self.lanes[h]; _,at=rails.project_pieces(l.axis,q)
                scores.append(abs(np.linalg.norm(q-at)-l.tube))
            k=int(np.argmin(scores))
            if scores[k]<=FIT:groups[k].append(contact)
        if not any(groups):return None
        both=all(groups) and any(sc.unit(a['normal']+b['normal'])[1]>=WALK_NY
                    and (a['normal']@sections[0][1][3])*(b['normal']@sections[0][1][3])<-EPS
                    for a in groups[0] for b in groups[1])
        owner=int(self.obs.mesh.owner[hit['triangle']]); material=self.obs.mesh.entities[owner]
        incoming=ph.State(p,hit['incoming'],np.zeros(3) if omega is None else omega)
        if both:
            lane=self.lanes[li]; j,q=self.fit(lane,p); t=lane.tangent[j]; up=support_normal(t)
            if np.linalg.norm(p-q)>FIT:return None
            v=sc.rolling_entry(ball,incoming.vel,up,omega); u=float(v@t)
            if abs(u)<=EPS:return None
            b=ph.State(q,u*t,ph.rolling_spin(u*t,up,ball.radius));d=1 if u>0 else -1
            n=hit.get('support_normal',up)
            rebound,regime=cf.collision_impulse(ball,incoming,n,float(material.get('elasticity',.3)))
            if rebound.vel@n<=EPS:rebound=None
            else:rebound.pos=rebound.pos+8*EPS*n
            return Catch(li,j,d,b,float(v@np.cross(up,d*t)),hit['triangle'],rebound)
        k=max(range(2),key=lambda k:max((n['normal'][1] for n in groups[k]),default=-1.))
        h=ids[k]; l=self.lanes[h]; j,(axis,t,up,side,r,inner)=sections[k]
        radial=p-axis; theta=math.atan2(float(radial@side),float(radial@up))
        n=sc.unit(up*math.cos(theta)+side*math.sin(theta)); sx=side*math.cos(theta)-up*math.sin(theta)
        rolled=sc.rolling_entry(ball,incoming.vel,n,omega); u=float(rolled@t); v=float(rolled@sx)
        if abs(u)<=EPS:return None
        d=1 if u>0 else -1
        bounce,regime=cf.collision_impulse(ball,incoming,hit['normal'],float(material.get('elasticity',.3)))
        if bounce.vel@hit['normal']<=EPS:bounce=None
        else:bounce.pos=bounce.pos+8*EPS*hit['normal']
        at=ph.State(axis+r*n,u*t+v*sx,ph.rolling_spin(u*t+v*sx,n,ball.radius))
        data=dict(line=l.line,contact='one_tube',angle=theta,lateral_offset=float(radial@side),lateral_speed=v,
                  incoming_lateral_speed=float(incoming.vel@sx),inner_angle=inner,
                  tube_axis=axis, travel_side='right' if side@np.cross(d*t,up)>0. else 'left')
        adjoining=n[1]>=WALK_NY and inner-EPS<=theta<=math.pi/2 and np.linalg.norm(at.pos-p)<=FIT
        if bounce is not None and sc.apex(ball,bounce.vel,n,sc.keys_along(u*t))>sc.RECAPTURE:adjoining=False
        data['supported']=bool(adjoining)
        data['impact_regime']=regime
        data['outgoing_velocity']=bounce.vel if bounce is not None else incoming.vel
        data['outgoing_omega']=bounce.omega if bounce is not None else incoming.omega
        push=sc.keys_along(u*t); force=cf.air_impulse(ball,push=push)/ph.DT
        angular=railcrown.swing(theta,v,-float(force@up),float(force@side),r,
                               sc.inertia_ratio(ball)*(ball.radius/r)**2,inner)
        data['barrier']=angular[0].barrier if angular else 0.
        data['decision']='rebound' if not adjoining else angular[-1].event if angular else 'balance'
        if not adjoining and bounce is None:return None
        return Catch(h,j,d,at,v,hit['triangle'],bounce,data)

    def capture(self, ball, hit, node, omega=None):
        """Inelastic normal constraint for the *additional* rail-ride path."""
        omega=hit.get('incoming_omega',omega)
        gap=np.maximum(0.,np.maximum(self.piece_lo-hit['position'],hit['position']-self.piece_hi))
        possible=np.unique(self.piece_lanes[np.einsum('ij,ij->i',gap,gap)<=(ball.radius+FIT)**2])
        n=hit.get('support_normal',hit['normal'])
        choices=[]; owner=self.obs.mesh.owner[hit['triangle']]
        cs=self.rail_contacts(hit,ball,owner)
        physical=set(self.lanes[i].parent for i in possible if self.lanes[i].parent in self.pairs)
        for li in sorted(physical):
            c=self.paired_capture(ball,hit,omega,li,cs)
            if c is not None:choices.append((False,float(np.linalg.norm(c.body.pos-hit['position'])),li,c))
        for li in sorted(possible):
            if self.lanes[li].parent in self.pairs:continue
            lane=self.lanes.get(li)
            if lane is None:continue
            j,p=self.fit(lane,hit['position']); t=lane.tangent[j]; up=support_normal(t)
            dist=float(np.linalg.norm(hit['position']-p))
            if dist>ball.radius+FIT:continue
            pair=False
            if lane.double:
                cross=sc.unit(np.cross(up,t)); width=lane.gauge/2+FIT
                if abs((hit['position']-p)@cross)>width:continue
                f=self.geometry_support(p,ball,True,t)
                pair=f>=0 and self.obs.mesh.owner[f]==owner
            if not pair:
                # A single upward contact may use its own nearest fitting,
                # not another unrelated side-by-side lane.
                if n[1]<WALK_NY:continue
                f=self.geometry_support(p,ball,False)
                if f<0 or self.obs.mesh.owner[f]!=owner:continue
            sv=sc.rolling_entry(ball,hit['incoming'],up if pair else n,omega)
            speed=float(sv@t); direction=1 if speed>=0 else -1
            if abs(speed)<=EPS:continue
            v=speed*t; t=t*direction
            lateral=float(sv@sc.unit(np.cross(up,t)))
            reaction=hit.get('support_normal',n)
            material=self.obs.mesh.entities[owner]
            incoming=ph.State(hit['position'],hit['incoming'],np.zeros(3) if omega is None else omega)
            bounced,regime=cf.collision_impulse(ball,incoming,reaction,float(material.get('elasticity',.3)))
            if bounced.vel@reaction<=EPS:
                bounced=None
            else:
                bounced.pos=bounced.pos+reaction*(8*EPS)
            choices.append((not pair,dist,int(li),Catch(li,j,direction,ph.State(p,v,ph.rolling_spin(v,up,ball.radius)),lateral,hit['triangle'],bounced)))
        return min(choices,key=lambda k:k[:3])[-1] if choices else None

    def candidates_at_end(self, ball, dep):
        """Nearby compatible rail starts, including a small downward step."""
        if self.tree is None: return ()
        lane=self.lanes[dep.source.lane]; p=dep.body.pos; t=sc.unit(dep.body.vel)
        key=(dep.source.lane,dep.source.direction)
        # Endpoint neighbourhood depends on the geometry, not on the recording.
        if key not in self.seam_cache:
            self.seam_cache[key]=self.tree.query_ball_point(p[XZ],2*ball.radius+FIT)
        options=[]
        for i in self.seam_cache[key]:
            li,d,q=self.endpoint[i]
            if li==dep.source.lane: continue
            next=self.lanes[li]; j=0 if d>0 else len(next.length)-1
            tangent=d*next.tangent[j]; n=support_normal(tangent)
            side=sc.unit(np.cross(n,tangent))
            off=q-p
            if (tangent@t<MIN_ALIGN or off@t<-ball.radius or abs(off@side)>ball.radius
                    or not -4*ball.radius-FIT<=off[1]<=FIT):
                continue
            options.append((li,d,q,tangent))
        return options

    def target_sleeves(self, ball, catch):
        """Nearby compatible successor chains; along each, use real mesh pieces."""
        lane=self.lanes[catch.lane]
        j=len(lane.length)-1 if catch.direction>0 else 0
        end=lane.p[-1 if catch.direction>0 else 0]
        tangent=catch.direction*lane.tangent[j]
        at=Departure(ph.State(end,tangent),0.,'chain_end',catch,(),(),np.zeros(2))
        candidates=self.candidates_at_end(ball,at)
        if lane.axis is not None and lane.parent in self.pairs:
            candidates=[*candidates,*[(i,catch.direction,None,None) for i in (lane.parent,*self.pairs[lane.parent]) if i!=catch.lane]]
        for i,d,_,_ in candidates:
            dest=self.lanes[i]
            half=(dest.gauge/2 if dest.double else ball.radius/2)+FIT
            for k,t in enumerate(dest.tangent):
                if np.linalg.norm(t[XZ])<EPS or d*t@tangent<MIN_ALIGN:continue
                a,b=dest.p[k:k+2]
                # This is an output ID, not a search target. The sleeve is
                # entirely independent of the density of the coarse grid.
                node=int(dest.ids[np.argmin(np.linalg.norm(self.g.pos[dest.ids]-a,axis=1))])
                yield railjoint.Sleeve(i,k,d,a,b,half,node)

    def joints(self, tf, ball, catch, law, end):
        for shape in self.target_sleeves(ball,catch):
            self.counts['rail_joint_targets']+=1
            for point,interval in railjoint.invert_sleeve(tf,ball,law,0.,end,shape):
                self.counts['rail_joint_intervals']+=1
                yield shape,point,interval

    def forward_joints(self, tf, ball, source, axis, hold, dur, segments, way, elapsed, recorded=False):
        """A straight tube only releases tangentially under outward load."""
        speed=float(np.linalg.norm(source.body.vel)); t=sc.unit(source.body.vel)
        up=support_normal(t)
        if cf.air_impulse(ball,push=hold)@up<=EPS:return
        u=axis.terminal_speed
        law=inverse.ReleaseLaw(source.body.pos,source.body.vel,ph.DT*u*t,
                               -(speed-u)*ph.DT/axis.loss*t,(u-speed)*t,math.log1p(-axis.loss))
        for target,q,(lo,hi) in self.joints(tf,ball,source,law,dur):
            for root in dict.fromkeys((lo,(lo+hi)*.5,hi)):
                state=law.at(root); aim=inverse.objective(tf,ball,state,q)
                if aim is None:continue
                push,n=aim
                if max(abs(push))>1.+EPS:continue
                if recorded:push=hold
                if cf.air_impulse(ball,push=push)@up<=EPS:continue
                c=replace(source,body=state,rebound=None)
                piece=sc.segment('rail',self.surface.name(source.triangle),source.body,state,root,hold,'joint_inverse')
                yield Departure(state,elapsed+root,'joint_tangent',c,(*segments,piece),(*way,state.pos),np.clip(push,-1.,1.),
                                intervals=tuple(float(axis.at(speed,t0)[0]) for t0 in (lo,hi)),aimed=target.node,
                                horizon=n+1.,joint=(target.lane,target.j))

    def departure_support(self, tf,ball,dep,schedule):
        """Attach the support prefix to the original tube faces, not all of an owner."""
        m=self.obs.mesh; owner=int(m.owner[dep.source.triangle]); release=dep.body.pos
        r=ball.radius+sc.SKIN
        span=2*r
        near=tf._box_triangles((release-span)[None],(release+span)[None],self.obs.lo,self.obs.hi,m._grid)
        faces=np.array([i for i in near if m.owner[i]==owner and m.rail[i]],dtype=int)
        tubes=m.tubes
        crown_release=dep.kind in ('crown_pressure','crown_open')
        if crown_release:
            lane=self.lanes[dep.source.lane]
            tubes=railjoint.tube_span(tubes,*lane.axis[dep.source.j:dep.source.j+2]) if lane.axis is not None else tubes[:0]
        axes,used=railjoint.bind_axes(tubes,self.obs.tris[faces],release,r)
        if crown_release and not axes:return None,0.
        if not axes:
            # Synthetic or unfitted geometry can still use its actual
            # connected upward mesh prefix. Never turn failure into immunity.
            source=tf.rail_support(self.obs,dep.source.body.pos,ball.radius)
            return source,max(0.,self.obs.rail_clearance(ball,dep.body,schedule[0][1],schedule[0][0],source))
        source=tf.RailSupport(owner,np.sort(faces[used]),axes)
        state=dep.body; prev=0.; loss=ball.lin_damp*ph.DT
        for end,push in schedule:
            width=end-prev
            if width<=EPS:continue
            imp=cf.air_impulse(ball,push=push)
            piece=motion.flight(state,imp,loss,width)
            exit=railjoint.support_end(piece,axes,r)
            if exit<width-EPS:return source,prev+exit
            at=cf.flight(ball,state,int(width),push=push)
            state=ph.State(piece.point(width),at.vel,at.omega)
            prev=end
        return source,prev

    def continued_interval(self, tf, ball, law, end, ids=None):
        targets = getattr(self.landing,'conformal_targets',None)
        if ids is None and targets is not None and targets.lowest is not None:
            radius=ball.radius+sc.SKIN
            limits=inverse.reach_planes(tf,ball,law,0.,end,targets.lowest,math.inf)
            if limits is not None:
                def planes(p):
                    bottom,top=p.heights(radius)
                    band=inverse.reach_planes(tf,ball,law,0.,end,bottom,top)
                    return () if band is None else (band[2],)
                witness = None
                def valid(q):
                    nonlocal witness
                    self.counts['rail_inverse_targets']+=1
                    witness=inverse.invert(tf,ball,law,0.,end,q)
                    return bool(witness)
                for v,q,k,label in targets.representatives(ball.name,radius,*limits[:2],planes,valid,
                                                           stats=self.counts,prefix='rail_road',anchor=law.p0[XZ],alternatives=True):
                    for ab in witness:
                        self.counts['rail_intervals']+=1
                        yield v,q,ab
            rest = self.g.info['rest_places']
            residual=np.r_[targets.unmapped,np.arange(rest,len(self.g.pos))]
            if self.bands is None:
                self.bands=[(a,b,residual[i],tree) for a,b,i,tree in tf._bands(self.g.pos[residual])]
        if ids is None:
            if self.bands is None:self.bands=list(tf._bands(self.g.pos))
            ids=inverse.candidates(tf,ball,law,0.,end,self.bands)
        cache={}; ranges=law.ranges(0.,end)
        for v in ids:
            q=self.g.pos[v]
            if q[1] not in cache:cache[q[1]]=inverse.bound(tf,ball,ranges,q[1])
            initial=cache[q[1]]
            if initial is None:continue
            # Rest-node centres are the same targets used by coarse launch
            # proposals. The exact flight may hit another support sooner.
            self.counts['rail_inverse_targets']+=1
            for ab in inverse.invert(tf,ball,law,0.,end,q,initial=initial):
                self.counts['rail_intervals']+=1
                yield v,q,ab

    def side_departures(self, tf, ball, c, pos, speed, axis, t, up, hold, dur, avail,
                        segments, way, elapsed, *, controls=None, clock=0.):
        lane=self.lanes[c.lane]; name=self.surface.name(c.triangle)
        for sign in (-1,1):
            side=sign*sc.unit(np.cross(up,t)); push=sc.keys_along(t+side) if controls is None else hold
            lateral=c.lateral*math.exp(-ball.lin_damp*ph.DT*elapsed)
            ext=crest(ball,lane,up,side,max(0.,sign*lateral),push,geometry=True)
            if ext is None:continue
            R,angle,d,vs,arc,phi,energy=ext; rim=None; extra=0.
            valley = -R*(math.cos(phi)*up+math.sin(phi)*side)
            if angle<=EPS:
                rim=tf.rim_release(ball,d-R*up,up,side,vs,0,R,inertia_ratio=sc.inertia_ratio(ball),push=push)
                if rim is None:continue
                extra=rim.seconds; vs=rim.start.vel; d=rim.start.pos
            side_axis=sc.law(ball,up,push).along(t)
            dx,v0=side_axis.at(speed,arc)
            L=math.log1p(-axis.loss); power=math.exp(L*arc)
            A=ph.DT*(-math.expm1(L*arc))/axis.loss + power*extra
            u=axis.terminal_speed
            release=inverse.ReleaseLaw(pos+d+(dx+v0*extra)*t,vs+v0*t,ph.DT*u*t,
                       (speed-u)*(A-ph.DT/axis.loss)*t,(speed-u)*power*t,L)
            # Both the on-rail part and the side climb must complete before
            # the end of this straight section / recorded control span.
            actual=min(dur, dur-arc-extra/ph.DT if controls is not None else dur)
            if actual<=EPS:continue
            from scipy.optimize import brentq
            advance=lambda tau: float((release.at(tau).pos-pos-d)@t)
            if advance(0.)>avail+EPS:continue
            if advance(actual)>avail:
                actual=brentq(lambda a: advance(a)-avail,0.,actual)
            if actual<=EPS:continue
            visited=set()
            def aims():
                for v,q,ab in self.continued_interval(tf,ball,release,actual,self.inverse_targets):
                    yield v,q,ab,None
                for target,q,ab in self.joints(tf,ball,c,release,actual):
                    yield target.node,q,ab,(target.lane,target.j)
            for v,q,ab,joint in aims():
                # Endpoint/midpoint, then quartiles when an initial witness is
                # blocked. They come from the solved interval, not a rail grid.
                p1,p2=ab; s0=float(axis.at(speed,p1)[0]); s1=float(axis.at(speed,p2)[0]); total=arc+extra/ph.DT
                for f in (0.,.5,1.,.25,.75):
                    root=p1+(p2-p1)*f
                    flight=release.at(root); aim=inverse.objective(tf,ball,flight,q)
                    if aim is None:continue
                    force,n=aim
                    if np.max(abs(force))>1+1.e-6:continue
                    control=np.clip(force,-1,1) if controls is None else controls(clock+elapsed+root+total)[0][1]
                    ident=(float(root),tuple(control),v) if controls is None else (float(root),tuple(control),joint is not None)
                    if ident in visited:continue
                    visited.add(ident)
                    displacement,sv=axis.at(speed,root); at=ph.State(pos+displacement*t,sv*t)
                    source=Catch(c.lane,c.j,c.direction,at,lateral,c.triangle)
                    prefix=sc.segment('rail',name,ph.State(pos,speed*t),at,root,hold,'inverse_release')
                    geom = [motion.arc(at.pos+valley,R*up,R*side,phi,angle-phi,t,energy,
                                       (sv,side_axis.impulse,side_axis.loss))]
                    if rim is not None and extra>0.:
                        dx0,vx0=side_axis.at(sv,arc)
                        geom.append(motion.arc(at.pos+dx0*t+rim.pivot,R*up,R*side,0.,rim.angle,
                                               rim.along*rim.axis+vx0*t,rim.energy))
                    part=sc.segment('rail_crest',name,at,flight,total,push,'side_energy',geometry=tuple(geom))
                    moved_rim=None
                    if rim is not None:
                        from dataclasses import replace
                        moved_rim=replace(rim,pivot=rim.pivot+(flight.pos-rim.start.pos),start=flight)
                    yield Departure(flight,elapsed+root+total,'side' if joint is None else 'joint_crest',source,(*segments,prefix,part),
                              (*way,at.pos,flight.pos),control,moved_rim,(s0,s1),v,n+1. if controls is None else None,f in (.25,.75),joint)

    def balanced_switches(self,tf,ball,c,j,at,t,up,side,theta,v,hold,law,limit,parts,way,elapsed):
        """Invert a delayed lateral key on an exactly balanced straight crown.

        The cross-section stays fixed before the switch. After it, angle,
        pressure root and arc duration are independent of switch station.
        Axial composition is therefore still exponential-linear, so use
        the same continuous destination inverse as a groove side release.
        """
        force=cf.air_impulse(ball,push=hold)/ph.DT
        if abs(v)>EPS or abs(-(force@up)*math.sin(theta)+(force@side)*math.cos(theta))>EPS:return
        from scipy.optimize import brentq
        u=float(at.vel@t); name=self.surface.name(c.triangle)
        if limit<=EPS:return
        room=float(law.at(u,limit)[0])
        source=replace(c,j=j,body=at,rebound=None)
        for x in (-1.,0.,1.):
            for z in (-1.,0.,1.):
                push=np.array([x,z])
                if abs(cf.air_impulse(ball,push=push)@side-force@side*ph.DT)<=EPS:continue
                control=lambda _:((math.inf,push),)
                dep=next(self.walk_crown(tf,ball,source,control,0.),None)
                if dep is None or dep.kind not in ('crown_pressure','crown_twin','crown_open') or dep.source.j!=j:continue
                delay=dep.tick; next_axis=sc.law(ball,up,push).along(t)
                ex=math.exp(math.log1p(-next_axis.loss)*delay)
                drift=ph.DT*(1.-ex)/next_axis.loss
                base=law.terminal_speed; delta=u-base
                release=inverse.ReleaseLaw(dep.body.pos,dep.body.vel,ph.DT*base*t,
                    delta*(drift-ph.DT/law.loss)*t,-delta*ex*t,math.log1p(-law.loss))
                dist=lambda q: float((release.at(q).pos-at.pos)@t)
                if dist(0.)>=room-EPS:continue
                hi=limit if dist(limit)<=room else brentq(lambda q:dist(q)-room,0.,limit)
                def aims():
                    for w,q,ab in self.continued_interval(tf,ball,release,hi,self.inverse_targets):yield w,q,ab,None
                    for target,q,ab in self.joints(tf,ball,source,release,hi):yield target.node,q,ab,(target.lane,target.j)
                for w,q,(lo,upper),joint in aims():
                    hi=upper
                    if hi-lo<=EPS:continue
                    span=tuple(float(law.at(u,h)[0]) for h in (lo,hi))
                    for fraction in (0.,.5,1.,.25,.75):
                        tick=lo+fraction*(hi-lo); goal=release.at(tick)
                        aim=inverse.objective(tf,ball,goal,q)
                        if aim is None or np.max(abs(aim[0]))>1.+EPS:continue
                        dx,speed=law.at(u,tick); st=ph.State(at.pos+dx*t,speed*t,at.omega)
                        owner=self.obs.mesh.owner[c.triangle]
                        if not self.surface.clear(tf,ball,[at.pos,st.pos],owner,up):continue
                        # Rebuild and validate the representative arc against
                        # the exact mesh; inversion is only an envelope.
                        move=next(self.walk_crown(tf,ball,replace(source,body=st),control,0.),None)
                        if move is None or move.kind!=dep.kind or abs(move.tick-delay)>1.e-5:continue
                        prefix=dict(sc.segment('rail',name,at,st,tick,hold,'crown_key_switch',geometry=(motion.rolling(at,law.impulse*t,law.loss,tick),)),rail_line=self.line_for(c.lane))
                        yield replace(move,tick=elapsed+tick+move.tick,pieces=(*parts,prefix,*move.pieces),
                                      way=(*way,*move.way),push=np.clip(aim[0],-1.,1.),intervals=span,
                                      aimed=w,horizon=aim[1]+1.,joint=joint,fallback=fraction in (.25,.75))

    def crown_miter(self, ball, lane, j, sign, state, owner, push, *, report=None):
        """Verify a continuous, compressive handoff on the actual tube skin."""
        from .conformal import PLANE_ERROR
        def reject(reason):
            if report is not None:report['reason']=reason
            return None
        k=j+sign
        base,t,up,side,r,inner=self.cross_section(lane,k,state.pos,ball)
        q=state.pos-base; n=sc.unit(q); t=sign*t
        if abs(np.linalg.norm(q)-r)>EPS or abs(n@t)>EPS:return reject('no_equal_finite_cylinder_contact')
        theta=math.atan2(n@side,n@up)
        if n[1]<=EPS or not inner-EPS<=theta<=math.pi/2:return reject('receiving_cross_section_not_supported')
        vn=float(state.vel@n)
        if vn>EPS or state.vel@t<=EPS:return reject('separating_or_reversing')
        m=self.obs.mesh
        contacts=m.contacts(state.pos,ball.radius,FIT)
        if any(c.distance<ball.radius-sc.RECAPTURE for c in contacts):return reject('original_mesh_penetration')
        start=lane.axis[k if sign>0 else k+1]
        real=[]
        for c in contacts:
            if not m.rail[c.triangle] or m.owner[c.triangle]!=owner or c.normal[1]<=EPS or c.normal@n<=0.:continue
            station=float((c.point-start)@t)
            if not -PLANE_ERROR<=station<=lane.length[k]+PLANE_ERROR:continue
            off=c.point-start-np.clip(station,0.,lane.length[k])*t
            # PLANE_ERROR associates the mesh skin to the fitted cylinder.
            # The ideal offset already contains SKIN; an export-sized
            # discrepancy is not a second dynamic recapture allowance.
            if abs(np.linalg.norm(off)-lane.tube)<=PLANE_ERROR and c.distance<=ball.radius+sc.SKIN+PLANE_ERROR:
                real.append(c)
        if not real:return reject('no_receiving_skin_contact')
        elasticity=m.entities[owner].get('elasticity',.3)
        rebound=-cf.collision_restitution(ball,elasticity)*vn*n
        if sc.apex(ball,rebound,n,push)>sc.RECAPTURE:return reject('rebound_exceeds_recapture_band')
        v=state.vel-vn*n
        side_v=float(v@(side*math.cos(theta)-up*math.sin(theta)))
        load=-float(cf.air_impulse(ball,push=push)@n)/ph.DT-side_v**2/r
        if load<=EPS:return reject('receiving_normal_load_nonpositive')
        return ph.State(state.pos,v,ph.rolling_spin(v,n,ball.radius)), int(real[0].triangle), load

    def walk_crown(self,tf,ball,c,controls,clock,*,diagnostics=None,supported_track=None):
        """Separate exact axial recurrence and fixed-input cross-section events."""
        if not c.crown['supported']:return
        from scipy.optimize import brentq
        lane=self.lanes[c.lane]; sign=c.direction; name=self.surface.name(c.triangle)
        # The angular initial value is not reset to the crown.
        holds=(None,) if controls is not None else tuple(dict.fromkeys(
            tuple(sc.keys_along(c.body.vel+a)) for a in (np.zeros(3),*[
                q*sc.unit(np.cross(UP,lane.tangent[c.j]))*max(1.,np.linalg.norm(c.body.vel)) for q in (-1,1)])))
        for choice in holds:
            j=c.j; at=c.body; elapsed=0.; pieces=[]; way=[at.pos]; face=c.triangle
            for _ in range(sc.MAX_SUPPORT_PIECES):
                axis,t,up,side,r,inner=self.cross_section(lane,j,at.pos,ball)
                t=sign*t; radial=at.pos-axis
                theta=math.atan2(float(radial@side),float(radial@up))
                across=side*math.cos(theta)-up*math.sin(theta)
                v=float(at.vel@across); u=float(at.vel@t)
                def note(event, position=None, **values):
                    if diagnostics is not None:diagnostics.append(dict(event=event,piece=j,tick=elapsed,
                        position=at.pos if position is None else position,angle=theta,lateral_speed=v,axial_speed=u,**values))
                if u<=EPS:note('no_forward_velocity');break
                push=np.array(choice) if controls is None else controls(clock+elapsed)[0][1]
                limit=math.inf if controls is None else controls(clock+elapsed)[0][0]
                if limit<=EPS:break
                law=sc.law(ball,up,push).along(t)
                finish=lane.axis[j+1 if sign>0 else j]; avail=max(0.,float((finish-axis)@t))
                end=law.over(u,avail)
                stopped=end is None
                n=cf._turn(u,law.impulse,law.loss) if stopped and law.impulse<0. else math.inf if stopped else end[1]/ph.DT
                if not math.isfinite(min(limit,n)):note('axial_run_unproved');break
                limit=min(limit,n)
                f=cf.air_impulse(ball,push=push)/ph.DT
                inertia=sc.inertia_ratio(ball)*(ball.radius/r)**2
                actions=railcrown.swing(theta,v,-float(f@up),float(f@side),r,inertia,inner)
                event='span'; body=at; used=0.; obstructed=False; joined=None
                k=j+sign; owner=int(self.obs.mesh.owner[face])
                receiving=sign*lane.tangent[k] if 0<=k<len(lane.length) else None
                bent=receiving is not None and receiving@t<1.-EPS and receiving@t>EPS
                def joint(path, offset):
                    if not bent:return None
                    z=railcrown.miter_root(path,finish,t,receiving)
                    if z is None:return None
                    duration=z if path.kind=='rolling' else path.duration(z)/ph.DT
                    _,sv=law.at(u,offset+duration)
                    vel=t*sv
                    if path.kind=='arc':
                        x,turn,_,c0,A,B,roll=path.arc
                        ang=x+turn*z
                        v2=c0+A*(math.cos(ang)-math.cos(x))+B*(math.sin(ang)-math.sin(x))
                        lv=math.copysign(math.sqrt(max(0.,v2)),turn)
                        vel=vel+lv*(side*math.cos(ang)-up*math.sin(ang))
                    p=path.point(z)
                    report={}
                    result=self.crown_miter(ball,lane,j,sign,ph.State(p,vel),owner,push,report=report if diagnostics is not None else None)
                    if result is None:
                        note('miter_mesh_or_load_unproved',p,**report)
                        return None
                    cut=path._replace(end=z) if path.kind=='rolling' else motion.arc(path.p,path.v,path.b,x,turn*z,t,(c0,A,B),roll)
                    return result,cut,duration,vel
                if not actions:
                    if controls is None:
                        yield from self.balanced_switches(tf,ball,c,j,at,t,up,side,theta,v,push,law,limit,parts,way,elapsed)
                    dx,speed=law.at(u,limit); body=ph.State(at.pos+t*dx,t*speed,at.omega); used=limit
                    geom=(motion.rolling(ph.State(at.pos,t*u),law.impulse*t,law.loss,used),)
                    contact=joint(geom[0],0.)
                    if contact is not None:
                        joined,path,used,vel=contact; geom=(path,); body=ph.State(path.point(path.end),vel,at.omega)
                    if not self.surface.clear(tf,ball,[at.pos,body.pos],owner,up):note('crown_mesh_obstruction');break
                    pieces.append(dict(sc.segment('rail',name,at,body,used,push,'crown_balance',geometry=geom),rail_line=lane.line))
                    if supported_track is not None:supported_track.append(pieces[-1])
                    way.append(body.pos)
                for action in actions:
                    initial=body; _,v0=law.at(u,used); dx,_=law.at(u,used)
                    p=axis+dx*t; turn=action.end-action.start
                    if abs(turn)<=EPS:
                        event=action.event;break
                    path=motion.arc(p,r*up,r*side,action.start,turn,t,action.energy,(v0,law.impulse,law.loss))
                    dur=path.duration(1.)/ph.DT
                    if action.event=='asymptote':dur=math.inf
                    left=limit-used
                    if dur>left:
                        z=brentq(lambda z:path.duration(z)/ph.DT-left,0.,1.) if left>EPS else 0.
                        angle=action.start+z*turn
                        de=(action.energy[1]*(math.cos(angle)-math.cos(action.start))+
                            action.energy[2]*(math.sin(angle)-math.sin(action.start)))
                        v1=math.copysign(math.sqrt(max(0.,action.energy[0]+de)),turn)
                        path=motion.arc(p,r*up,r*side,action.start,z*turn,t,action.energy,(v0,law.impulse,law.loss))
                        used=limit;event='span'
                    else:
                        angle=action.end;v1=action.end_speed;used+=dur;event=action.event
                    contact=joint(path,used-path.duration(1.)/ph.DT)
                    if contact is not None:
                        joined,cut,duration,vel=contact
                        used+=duration-path.duration(1.)/ph.DT
                        path=cut; event='miter'; angle=path.arc[0]+path.arc[1]
                        v1=float(vel@(math.cos(angle)*side-math.sin(angle)*up))
                    _,speed=law.at(u,used); normal=math.cos(angle)*up+math.sin(angle)*side
                    velocity=speed*t+v1*(math.cos(angle)*side-math.sin(angle)*up)
                    body=ph.State(path.point(1.),velocity,ph.rolling_spin(velocity,normal,ball.radius))
                    # Preserve the analytic arc for the actual map guards.
                    m=self.obs.mesh;lo,hi=path.bounds(0.,1.);radius=ball.radius
                    near=tf._box_triangles((lo-radius)[None],(hi+radius)[None],self.obs.lo,self.obs.hi,m._grid)
                    faces=[int(i) for i in near if m.owner[i]!=owner]
                    if faces:
                        mask=np.zeros(len(m.A),bool);mask[faces]=True; obs=m.subset(mask)
                        if motion.near_hit(path,lambda p:obs.distance(p,radius+1.)[0],radius-sc.RECAPTURE) is not None:
                            obstructed=True;break
                    if any(motion.box_hit(path,a,b) is not None for a,b in self.obs.boxes):
                        obstructed=True;break
                    ticks=path.duration(1.)/ph.DT
                    pieces.append(dict(sc.segment('rail_crown',name,initial,body,ticks,push,event,geometry=(path,)),
                                       rail_line=lane.line,barrier=action.barrier,lateral_start=action.speed,lateral_end=v1))
                    if supported_track is not None:supported_track.append(pieces[-1])
                    way.append(body.pos)
                    if event!='turn':break
                else:
                    if actions:event='invalid'
                if obstructed:note('crown_mesh_obstruction');break
                if actions and (not pieces or not np.array_equal(way[-1],body.pos)) and used>EPS:break
                elapsed+=used
                if joined is not None:
                    at,face,load=joined
                    pieces[-1].update(crown_joint=dict(incoming_piece=j,receiving_piece=k,face=face,normal_load=load))
                    note('miter_supported',at.pos,receiving_piece=k,normal_load=load,velocity=at.vel)
                    j=k; continue
                if event in ('pressure','open','twin'):
                    note(event,body.pos)
                    source=replace(c,j=j,body=body,triangle=face,rebound=None)
                    # A finite time under this input is a degenerate inverse
                    # interval, not an invitation to choose a release station.
                    s=float((body.pos-lane.axis[j])@ (sign*t))
                    yield Departure(body,elapsed,'crown_'+event,source,tuple(pieces),tuple(way),push,intervals=(s,s))
                    break
                if event!='span':break
                if used<n-EPS:
                    at=body;continue
                if stopped:note('axial_stop',body.pos);break
                if not 0<=k<len(lane.length) or lane.tangent[k]@lane.tangent[j]<1.-EPS:
                    if receiving is None:
                        note('finite_end',body.pos)
                    else:
                        q=sc.unit(body.pos-finish)
                        if q@receiving<-EPS and q@t>=-EPS:
                            load=railcrown.exposed_load(q,body.vel,f,r)
                            note('noncollinear_joint',body.pos,exposed_cap_load=load,
                                 cap_status='nonpositive_load' if load<=EPS else 'cap_motion_unproved')
                        else:note('noncollinear_joint',body.pos)
                    source=replace(c,j=j,body=body,triangle=face,rebound=None)
                    yield Departure(body,elapsed,'crown_end',source,tuple(pieces),tuple(way),push)
                    break
                at=body;j=k

    def walk(self, tf, ball, c, controls=None, clock=0., *, search_side=True, require_groove_stability=False):
        """Advance between analytic key changes and geometric rail events."""
        lane=self.lanes[c.lane]; j,sign=c.j,c.direction; b=c.body
        if lane.axis is not None:
            yield from self.walk_crown(tf,ball,c,controls,clock)
            return
        parts=[]; way=[b.pos]; clock0=clock; elapsed=0.; count=0
        while 0<=j<len(lane.length) and count<sc.MAX_SUPPORT_PIECES:
            count+=1; t=sign*lane.tangent[j]; up=support_normal(t)
            initial=b; speed=float(b.vel@t)
            if speed < -EPS or (speed<=EPS and not require_groove_stability):return
            speed=max(0.,speed)
            k=j+sign
            p=lane.p[k if sign>0 else k+1]
            avail=max(0.,float((p-b.pos)@t)); push=sc.keys_along(t)
            limit=math.inf
            if controls is not None:
                limit,push=controls(clock0+elapsed)[0]
                if limit<=EPS:return
            if lane.double and require_groove_stability:
                cross=sc.unit(np.cross(up,t))
                if any(crest(ball,lane,up,sign*cross,max(0.,sign*c.lateral),push) is not None for sign in (-1,1)):
                    return  # a possible side release is not a same-groove witness
            law=sc.law(ball,up,push).along(t)
            end=law.over(speed,avail)
            if end is None:return
            dur=end[1]/ph.DT; move_tick=min(limit,dur)
            distance,end_speed=law.at(speed,move_tick) if limit<dur else (avail,end[0]); dest=b.pos+distance*t
            owner=self.obs.mesh.owner[c.triangle]; name=self.surface.name(c.triangle)
            if not self.surface.clear(tf,ball,[b.pos,dest],owner,up):return
            if search_side:
                yield from self.forward_joints(tf,ball,replace(c,j=j,body=b),law,push,move_tick,parts,way,elapsed,
                                               recorded=controls is not None)
            if lane.double and search_side and self.geometry_support(b.pos,ball,True,t)>=0:
                yield from self.side_departures(tf,ball,Catch(c.lane,j,sign,b,c.lateral,c.triangle),b.pos,speed,
                    law,t,up,push,move_tick,avail,parts,way,elapsed,controls=controls,clock=clock0)
            end_state=ph.State(dest,end_speed*t,ph.rolling_spin(end_speed*t,up,ball.radius))
            elapsed+=move_tick; b=end_state
            if move_tick>EPS:
                parts.append(sc.segment('rail',name,initial,b,move_tick,push,'key_change' if limit<dur else 'track_piece'))
                way.append(b.pos)
            if limit<dur-EPS:continue
            if not 0<=k<len(lane.length):
                source=Catch(c.lane,j,sign,b,c.lateral,c.triangle)
                yield Departure(b,elapsed,'chain_end',source,tuple(parts),tuple(way),push)
                return
            t1=sign*lane.tangent[k]; n1=support_normal(t1)
            curve=self.surface.rail_curvature(b.pos,t,t1,ball.radius+lane.tube+sc.SKIN)
            acceleration=(cf.air_impulse(ball,push=push)/ph.DT)@up
            if t1@t<=0.:
                # The fitted chain folds back; a compressive bend alone is
                # not lift-off. A real ahead obstruction is resolved in air.
                yield Departure(b,elapsed,'foldback',Catch(c.lane,j,sign,b,c.lateral,c.triangle),tuple(parts),tuple(way),push)
                return
            if t1@up<-EPS and acceleration+end_speed*end_speed*curve>=0.:
                rim=tf.convex_release(ball,b.pos,up,n1,t,b.vel,push,sc.inertia_ratio(ball))
                if rim is not None:
                    rt=rim.seconds/ph.DT
                    yield Departure(rim.start,elapsed+rt,'convex',Catch(c.lane,j,sign,b,c.lateral,c.triangle),
                      (*parts,sc.segment('rim',name,b,rim.start,rt,push,'convex',geometry=(rim.motion(),))),(*way,*rim.way()),push,rim)
                    return
            # Project momentum at a concave kink; it does not force release.
            b=ph.State(b.pos,(b.vel@t1)*t1,ph.rolling_spin((b.vel@t1)*t1,n1,ball.radius)); j=k

    def fixed_aims(self, tf, ball, dep):
        """A physical chain end/kink is a degenerate inverse interval."""
        from dataclasses import replace
        law=inverse.ReleaseLaw(dep.body.pos,dep.body.vel,np.zeros(3),np.zeros(3),np.zeros(3),-1.)
        def aims():
            for v,q,_ in self.continued_interval(tf,ball,law,0.,self.inverse_targets):yield v,q,None
            for target,q,_ in self.joints(tf,ball,dep.source,law,0.):yield target.node,q,(target.lane,target.j)
        for v,q,joint in aims():
            objective=inverse.objective(tf,ball,dep.body,q)
            if objective is None:continue
            push,t=objective
            if np.max(abs(push))>1.+EPS:continue
            yield replace(dep,push=np.clip(push,-1.,1.),intervals=(0.,0.),aimed=v,horizon=t+1.,joint=joint)

    def twin_transfer(self,tf,ball,dep):
        lane=self.lanes[dep.source.lane]; groove=self.lanes[lane.parent]
        p=dep.body.pos; j,q=self.fit(groove,p); t=groove.tangent[j]
        face=self.geometry_support(p,ball,True,t)
        if face<0 or np.linalg.norm(q-p)>FIT:return None
        if self.obs.mesh.owner[face]!=self.obs.mesh.owner[dep.source.triangle]:return None
        v=float(dep.body.vel@t); d=1 if v>0 else -1
        if abs(v)<=EPS:return None
        body=ph.State(q,v*t,ph.rolling_spin(v*t,support_normal(t),ball.radius))
        c=Catch(lane.parent,j,d,body,0.,face)
        node=int(groove.ids[np.argmin(np.linalg.norm(self.g.pos[groove.ids]-q,axis=1))])
        event=dict(object=self.surface.name(face),triangle=face,position=p,normal=support_normal(t),incoming=dep.body.vel,
                   kind='dual_rail',tick=0.,rail_line='groove')
        return dict(reason='transfer',node=node,face=face,ticks=0.,state=body,segments=(),way=(p,q),contacts=(event,),
                    bounces=0,catch=c,clearance=0.,axes=())

    def air(self, tf, ball, dep, controls=None, clock=0., *, diagnostics=False, allow_zero_rail=False, bounce_budget=None):
        if dep.kind=='crown_twin':
            transfer=self.twin_transfer(tf,ball,dep)
            if transfer is not None:return transfer
        launch=dep.body
        if controls is None:
            ticks=dep.horizon or cf.landing_step(ball,launch,min(self.obs.bottom-ball.radius-sc.SKIN,launch.pos[1]-ball.radius))
            if not ticks:return None
            schedule=((float(ticks),dep.push),)
        else:
            schedule=controls(clock+dep.tick)
        if not len(schedule) or schedule[-1][0]<=EPS:return None
        hold=schedule[0][1]
        if dep.kind=='side':
            # Road-aimed side releases are the existing v13 path. Only
            # steel-joint and endpoint/rebound proposals use the tube cap.
            departure=tf.rail_support(self.obs,dep.source.body.pos,ball.radius)
            clear=max(0.,self.obs.rail_clearance(ball,launch,hold,schedule[0][0],departure))
        else:
            departure,clear=self.departure_support(tf,ball,dep,schedule)
        caught=None
        def accept(event):
            nonlocal caught
            if not self.obs.mesh.rail[event['triangle']]:return True
            caught=self.capture(ball,event,-1)
            if caught is not None:return True
            # A non-incoming initial overlap does not prove a rebound or a
            # ride. Let shared search report that event as unresolved.
            return (event['tick']<=EPS or event['incoming']@event['normal']>=-EPS)
        kwargs={'accept_support':accept,'follow_below':True} if allow_zero_rail else {}
        if bounce_budget is not None:kwargs['max_bounces']=bounce_budget
        air=tf.trace_air(self.obs,ball,launch,schedule,rim=dep.rim,departure=departure,clearance=clear,incoming_only=True,**kwargs)
        why,i,dt,hit=air.result
        meta=dict(clearance=clear,axes=departure.axes if departure is not None else ())
        if diagnostics:meta.update(air_parts=air.parts,event=air.result,waived_faces=departure.faces if departure else ())
        if (why!='land' or (dt<=EPS and self.obs.mesh.owner[i]==self.obs.mesh.owner[dep.source.triangle]
                             and not (allow_zero_rail and self.obs.mesh.rail[i]))
                or (dep.kind=='side' and dt<=max(.05,clear+EPS)) or (dep.kind=='joint_crest' and dt<=.05)):
            return dict(reason=why if why!='land' else 'never_airborne',contacts=air.contacts,**meta)
        node=self.landing.near(i,hit)
        omega=None
        segments=[]; way=[launch.pos]; cursor=0.
        for state,push,t in air.parts:
            point=tf._points(state,t,ball.lin_damp*ph.DT,cf.air_impulse(ball,push=push))
            at=cf.flight(ball,state,math.ceil(t),push=push)
            end=ph.State(point,at.vel,at.omega); omega=at.omega
            piece=next((c for c in air.contacts if abs(c['tick']-cursor-t)<=EPS),None)
            segments.append(sc.segment('flight',piece['object'] if piece else '',state,end,t,push,piece['kind'] if piece else 'key_change',geometry=(tf.air_geometry(ball,state,push,t),)))
            _, points=tf._samples(state,cf.air_impulse(ball,push=push),ball.lin_damp*ph.DT,t,1.)
            way.extend(points); cursor+=t
        last=air.contacts[-1]; c=(caught if allow_zero_rail else self.capture(ball,last,node,omega)) if self.obs.mesh.rail[i] else None
        if c is not None:last.update(rail_line=self.line_for(c.lane),rail_capture=c.crown)
        if node<0 and c is not None:
            ids=self.lanes[c.lane].ids
            node=int(ids[np.argmin(np.linalg.norm(self.g.pos[ids]-c.body.pos,axis=1))])
        if allow_zero_rail and self.obs.mesh.rail[i] and c is None:
            return dict(reason='rail_capture_unresolved',contacts=air.contacts,**meta)
        if node<0:
            return dict(reason='no_landing_node',contacts=air.contacts,**meta)
        return dict(reason='land',node=node,face=i,ticks=dt,push=hold,state=ph.State(hit,last['incoming'],last.get('incoming_omega',omega)),
                    segments=tuple(segments),way=(*way,hit),contacts=air.contacts,bounces=air.bounces,catch=c,**meta)

    def search(self, tf, ball, c, *, controls=None, clock=0., stats=None, confirmed=None):
        """Check representatives of solved exits; expand earliest arrival per next lane.

        All *mapped* exits are yielded before narrowing just the chain search.
        This limits repeated rail-to-rail expansions, not graph destinations.
        """
        owner=self.obs.mesh.owner
        confirmed=set() if confirmed is None else confirmed
        rest=self.g.info.get('rest_places',0)
        def goal(node):
            if self.labels is None or not 0<=node<rest:return None
            label=int(self.labels[node]);return label if label>=0 else None
        q=deque([(c,0.,0.,(),(),(),(),0,frozenset({(c.lane,c.direction)}),frozenset({int(owner[c.triangle])}))])
        frontier=ArrivalFrontier()
        while q:
            item,elapsed,onrail,path,way,contacts,exits,bounces,visited,objects=q.popleft()
            # Equal kinematics with equal control/budget history may use the
            # earlier arrival. Different coordinates or crown angles cannot
            # be covered by a speed interval alone.
            rb=item.rebound
            last=exits[-1] if exits else None
            itinerary=tuple(dict.fromkeys((x['object'],x['rail_line']) for x in exits))
            key=(item.lane,item.j,item.direction,item.triangle,item.body.pos.tobytes(),item.body.omega.tobytes(),
                 item.lateral,None if rb is None else (rb.pos.tobytes(),rb.vel.tobytes(),rb.omega.tobytes()),
                 None if last is None else (last['lane'],tuple(last['push'])),itinerary,
                 None if controls is None else clock+elapsed)
            if not frontier.accept(key,elapsed,item.body.vel,item.body.vel,visited,objects):
                if stats is not None:stats['rail_arrival_dominated']+=1
                continue
            if stats is not None:stats['rail_arrival_expanded']+=1
            onward={}; covered=set()
            raw=self.walk(tf,ball,item,controls,clock+elapsed)
            def reps():
                if item.rebound is not None:
                    previous=exits[-1] if exits else None
                    same_hop=previous is not None and previous['lane']==item.lane
                    # ASSUMED graph representative: no new key switch at
                    # every repeat collision of the same short hop sequence.
                    hold=previous['push'] if same_hop and controls is None else sc.keys_along(item.body.vel)
                    bounce=Departure(item.rebound,0.,'rail_rebound',item,(),(),hold)
                    yield bounce
                    if item.crown is not None and controls is None and not same_hop:
                        for x in (-1.,0.,1.):
                            for z in (-1.,0.,1.):
                                if (x,z)!=tuple(hold):yield replace(bounce,push=np.array([x,z]))
                    if controls is None and not same_hop:
                        yield from self.fixed_aims(tf,ball,bounce)
                for dep in raw:
                    yield dep
                    if dep.kind not in ('side','joint_crest','joint_tangent','crown_twin','rail_rebound') and controls is None:
                        yield from self.fixed_aims(tf,ball,dep)
            for dep in reps():
                mark=(dep.aimed,dep.intervals,dep.source.j)
                label=goal(dep.aimed) if dep.joint is None else None
                if label is not None and label in confirmed:
                    if stats is not None:stats['rail_confirmed_aim_skipped']+=1
                    continue
                if dep.fallback and mark in covered:continue
                a=self.air(tf,ball,dep,controls,clock+elapsed)
                if a is None:continue
                why=a['reason']
                if stats is not None:stats['rail_candidate_'+why]+=1
                if why not in ('land','transfer'):continue
                status=dict(mode=dep.kind,object=self.surface.name(dep.source.triangle),position=dep.body.pos,
                            velocity=dep.body.vel,tick=elapsed+dep.tick,push=dep.push,interval=dep.intervals,target_node=dep.aimed,
                            joint=dep.joint,source_piece=dep.source.j,clearance=a['clearance'],lane=dep.source.lane,rail_line=self.line_for(dep.source.lane),
                            capture=dep.source.crown)
                p=(*path,*dep.pieces,*a['segments']); w=(*way,*dep.way,*a['way']); after=elapsed+dep.tick+a['ticks'];
                riding=onrail+dep.tick
                hits=(*contacts,*(dict(x,tick=x['tick']+elapsed+dep.tick) for x in a['contacts']))
                e=(*exits,status); nb=bounces+a['bounces']; n=a.get('catch')
                if why=='land':
                    if dep.aimed>=0 and (a['node']==dep.aimed or self.labels is not None and self.labels[a['node']]==self.labels[dep.aimed]):covered.add(mark)
                    # Do not return an immediate re-landing on this same
                    # fitted lane as a new destination. Other lanes count.
                    if n is None or n.lane!=item.lane:
                        label=goal(a['node']) if not self.obs.mesh.rail[a['face']] else None
                        if label is not None:confirmed.add(label)
                        yield dict(node=a['node'],face=a['face'],state=a['state'],ticks=after,ride_ticks=riding,
                                   segments=p,way=w,contacts=hits,departures=e,bounces=nb)
                if n is None:continue
                identities=objects|{int(owner[n.triangle])}
                if len(identities)>MAX_CAPTURES:continue
                key=(n.lane,n.direction)
                token=key
                if key in visited:
                    # A witnessed hop may land farther up the *same* physical
                    # rail. Keep one such recontact event; it is not a route
                    # to the same rolling component and cannot loop forever.
                    if dep.kind not in ('rail_rebound','crown_pressure','crown_open','crown_end') or n.lane!=item.lane or n.direction!=item.direction:continue
                    advance=(n.body.pos-item.body.pos)@sc.unit(item.body.vel)
                    repeat=sum(1 for k in visited if k[:2]==key)-1
                    if advance<=EPS or repeat >= (MAX_CAPTURES-1 if item.crown is not None else 1):continue
                    token=(*key,'recontact',repeat)
                future=(n,after,riding,p,w,hits,e,nb,visited|{token},identities)
                # Different crown angles/velocities are not dominated by
                # earliest arrival to a tube. Collapse only identical states.
                state_key=token if n.crown is None else (token,n.j,n.body.pos.tobytes(),n.body.vel.tobytes())
                if state_key not in onward or after<onward[state_key][1]:onward[state_key]=future
            q.extend(onward.values())
