"""Supported rail corridors and closed-form departure opportunities."""
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.spatial import cKDTree

from . import closedform as cf, physics as ph, supportcarry as sc, rail_inverse as inverse, motion, rails, railjoint
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


@dataclass
class Catch:
    lane: int
    j: int
    direction: int
    body: ph.State
    lateral: float               # signed in support cross-plane
    triangle: int
    rebound: object = None


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


class RailPaths:
    def __init__(self, g, obs, landing):
        self.g,self.obs,self.landing = g,obs,landing
        self.surface = sc.SurfacePaths(g,obs,landing)
        self.lanes={}
        self.endpoint=[]
        lo,hi,owners=[],[],[]
        for ids in _lane_chains(g):
            if len(ids)<2: continue
            li=int(g.lane[ids[0]]); m=g.lanes[li]; p=self.surface.rail_geometry[li]
            if len(p)<2:continue
            delta=np.diff(p,axis=0)
            L=np.linalg.norm(delta,axis=1)
            if np.any(L<=EPS): continue
            self.lanes[li]=Lane(ids,p,L,delta/L[:,None],m['double'],m['radius'],m['gauge'])
            self.endpoint.extend(((li,1,p[0]),(li,-1,p[-1])))
            lo.extend(np.minimum(p[:-1],p[1:])); hi.extend(np.maximum(p[:-1],p[1:]))
            owners.extend([li]*len(L))
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

    def capture(self, ball, hit, node, omega=None):
        """Inelastic normal constraint for the *additional* rail-ride path."""
        gap=np.maximum(0.,np.maximum(self.piece_lo-hit['position'],hit['position']-self.piece_hi))
        possible=np.unique(self.piece_lanes[np.einsum('ij,ij->i',gap,gap)<=(ball.radius+FIT)**2])
        n=hit.get('support_normal',hit['normal'])
        choices=[]; owner=self.obs.mesh.owner[hit['triangle']]
        for li in sorted(possible):
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
            bounced=cf.rebound(ball,incoming,reaction,float(material.get('elasticity',.3)))
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
        for i,d,_,_ in self.candidates_at_end(ball,at):
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
        axes,used=railjoint.bind_axes(m.tubes,self.obs.tris[faces],release,r)
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

    def walk(self, tf, ball, c, controls=None, clock=0., *, search_side=True):
        """Advance between analytic key changes and geometric rail events."""
        lane=self.lanes[c.lane]; j,sign=c.j,c.direction; b=c.body
        parts=[]; way=[b.pos]; clock0=clock; elapsed=0.; count=0
        while 0<=j<len(lane.length) and count<sc.MAX_SUPPORT_PIECES:
            count+=1; t=sign*lane.tangent[j]; up=support_normal(t)
            initial=b; speed=float(b.vel@t)
            if speed<=EPS:return
            k=j+sign
            p=lane.p[k if sign>0 else k+1]
            avail=max(0.,float((p-b.pos)@t)); push=sc.keys_along(t)
            limit=math.inf
            if controls is not None:
                limit,push=controls(clock0+elapsed)[0]
                if limit<=EPS:return
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

    def air(self, tf, ball, dep, controls=None, clock=0., *, diagnostics=False):
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
        air=tf.trace_air(self.obs,ball,launch,schedule,rim=dep.rim,departure=departure,clearance=clear,incoming_only=True)
        why,i,dt,hit=air.result
        meta=dict(clearance=clear,axes=departure.axes if departure is not None else ())
        if diagnostics:meta.update(air_parts=air.parts,event=air.result,waived_faces=departure.faces if departure else ())
        if (why!='land' or (dt<=EPS and self.obs.mesh.owner[i]==self.obs.mesh.owner[dep.source.triangle])
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
        last=air.contacts[-1]; c=self.capture(ball,last,node,omega) if self.obs.mesh.rail[i] else None
        if node<0 and c is not None:
            ids=self.lanes[c.lane].ids
            node=int(ids[np.argmin(np.linalg.norm(self.g.pos[ids]-c.body.pos,axis=1))])
        if node<0:
            return dict(reason='no_landing_node',contacts=air.contacts,**meta)
        return dict(reason='land',node=node,face=i,ticks=dt,push=hold,state=ph.State(hit,last['incoming'],omega),
                    segments=tuple(segments),way=(*way,hit),contacts=air.contacts,bounces=air.bounces,catch=c,**meta)

    def search(self, tf, ball, c, *, controls=None, clock=0., stats=None):
        """Check representatives of solved exits; expand earliest arrival per next lane.

        All *mapped* exits are yielded before narrowing just the chain search.
        This limits repeated rail-to-rail expansions, not graph destinations.
        """
        owner=self.obs.mesh.owner
        q=deque([(c,0.,0.,(),(),(),(),0,frozenset({(c.lane,c.direction)}),frozenset({int(owner[c.triangle])}))])
        while q:
            item,elapsed,onrail,path,way,contacts,exits,bounces,visited,objects=q.popleft()
            onward={}; covered=set()
            raw=self.walk(tf,ball,item,controls,clock+elapsed)
            def reps():
                if item.rebound is not None:
                    yield Departure(item.rebound,0.,'rail_rebound',item,(),(),sc.keys_along(item.body.vel))
                for dep in raw:
                    yield dep
                    if dep.kind not in ('side','joint_crest','joint_tangent') and controls is None:
                        yield from self.fixed_aims(tf,ball,dep)
            for dep in reps():
                mark=(dep.aimed,dep.intervals,dep.source.j)
                if dep.fallback and mark in covered:continue
                a=self.air(tf,ball,dep,controls,clock+elapsed)
                if a is None:continue
                why=a['reason']
                if stats is not None:stats['rail_candidate_'+why]+=1
                if why not in ('land','transfer'):continue
                status=dict(mode=dep.kind,object=self.surface.name(dep.source.triangle),position=dep.body.pos,
                            velocity=dep.body.vel,tick=elapsed+dep.tick,push=dep.push,interval=dep.intervals,target_node=dep.aimed,
                            joint=dep.joint,source_piece=dep.source.j,clearance=a['clearance'])
                p=(*path,*dep.pieces,*a['segments']); w=(*way,*dep.way,*a['way']); after=elapsed+dep.tick+a['ticks'];
                riding=onrail+dep.tick
                hits=(*contacts,*(dict(x,tick=x['tick']+elapsed+dep.tick) for x in a['contacts']))
                e=(*exits,status); nb=bounces+a['bounces']; n=a.get('catch')
                if why=='land':
                    if dep.aimed>=0 and (a['node']==dep.aimed or self.labels is not None and self.labels[a['node']]==self.labels[dep.aimed]):covered.add(mark)
                    # Do not return an immediate re-landing on this same
                    # fitted lane as a new destination. Other lanes count.
                    if n is None or n.lane!=item.lane:
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
                    if dep.kind!='rail_rebound' or n.lane!=item.lane or n.direction!=item.direction:continue
                    advance=(n.body.pos-item.body.pos)@sc.unit(item.body.vel)
                    token=(*key,'recontact')
                    if advance<=EPS or token in visited:continue
                future=(n,after,riding,p,w,hits,e,nb,visited|{token},identities)
                if token not in onward or after<onward[token][1]:onward[token]=future
            q.extend(onward.values())
