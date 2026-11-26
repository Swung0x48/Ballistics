"""Event-directed, literal integer control for an unstable rail crown."""
import math

import numpy as np
from scipy.optimize import brentq

from . import closedform as cf, motion, physics as ph, railcrown, supportcarry as sc

EPS=sc.EPS
KEYS=tuple(np.array((x,z),float) for x in (-1,0,1) for z in (-1,0,1))


def correction(ball,clock,theta,v,u,t,up,side,r,inner,avail,hold):
    """Choose an integer near a crossing of the next key's separatrix.

    The current first angular action is monotone. At a candidate switch,
    energy in the next potential equals A+(C-C')cos+Ddiff*sin. Intersect
    this sinusoid with the saddle energy, then check the adjacent integer
    ticks (not a walk through tick positions). A certified correction
    must really turn back before pressure vanishes or a tube is left.
    """
    force=cf.air_impulse(ball,push=hold)/ph.DT
    g,f=-float(force@up),float(force@side)
    k=sc.inertia_ratio(ball)*(ball.radius/r)**2
    a=railcrown.swing(theta,v,g,f,r,k,inner)
    if not a or a[0].event=='turn':return None
    s=a[0]; turn=s.end-theta
    if abs(turn)<=EPS:return None
    sign=1 if turn>0. else -1
    E,C,D=s.energy; A=E-C*math.cos(theta)-D*math.sin(theta)
    path=motion.arc(np.zeros(3),r*up,r*side,theta,turn,np.zeros(3),s.energy)
    duration=path.duration(1.)/ph.DT
    law=sc.law(ball,up,hold).along(t); edge=law.over(u,avail)
    bound=min(duration,edge[1]/ph.DT if edge else math.inf)
    if not math.isfinite(bound) or bound<=EPS:return None
    lo,hi=sorted((theta,s.end)); best=None
    B=2*r/(1+k)
    for key in KEYS:
        if np.array_equal(key,hold):continue
        accel=cf.air_impulse(ball,push=key)/ph.DT
        gg,ff=-float(accel@up),float(accel@side)
        saddle=math.atan2(-ff,gg)
        if gg<=EPS or not inner+EPS<saddle<math.pi/2-EPS:continue
        c,d=-B*gg,B*ff
        roots=railcrown.roots(C-c,D-d,A+c*math.cos(saddle)+d*math.sin(saddle),lo,hi)
        times=[max(0.,math.ceil(clock-EPS)-clock)]
        times.extend(path.duration((x-theta)/turn)/ph.DT for x in roots)
        for tick in sorted({i for dt in times for i in (math.floor(clock+dt),math.ceil(clock+dt))}):
            dt=tick-clock
            if not -EPS<=dt<bound-EPS:continue
            dt=max(0.,dt); dx,speed=law.at(u,dt)
            if not 0.<=dx<avail-EPS or speed<=EPS:continue
            w=0. if dt<=EPS else brentq(lambda x:path.duration(x)/ph.DT-dt,0.,1.)
            angle=theta+w*turn
            lateral=sign*math.sqrt(max(0.,A+C*math.cos(angle)+D*math.sin(angle)))
            next_actions=railcrown.swing(angle,lateral,gg,ff,r,k,inner)
            if not next_actions or next_actions[0].event!='turn':continue
            end=next_actions[0].end
            # The corrective turn should not run past the unforced crown
            # into a less recoverable angle before the next decision.
            next_law=sc.law(ball,up,key).along(t)
            reaches_joint=next_law.over(speed,avail-dx) is not None
            if not reaches_joint and next_law.impulse<0.:
                turn_path=motion.arc(np.zeros(3),r*up,r*side,angle,end-angle,
                                     np.zeros(3),next_actions[0].energy)
                if turn_path.duration(1.)/ph.DT>=cf._turn(speed,next_law.impulse,next_law.loss)-EPS:
                    continue
            score=(not reaches_joint,next_law.impulse<=0.,abs(end),abs(angle),tick,-next_law.impulse)
            if best is None or score<best[0]:
                best=(score,int(tick),key.copy(),dict(angle=angle,lateral_speed=lateral,
                       turn_angle=end,saddle=saddle,lateral_load_margin=gg*math.cos(angle)-ff*math.sin(angle)-lateral*lateral/r))
    return None if best is None else best[1:]


def execution_table(seed,dep,end_key,last_tick):
    """Absolute integer key boundaries, rebuilt from checked pieces."""
    if not seed['key_table']:return ()
    prior=list(seed['key_table']); _,active=prior.pop(); t=float(seed['ticks'])
    for part in (*dep.pieces,dict(push=end_key,ticks=0.)):
        key=np.asarray(part['push'])
        if not np.array_equal(key,active):
            n=round(t)
            if abs(t-n)>5e-6:return None
            if not prior or n>prior[-1][0]:prior.append((n,active.copy()))
            active=key
        t+=part['ticks']
    n=math.ceil(last_tick)
    if not prior or n>prior[-1][0]:prior.append((n,active.copy()))
    return tuple(prior)


def standalone(paths,ball,c,hit):
    """Recover the true tube axis for a single, unpaired rail hit.

    The legacy single lane used a vertical-offset fit and threw away its
    cross speed. Keep it for diagnostics and use a separate cylinder only
    for a checked dynamic run, with the actual support feature.
    """
    from dataclasses import replace
    from . import railride, rails
    lane=paths.lanes[c.lane]
    if lane.axis is not None:return c
    if lane.double or hit is None:return None
    idx=-1-lane.parent
    axis=lane.p-(ph.WOOD.radius+lane.tube)*sc.UP
    alt=replace(lane,axis=axis,twin=None,line='single_crown')
    j,_=rails.project_pieces(axis,hit['position']); center,t,up,side,r,inner=paths.cross_section(alt,j,hit['position'],ball)
    radial=hit['position']-center; radial-=t*(radial@t)
    n=sc.unit(radial)
    if n@up<=EPS or abs(np.linalg.norm(radial)-r)>railride.FIT:return None
    v=hit['incoming']-n*(hit['incoming']@n); u=float(v@t)
    if abs(u)<=EPS:return None
    p=center+r*n
    cols=paths.obs.mesh.contacts(p,ball.radius,sc.RECAPTURE)
    if any(x.distance<ball.radius-sc.RECAPTURE for x in cols):return None
    owner=paths.obs.mesh.owner[hit['triangle']]
    if not any(paths.obs.mesh.rail[x.triangle] and paths.obs.mesh.owner[x.triangle]==owner and x.normal@n>0. for x in cols):return None
    paths.lanes[idx]=alt
    theta=math.atan2(float(n@side),float(n@up))
    lateral=float(v@(side*math.cos(theta)-up*math.sin(theta)))
    constraint=ph.State(p,v,np.asarray(hit.get('incoming_omega',np.zeros(3))))
    return replace(c,lane=idx,j=j,direction=1 if u>0. else -1,body=constraint,lateral=lateral,
                   crown=dict(supported=True,angle=theta,lateral_speed=lateral,lateral_offset=float(radial@side),
                              line=alt.line,contact='one_tube'))


def verify_track(tf,obs,ball,parts):
    """New dynamic witness must not waive every face of its rail owner."""
    from .railstatic import segment_features
    radius=ball.radius-sc.RECAPTURE
    for part in parts:
        if part.get('rail_line') not in ('left_crown','right_crown','single_crown'):continue
        for piece in part['motion']:
            # On a straight run the finite feature test is a quadratic
            # clearance test. For a swing use bounded continuous distance
            # against the original mesh, including same-owner obstacles.
            if piece.kind in ('line','rolling'):
                from .conformal import PLANE_ERROR
                _,hit=segment_features(tf,obs,piece.point(0.),piece.point(piece.end),radius+PLANE_ERROR+EPS)
                if hit:return False
            elif motion.near_hit(piece,lambda p:obs.mesh.distance(p,radius+1.)[0],radius) is not None:
                return False
    return True
