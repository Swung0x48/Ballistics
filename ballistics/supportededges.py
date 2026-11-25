"""Supported, literal-key transfers between finite conformal road spans."""
from collections import Counter
from itertools import product
import math

import numpy as np
from shapely import LineString

from . import conformal, conformalsearch as cs, motion, physics as ph, roadway, supportcarry as sc, terrain

EPS=1.e-7
XZ=[0,2]
KEYS=tuple(np.asarray(k,dtype=float) for k in product((-1,0,1),repeat=2) if any(k))


def _overlap(topology, i, j, tol):
    """Affine correspondence at equal coordinates along the source span.

    Clip the *common* finite interval against the 3D export band. The gap
    vector is affine there, so its squared norm gives one quadratic interval.
    A tiny wedge at a crossing is not a T seam: it need not support a ball.
    """
    k,f,a,b=topology.edges[i]; j,h,x,y=topology.edges[j]
    ray=(b-a)[XZ]; length=np.linalg.norm(ray); t=ray/length
    u=(y-x)[XZ]; longitudinal=float(u@t)
    if k==j or longitudinal>=-EPS:return None
    # Export noise may turn a span, but a transverse boundary intersection
    # is not a shared bearing line. Use the measured plane-normal tolerance.
    if -longitudinal < np.linalg.norm(u)*conformal.ANGLE:return None
    x0=float((x-a)[XZ]@t); x1=x0+longitudinal
    lo,hi=max(x1,0.),min(x0,length)
    if hi-lo<=EPS:return None
    p,q=topology.surface.patches[k],topology.surface.patches[j]
    def at(s):
        left=a[XZ]+s*t; right=x[XZ]+((s-x0)/longitudinal)*u
        return left,right,np.r_[right-left,q.height(right)-p.height(left)]
    d0=at(lo)[2]; d1=at(hi)[2]; delta=d1-d0
    from . import sweep
    l,h1=sweep.sphere_span(d0,delta,tol,0.,1.)
    if (h1-l)*(hi-lo)<=EPS:return None
    mid=lo+(l+h1)*.5*(hi-lo); left,right,_=at(mid)
    ends=np.array([at(lo+z*(hi-lo))[0] for z in (l,h1)])
    return (k,f,a,b,left),(j,h,x,y,right),ends


def overlaps(topology):
    """Finite opposite spans, including T junctions and export-size seams."""
    tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    for i,line in enumerate(topology.lines):
        for j in topology.tree.query(line.buffer(tol,quad_segs=1)):
            if j<=i:continue
            row=_overlap(topology,i,j,tol)
            if row is not None:yield row


def _xy_depth(p,at,d):
    width=np.linalg.norm((p.high-p.low)[XZ]); d=sc.unit(d)
    line=LineString([at,at-d*(width+conformal.PLANE_ERROR)])
    cut=line.intersection(p.polygon.buffer(EPS,quad_segs=1))
    from .sharedsearch import _lines
    dist=0.
    for item in _lines(cut):
        lo,hi=sorted(float((at-np.asarray(x))@d) for x in (item.coords[0],item.coords[-1]))
        if lo<=2*EPS:dist=max(dist,hi)
    return dist


def _convex_clear(tf,obs,ball,curve,span,bearing):
    """Analytic conservative clearance without feature roots at intended support.

    The arc is enclosed by the axis chord at its middle radial vector
    expanded by 2 r sin(angle/4). The two bearing patches have a separate
    dot-product certificate, valid for every point of their raw triangles.
    An overlapping enclosure is unknown, never a collision verdict.
    """
    a,b=span; t=sc.unit(b-a); r=ball.radius+sc.SKIN
    angle=curve.arc[1]; axial=curve.arc[2]*curve.axial(curve.end)
    along=np.array([(curve.p-a)@t,(curve.p+axial-a)@t]); length=np.linalg.norm(b-a)
    if np.min(along)<-EPS or np.max(along)>length+EPS:return 'outside_support'
    hull=obs.tris[list(bearing)].reshape(-1,3)-curve.p
    u=hull@(curve.v/r); v=hull@(curve.b/r)
    theta=np.arctan2(v,u); support=np.maximum(u,u*math.cos(angle)+v*math.sin(angle))
    support=np.where((theta>=0.)&(theta<=angle),np.hypot(u,v),support)
    # d^2 = r^2 + |q_perp|^2 - 2 r n(theta).q_perp + axial^2.
    # This only waives the named bearing faces, within the support band.
    min_clear=ball.radius-sc.RECAPTURE
    if np.max(support,initial=-math.inf)>(r*r-min_clear*min_clear)/(2*r):return 'bearing_unproved'
    mid=curve.p+curve.v*math.cos(angle*.5)+curve.b*math.sin(angle*.5)
    margin=2*r*math.sin(angle*.25)
    if cs._capsule_blocked(tf,obs,mid,mid+axial,r+conformal.PLANE_ERROR+margin,bearing):return 'clearance_unproved'
    if any(motion.box_hit(curve,x,y) is not None for x,y in obs.boxes):return 'death_box'
    return 'clear'


def prove(tf,obs,surface,ball,source,target,bearing_span=None):
    k,f,a,b,xy=source; j,h,x,y,qxy=target
    p,q=surface.patches[k],surface.patches[j]; n,new=p.normal,q.normal; r=ball.radius+sc.SKIN
    if not cs._rest(ball,p):return None,'source_requires_entry_velocity'
    t=sc.unit(b-a); outward=sc.unit(np.cross(t,n)); inward=-sc.unit(np.cross(y-x,new))
    u0=np.array([xy[0],p.height(xy),xy[1]]); u1=np.array([qxy[0],q.height(qxy),qxy[1]])
    tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    if np.linalg.norm(u1-u0)>tol+EPS:return None,'support_gap'
    # Normals estimated independently on nearly flat exported charts can
    # tilt either way. If both offset centres still agree within the same
    # contact band, use the flat micro-seam rather than extrapolating the
    # intersection of two nearly parallel offset planes a long way off.
    flat=np.linalg.norm(u1+r*new-u0-r*n)<=tol+EPS
    convex=not flat and float(new@outward)>EPS
    best='no_literal_crossing_key'
    choices=[]
    for key in KEYS:
        law=sc.law(ball,n,key); future=sc.law(ball,new,key)
        if law.impulse@outward>EPS:
            choices.append((float(law.impulse@outward),key,law,future))
    for _,key,law,future in sorted(choices,key=lambda x:-x[0]):
        d=sc.unit(law.impulse); scale=np.linalg.norm(d[XZ]); depth=_xy_depth(p,xy,d[XZ]);
        if depth<=4*EPS:best='no_runway';continue
        source_end=u0+r*n; shift=0.; concave=not flat and float(new@outward)<-EPS
        if concave:
            # The new face intercepts the sphere *before* its old foot
            # reaches the mesh lip. For an affine runway ray, intersect the
            # two radius-offset support planes. Both feet must be on the
            # finite charts; extending either plane past its polygon would
            # silently skip the inner rim/vertex.
            den=float(d@new)
            if den>=-EPS:best='concave_no_forward_tangency';continue
            shift=(q.offset+r-source_end@new)/den
            if shift>EPS or depth+shift*scale<=4*EPS:
                best='concave_tangency_outside_runway';continue
            source_end=source_end+shift*d
            from shapely import Point
            if not p.polygon.buffer(EPS).covers(Point((source_end-r*n)[XZ])) or not q.polygon.buffer(EPS).covers(Point((source_end-r*new)[XZ])):
                best='concave_tangency_outside_patch';continue
        # Shorten the finite approach at the exact intercept; a radius-long
        # local representative is not the upper bound on a possible run-up.
        distance=min((depth+shift*scale)*.5,ball.radius)/scale
        start=ph.State(source_end-distance*d,np.zeros(3)); end=law.along(d).over(0.,distance)
        if end is None:continue
        speed,seconds=end; ticks=seconds/ph.DT; at=ph.State(source_end,speed*d); pieces=[motion.rolling(start,law.impulse,law.loss,ticks)]
        if convex:
            turn=sc.unit(new-n*(new@n)); angle=math.acos(float(np.clip(new@n,-1.,1.)))
            rim=tf.rim_release(ball,u0,n,turn,at.vel,0,push=key,inertia_ratio=sc.inertia_ratio(ball))
            if rim is None or rim.angle<angle-EPS:best='convex_release';continue
            curve=motion.arc(u0,r*n,r*turn,0.,angle,rim.along*rim.axis,rim.energy)
            v,c,e=rim.energy; across=math.sqrt(max(0.,v+c*(math.cos(angle)-1.)+e*math.sin(angle)))
            at=ph.State(curve.point(1.),rim.along*rim.axis+across*(turn*math.cos(angle)-n*math.sin(angle)))
            foot=at.pos-r*new
            # Do not transport a convex run to an arbitrary seam midpoint.
            u=foot[XZ]-qxy; side=np.array([inward[0],inward[2]])
            if abs(u@side)>tol or np.linalg.norm(at.pos-new*(at.pos@new-q.offset-r)-at.pos)>tol:
                best='convex_support_offset';continue
            endpos=at.pos-new*(at.pos@new-q.offset-r); pieces.append(curve);ticks+=curve.duration(1.)/ph.DT
        else:
            endpos=source_end if concave else u1+r*new
        if convex:
            verdict=_convex_clear(tf,obs,ball,curve,(a,b) if bearing_span is None else bearing_span,set(p.faces)|set(q.faces))
            if verdict!='clear':best='convex_'+verdict;continue
        radius=ball.radius-sc.RECAPTURE
        # From rest under one constant plane input the swept spatial path is
        # exactly a straight segment, regardless of the time parametrization.
        # Solve sphere/feature quadratics on that segment, not air PSI roots.
        if cs._capsule_blocked(tf,obs,start.pos,source_end,radius,set(p.faces)) or any(
                motion.box_hit(motion.line(start.pos,source_end),lo,hi) is not None for lo,hi in obs.boxes):
            best='source_obstruction';continue
        if cs._capsule_blocked(tf,obs,at.pos,endpos,radius,set(p.faces)|set(q.faces)) or any(motion.box_hit(motion.line(at.pos,endpos),lo,hi) is not None for lo,hi in obs.boxes):
            best='seam_obstruction';continue
        endvel=at.vel-new*(at.vel@new)
        if endvel@inward<=EPS and future.impulse@inward<=EPS:
            best='cannot_enter_next_patch';continue
        # The first receiving PSI is an exact chord (position first,
        # then update velocity). Clip that whole chord against the polygon
        # and the mesh. This is one physical leg, not a step search.
        dp,v=future.at(endvel,1.); final=ph.State(endpos+dp,v)
        foot0=(endpos-r*new)[XZ]; foot1=(final.pos-r*new)[XZ]
        if dp@inward<=EPS or not q.polygon.buffer(EPS,quad_segs=1).covers(LineString([foot0,foot1])):
            best='next_patch_obstruction_or_exit';continue
        leg=motion.line(endpos,final.pos)
        if cs._capsule_blocked(tf,obs,endpos,final.pos,radius,set(q.faces)) or any(motion.box_hit(leg,lo,hi) is not None for lo,hi in obs.boxes):
            best='next_patch_obstruction_or_exit';continue
        pieces.extend((motion.line(at.pos,endpos),leg))
        return dict(kind='roll',source_patch=k,target_patch=j,origin=start.pos,position=final.pos,
                    origin_velocity=start.vel,end_velocity=final.vel,ticks=ticks+1.,key_table=((math.ceil(ticks+1.),key.copy()),),
                    motion=tuple(pieces),mode='convex_supported' if convex else 'concave_or_flat'), 'literal_roll'
    return None,best


def step_proposal(tf, paths, ball, lip):
    """One finite v15 lip witness; a failed proposal is not a denial."""
    surface=paths.g.info['conformal']; obs=paths.obs
    p,q=(surface.patches[k] for k in (lip.lower,lip.upper))
    pivot=(lip.a+lip.b)*.5; height=lip.height(surface,pivot)
    r=ball.radius+sc.SKIN; inertia=sc.inertia_ratio(ball)
    f=lip.forward; d=np.array([f[0],0.,f[1]])
    info=dict(height=float(height),required_static_ratio=terrain.torque_ratio(height,ball.radius),
              available_static_ratio=terrain.ratio(ball)*float(np.abs(f).sum()),
              required_entry_speed=terrain.step_entry_speed(ball,height,f,inertia),entry_speed=None)
    if not 0.<height<ball.radius:return None,'step_outside_pivot_model',info
    if min(p.normal[1],q.normal[1])<1.-EPS:return None,'step_nonhorizontal_charts',info
    # The axis-aligned literal branch is exact for this pivot model. An
    # oblique approach needs a separate slip/impact calculation.
    choices=[key for key in KEYS if key@f>EPS and abs(key@np.array([-f[1],f[0]]))<=EPS]
    if not choices:return None,'step_oblique_keys_unproved',info
    key=max(choices,key=lambda v:v@f); law=sc.law(ball,p.normal,key)
    at=pivot+r*sc.UP-height*sc.UP-math.sqrt(height*(2*r-height))*d
    depth=_xy_depth(p,at[XZ],f)
    if depth<=EPS:return None,'step_no_aligned_runway',info
    # All arc pressure and Coulomb margins decrease affinely in squared
    # captured speed. V15's zero-speed angular extrema give the upper
    # bound in closed form, while the work barrier gives the lower bound.
    force=terrain.ratio(ball)*ph.GRAVITY*float(key@f)
    _,barrier,_=terrain.step_work(height,r,force)
    capture=1.-height/(r*(1.+inertia))
    required=math.sqrt(2*max(0.,barrier)/(1.+inertia))/capture
    info['required_entry_speed']=required
    material=obs.mesh.entities[lip.owner] if lip.owner>=0 else {}
    mu=ball.friction*float(material.get('friction',.7))
    if mu<=0.:return None,'step_no_friction_certificate',info
    _,n0,t0=terrain.step_contact_limits(height,r,force,inertia,0.,mu)
    theta=math.acos(1.-height/r)
    capture_margin=mu-inertia/(1.+inertia)*(1.-math.cos(theta))/math.sin(theta)
    upper=math.sqrt(r*max(0.,min(n0,t0/mu)))/capture
    info.update(max_contact_speed=upper,impact_friction_margin=capture_margin)
    if capture_margin<0. or upper<=required+EPS:return None,'step_contact_interval_empty',info
    axis=law.along(d); crossing=axis.over(0.,depth*.5)
    info['runway_entry_speed']=crossing[0] if crossing else 0.
    if crossing is None or crossing[0]<=required+EPS:return None,'step_needs_momentum_entry',info
    speed=(required+min(crossing[0],upper))*.5
    tick=math.log1p(-speed/axis.terminal_speed)/math.log1p(-axis.loss)
    length=axis.at(0.,tick)[0]; start=ph.State(at-length*d,np.zeros(3))
    body=ph.State(at,speed*d,ph.rolling_spin(speed*d,sc.UP,ball.radius))
    info['entry_speed']=speed
    if cs._capsule_blocked(tf,obs,start.pos,at,ball.radius-sc.RECAPTURE,set(p.faces)) or any(
            motion.box_hit(motion.line(start.pos,at),lo,hi) is not None for lo,hi in obs.boxes):
        return None,'step_runway_obstruction',info
    result=paths.climb_lip(tf,ball,lip,pivot,body,key,math.inf)
    if result is None:return None,'step_arc_unproved',info
    top,t,arc,code,limits=result; info['contact_margins']=limits
    if code!='ideal_step_roll':return None,code,info
    # Require a whole receiving chord, not just arrival at the crown.
    dp,vel=sc.law(ball,q.normal,key).at(top.vel,1.); pos=top.pos+dp
    if not q.polygon.buffer(EPS).covers(LineString([top.pos[XZ],pos[XZ]])):
        return None,'step_receiving_chord_outside',info
    leg=motion.line(top.pos,pos)
    if cs._capsule_blocked(tf,obs,top.pos,pos,ball.radius-sc.RECAPTURE,set(q.faces)) or any(
            motion.box_hit(leg,lo,hi) is not None for lo,hi in obs.boxes):return None,'step_receiving_chord_blocked',info
    ticks=tick+t+1.
    return dict(kind='roll',source_patch=lip.lower,target_patch=lip.upper,origin=start.pos,position=pos,
                origin_velocity=start.vel,end_velocity=vel,ticks=ticks,key_table=((math.ceil(ticks),key.copy()),),
                motion=(motion.rolling(start,law.impulse,law.loss,tick),arc,leg),mode='ideal_step_roll'),'ideal_step_roll',info


def slope_requirement(ball,patch,forward,distance):
    """Minimum aligned inlet speed to reach a point before the uphill turn.

    With the best literal world key, distance to the turn is monotonically
    increasing with inlet speed since every position coefficient is positive.
    This is a longitudinal bound, not a proof of lateral clearance.
    """
    laws=[sc.law(ball,patch.normal,key).along(forward) for key in KEYS]
    axis=max(laws,key=lambda law:law.impulse)
    if distance<=0. or axis.impulse>0.:return 0.
    if axis.impulse==0.:return distance*axis.loss/ph.DT
    from . import closedform as cf
    from scipy.optimize import brentq
    def reach(v):return axis.at(v,cf._turn(v,axis.impulse,axis.loss))[0]
    hi=math.sqrt(2*max(0.,distance)*ph.GRAVITY)+abs(axis.terminal_speed)
    while reach(hi)<distance:hi*=2.
    return brentq(lambda v:reach(v)-distance,0.,hi)


def build(tf,g,obs,layout,pick,*,log=lambda *_:None):
    from types import SimpleNamespace
    surface=g.info['conformal']; verified={}; pending=set(); stats=Counter(); audit=[]
    proofs={}
    def publish(a,b,row):
        if a==b or (a,b) in verified:return
        verified[a,b]=row;pending.discard((a,b))
        i,j=pick(row['source_patch'],row['origin']),pick(row['target_patch'],row['position'])
        if i<0 or j<0 or i==j:return
        how='roll'; bit=1<<tf.KIND_INDEX[a.ball]; masks=g.info.setdefault('terrain_ball_masks',{})
        group=(i,j,how); previous=any(e[0]==j and e[2]==how for e in g.adj[i]);masks[group]=masks.get(group,7 if previous else 0)|bit
        if not previous:g.adj[i].append((j,max(EPS,float(np.linalg.norm(row['position']-row['origin']))),how))
        row['mapped']=(i,j)
    def record(a,b,kind,pair,segment,row,reason,**extra):
        rec=dict(source=a,target=b,kind=kind,patch_pair=pair,segment=segment,
                 status='verified' if row is not None else 'unresolved',reason=reason,**extra)
        if row is not None:rec['witness']=row;publish(a,b,row)
        elif a!=b and (a,b) not in verified:pending.add((a,b))
        audit.append(rec);stats[reason]+=1
        return rec
    for L,R,span in overlaps(roadway.Topology(surface)):
        stats['spans']+=1
        if stats['spans']%1000==0:log('shared rolling progress: %d spans, %d checked pairs, %d unresolved pairs' % (stats['spans'],len(verified),len(pending)))
        for ball in ph.BALLS.values():
            index=layout.patch_nodes[ball.name]
            for u,v in ((L,R),(R,L)):
                a,b=index[u[0]],index[v[0]]
                if a==b:continue
                pair=(u[0],v[0]); p=surface.patches[u[0]]
                ends=np.column_stack((span[:,0],p.height(span),span[:,1]))
                row,reason=prove(tf,obs,surface,ball,u,v,bearing_span=ends)
                guarded=any(k in layout.roads[node].guarded_patches for node,k in zip((a,b),pair))
                rec=record(a,b,'finite_seam',pair,ends,row,reason,mechanism_screen_required=guarded)
                proofs.setdefault((a.ball,pair),[]).append(rec)
    lips={(lip.lower,lip.upper,conformal.edge_key(lip.a,lip.b)):lip
          for group in g.info.get('terrain_lips',{}).values() for lip in group}
    paths=SimpleNamespace(g=g,obs=obs,climb_lip=lambda *a:sc.SurfacePaths.climb_lip(paths,*a))
    balls={b.name:b for b in ph.BALLS.values()}
    for i,portal in enumerate(layout.portals):
        a,b=portal.source,portal.target; pair=portal.patch_pair; seg=portal.segment
        if portal.kind=='rail_ring':
            record(a,b,portal.kind,pair,seg,None,'rail_ring_requires_supported_inlet',portal_index=i);continue
        if portal.kind=='step_lip':
            lip=lips.get((*pair,conformal.edge_key(*seg)))
            if lip is None:row,reason,detail=None,'step_geometry_missing',{}
            elif any(k in layout.roads[node].guarded_patches for node,k in zip((a,b),pair)):
                row,reason,detail=None,'mechanism_guard_unproved',{}
            else:row,reason,detail=step_proposal(tf,paths,balls[a.ball],lip)
            record(a,b,portal.kind,pair,seg,row,reason,portal_index=i,detail=detail);continue
        # A proof on another span between the same patches is insufficient.
        span=LineString(seg[:,XZ]).buffer(EPS)
        matches=(r for r in proofs.get((a.ball,pair),()) if span.covers(LineString(r['segment'][:,XZ])))
        r=min(matches,key=lambda r:r['status']!='verified',default=None)
        guarded=any(k in layout.roads[node].guarded_patches for node,k in zip((a,b),pair))
        record(a,b,portal.kind,pair,seg,None if r is None else r.get('witness'),
               'no_positive_length_support_in_export_band' if r is None else r['reason'],portal_index=i,mechanism_screen_required=guarded)
    stats['pairs']=len(verified);stats['unresolved']=len(pending)
    g.info['portal_verdicts']=audit
    return verified,pending,dict(stats)
