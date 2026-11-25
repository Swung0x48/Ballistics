"""Supported, literal-key transfers between finite conformal road spans."""
from collections import Counter
from itertools import product
import math

import numpy as np
from shapely import LineString

from . import conformal, conformalsearch as cs, motion, physics as ph, roadway, supportcarry as sc

EPS=1.e-7
XZ=[0,2]
KEYS=tuple(np.asarray(k,dtype=float) for k in product((-1,0,1),repeat=2) if any(k))


def overlaps(topology):
    """Finite opposite spans, including T junctions and export-size seams."""
    tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    surface=topology.surface
    for i,(k,f,a,b) in enumerate(topology.edges):
        ray=(b-a)[XZ]; length=np.linalg.norm(ray); t=ray/length
        n=np.array([-t[1],t[0]])
        for index in topology.tree.query(topology.lines[i].buffer(tol,quad_segs=1)):
            if index<=i:continue
            j,h,x,y=topology.edges[index]
            if k==j or (y-x)[XZ]@t>=-EPS:continue
            if max(abs((x-a)[XZ]@n),abs((y-a)[XZ]@n))>tol+EPS:continue
            lo,hi=sorted(((x-a)[XZ]@t,(y-a)[XZ]@t));lo=max(lo,0.);hi=min(hi,length)
            if hi-lo<=EPS:continue
            p=surface.patches[k];q=surface.patches[j]; ends=np.array([a[XZ]+z*t for z in (lo,hi)])
            if np.max(abs(p.height(ends)-q.height(ends)))>tol+EPS:continue
            centre=a[XZ]+(lo+hi)*.5*t
            u=(y-x)[XZ]; frac=float(np.clip((centre-x[XZ])@u/(u@u),0.,1.)); other=x[XZ]+frac*u
            yield (k,f,a,b,centre),(j,h,x,y,other),ends


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


def prove(tf,obs,surface,ball,source,target):
    k,f,a,b,xy=source; j,h,x,y,qxy=target
    p,q=surface.patches[k],surface.patches[j]; n,new=p.normal,q.normal; r=ball.radius+sc.SKIN
    if not cs._rest(ball,p):return None,'no_rest_support'
    t=sc.unit(b-a); outward=sc.unit(np.cross(t,n)); inward=-sc.unit(np.cross(y-x,new))
    u0=np.array([xy[0],p.height(xy),xy[1]]); u1=np.array([qxy[0],q.height(qxy),qxy[1]])
    tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    if np.linalg.norm(u1-u0)>tol+EPS:return None,'support_gap'
    convex=float(new@outward)>EPS
    best='no_crossing_key'
    choices=[]
    for key in KEYS:
        law=sc.law(ball,n,key); future=sc.law(ball,new,key)
        if law.impulse@outward>EPS and future.impulse@inward>EPS:
            choices.append((float(law.impulse@outward),key,law,future))
    for _,key,law,future in sorted(choices,key=lambda x:-x[0]):
        d=sc.unit(law.impulse); scale=np.linalg.norm(d[XZ]); depth=_xy_depth(p,xy,d[XZ]);
        if depth<=4*EPS:best='no_runway';continue
        # One finite representative from the exact inlet interval. A short
        # convex approach deliberately tests remaining in contact, not flight.
        distance=min(depth*.5,ball.radius) / scale
        start=ph.State(u0+r*n-distance*d,np.zeros(3)); end=law.along(d).over(0.,distance)
        if end is None:continue
        speed,seconds=end; ticks=seconds/ph.DT; at=ph.State(u0+r*n,speed*d); pieces=[motion.rolling(start,law.impulse,law.loss,ticks)]
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
            endpos=u1+r*new
        if convex:
            verdict=_convex_clear(tf,obs,ball,curve,(a,b),set(p.faces)|set(q.faces))
            if verdict!='clear':best='convex_'+verdict;continue
        radius=ball.radius-sc.RECAPTURE
        # From rest under one constant plane input the swept spatial path is
        # exactly a straight segment, regardless of the time parametrization.
        # Solve sphere/feature quadratics on that segment, not air PSI roots.
        source_end=u0+r*n
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


def build(tf,g,obs,layout,pick,*,log=lambda *_:None):
    surface=g.info['conformal']; verified={}; pending=set(); stats=Counter()
    for L,R,_ in overlaps(roadway.Topology(surface)):
        stats['spans']+=1
        if stats['spans']%1000==0:log('shared rolling progress: %d spans, %d checked pairs, %d unresolved pairs' % (stats['spans'],len(verified),len(pending)))
        for ball in ph.BALLS.values():
            index=layout.patch_nodes[ball.name]
            for u,v in ((L,R),(R,L)):
                a,b=index[u[0]],index[v[0]]
                if a==b or (a,b) in verified:continue
                row,reason=prove(tf,obs,surface,ball,u,v);stats[reason]+=1
                if row is None:pending.add((a,b));continue
                verified[a,b]=row;pending.discard((a,b))
                i,j=pick(u[0],row['origin']),pick(v[0],row['position'])
                if i<0 or j<0 or i==j:continue
                how='roll'; bit=1<<tf.KIND_INDEX[ball.name]; masks=g.info.setdefault('terrain_ball_masks',{})
                group=(i,j,how); previous=any(e[0]==j and e[2]==how for e in g.adj[i]);masks[group]=masks.get(group,7 if previous else 0)|bit
                if not previous:g.adj[i].append((j,max(EPS,float(np.linalg.norm(row['position']-row['origin']))),how))
                row['mapped']=(i,j)
    stats['pairs']=len(verified);stats['unresolved']=len(pending)
    return verified,pending,dict(stats)
