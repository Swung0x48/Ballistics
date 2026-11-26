"""Finite road/rail bearing joins, independent of the mechanism host grid."""
from collections import defaultdict
import math

import numpy as np
from shapely import LineString, Point, STRtree, box

from . import conformal, conformalsearch as cs, motion, physics as ph
from . import quasistatic, railride, rimclear, supportcarry as sc, sweep

XZ = [0, 2]
EPS = sc.EPS


def bearing_intervals(patch, origin, tangent, length, radius, tol):
    """Metric rail abscissae with a finite source chart and an offset-plane join.

    Projection back to the road plane is affine. Intersect the projected
    segment with the polygon and clip each interval by the signed elevation
    band. There is no rail station resampling.
    """
    n = patch.normal
    e0 = float(origin@n-patch.offset-radius)
    rate = float(tangent@n)
    lo,hi = sweep.linear_clip(tol-e0,-rate,0.,length)
    lo,hi = sweep.linear_clip(tol+e0,rate,lo,hi)
    if hi < lo:
        return ()
    a = (origin-(e0+radius)*n)[XZ]
    d = (tangent-rate*n)[XZ]
    start,end = a+lo*d,a+hi*d
    if np.linalg.norm(end-start)<=EPS:
        return ((lo,hi),) if patch.polygon.covers(Point(start)) else ()
    cut = patch.polygon.intersection(LineString((start,end)))
    out=[]
    def add(part):
        if part.is_empty:
            return
        if part.geom_type in ('Point','LineString'):
            coords=part.coords
            x,y=(float((np.asarray(pt)-a)@d/(d@d)) for pt in (coords[0],coords[-1]))
            l,h=max(lo,min(x,y)),min(hi,max(x,y))
            if h>=l-EPS:out.append((l,max(l,h)))
        elif hasattr(part,'geoms'):
            for sub in part.geoms:add(sub)
    add(cut)
    return tuple(out)


def _approach(patch, finish, direction):
    xy=finish[XZ]; d=direction[XZ]; q=float(d@d)
    if q<=EPS**2:
        return 0.
    far=xy-d*(float(np.linalg.norm(patch.high-patch.low))+1.)/math.sqrt(q)
    cut=patch.polygon.buffer(EPS,quad_segs=1).intersection(LineString((xy,far)))
    parts=cut.geoms if hasattr(cut,'geoms') else (cut,)
    for part in parts:
        if part.is_empty or part.geom_type!='LineString':continue
        a,b=sorted(float((xy-np.asarray(p))@d/q) for p in (part.coords[0],part.coords[-1]))
        if a<=4*EPS and b>EPS:return b
    return 0.


def supported_inlets(tf, paths, surface, layout, chains, balls, stats):
    """Emit checked road-to-rail handoffs from finite offset charts.

    Inspect the end chart up to one sphere diameter in either direction;
    this is a representative search scope, not a bound on possible access.
    Within a chart the zero offset-plane station (or its clipped limit) is
    preferred. At an exact coplanar overlap use an interior representative.
    """
    from .conformalsearch import Rail
    from .sharedsearch import KEYS
    tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    polygons=[p.polygon for p in surface.patches]
    tree=STRtree(polygons)
    obs=paths.obs; mesh=obs.mesh
    for lane_id, sign, _ in paths.endpoint:
        lane=paths.lanes[lane_id]
        # Work on each finite straight piece in the endpoint neighbourhood.
        pieces=range(len(lane.length)) if sign>0 else range(len(lane.length)-1,-1,-1)
        for ball in balls.values():
            node=Rail(ball.name,lane.parent,lane.line,-1,sign)
            if node not in chains:continue
            r=ball.radius+sc.SKIN; available=2*r; station=0.
            for j in pieces:
                t=sign*lane.tangent[j]; up=railride.support_normal(t); length=min(available,lane.length[j])
                end=j if sign>0 else j+1
                anchor=lane.p[end] if lane.axis is None else lane.axis[end]+(r+lane.tube)*up
                far=anchor+length*t; pad=r+tol
                for k in tree.query(box(*np.r_[np.minimum(anchor,far)[XZ]-pad,np.maximum(anchor,far)[XZ]+pad])):
                    k=int(k); p=surface.patches[k]; source=layout.patch_nodes[ball.name][k]
                    if k not in layout.roads[source].rest_patches or not quasistatic.holds(ball,p.normal):continue
                    n=p.normal
                    ab=bearing_intervals(p,anchor,t,length,r,tol)
                    stations=[]
                    for low,high in ab:
                        rate=float(t@n); error=float(anchor@n-p.offset-r)
                        u=float(np.clip(-error/rate,low,high)) if abs(rate)>EPS else (low+high)*.5
                        stations.append((anchor+u*t,low,high))
                    if lane.axis is not None:
                        # The road can meet a crown off-centre; solve the
                        # finite end ring against the offset road plane.
                        axis=lane.axis[end]; radial=r+lane.tube
                        side=sc.unit(np.cross(up,t))
                        for theta in rimclear.trig_zeros(axis@n-p.offset-r,radial*(up@n),radial*(side@n),-math.pi/2,math.pi/2):
                            z=up*math.cos(theta)+side*math.sin(theta)
                            if quasistatic.holds(ball,z):
                                q=axis+radial*z
                                if p.polygon.covers(Point((q-r*n)[XZ])):
                                    stations.append((q,0.,0.))
                    for center,low,high in stations:
                        stats['bearing_intersections']+=1
                        at=center-(center@n-p.offset-r)*n
                        if lane.double and paths.geometry_support(center,ball,True,t)<0:
                            stats['bearing_no_opposing_reactions']+=1;continue
                        contacts=[c for c in mesh.contacts(center,ball.radius,railride.FIT) if mesh.rail[c.triangle] and c.normal[1]>EPS and
                                  abs(np.linalg.norm(c.point-center)-r)<=tol+EPS]
                        if not contacts:
                            stats['bearing_no_raw_rail_support']+=1;continue
                        # A raw normal with an old-rail contact on a different
                        # tube cannot publish this chart; capture checks that.
                        for key in sorted(KEYS,key=lambda k: -(k@t[XZ])):
                            law=sc.law(ball,n,key); d=sc.unit(law.impulse)
                            if d@t<=EPS:continue
                            depth=_approach(p,at-r*n,d)
                            if depth<=EPS:continue
                            distance=min(depth*.5,r)
                            track=law.along(d).over(0.,distance)
                            if track is None:continue
                            speed,seconds=track; incoming=speed*d
                            if cf_load(ball,key,n)<=EPS:continue
                            start=ph.State(at-distance*d,np.zeros(3)); tick=seconds/ph.DT
                            radius=ball.radius-sc.RECAPTURE
                            legs=((start.pos,at),(at,center))
                            if any(cs._capsule_blocked(tf,obs,a,b,radius,set(p.faces)) or any(
                                    motion.box_hit(motion.line(a,b),lo,hi) is not None for lo,hi in obs.boxes) for a,b in legs):
                                stats['bearing_approach_obstructed']+=1;continue
                            first=max(contacts,key=lambda c:c.normal[1]); hit=dict(triangle=first.triangle,
                                point=first.point,normal=first.normal,position=center,incoming=incoming,
                                manifold=[dict(triangle=c.triangle,point=c.point,normal=c.normal) for c in contacts],
                                incoming_omega=ph.rolling_spin(incoming,n,ball.radius))
                            caught=paths.capture(ball,hit,-1)
                            if caught is None or caught.lane!=lane_id or caught.direction!=sign:
                                stats['bearing_other_chart_or_no_capture']+=1;continue
                            if caught.crown is not None and not caught.crown.get('supported'):
                                stats['bearing_crown_load_unproved']+=1;continue
                            receiver=(sc.unit(caught.body.pos-caught.crown['tube_axis']) if caught.crown is not None else up)
                            if not quasistatic.holds(ball,receiver) or cf_load(ball,key,receiver)<=EPS:
                                stats['bearing_receiver_load_unproved']+=1;continue
                            if np.linalg.norm(center-caught.body.pos)>tol+EPS:
                                stats['bearing_capture_outside_join_band']+=1;continue
                            if cs._capsule_blocked(tf,obs,center,caught.body.pos,radius,set(p.faces)):
                                stats['bearing_capture_obstructed']+=1;continue
                            seed=dict(source_patch=k,origin=start.pos,launch=center,velocity=incoming,
                                      face=caught.triangle,ticks=tick,flight_ticks=0.,runup=distance,rim_ticks=0.,
                                      motion=(motion.rolling(start,law.impulse,law.loss,tick),motion.line(at,center),
                                              motion.line(center,caught.body.pos)),contacts=(),
                                      key_table=((math.ceil(tick),key.copy()),),inlet_kind='offset_road_rail_join')
                            stats['bearing_supported_inlets']+=1
                            yield source,ball,seed,caught,key,(station+low,station+high),'offset_road_rail_join'
                            break
                station+=length;available-=length
                if available<=EPS:break


def cf_load(ball,key,normal):
    from . import closedform as cf
    return -float(cf.air_impulse(ball,push=key)@normal/ph.DT)
