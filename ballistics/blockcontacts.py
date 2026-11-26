"""Certificates for block contacts not represented by a finite welded seam."""
from collections import defaultdict
import math

import numpy as np
from shapely import STRtree

from . import conformal, motion, physics as ph, quasistatic as qs, rimclear, supportcarry as sc

XZ = [0, 2]
EPS = sc.EPS


def _foot_leg(tf, obs, surface, patch, point, normal, r, radius, incident):
    # Each incident triangle supplies a finite supported ray into the chart.
    at = point + r * normal
    for f in incident:
        if surface.face_patch[f] != patch:
            continue
        target = surface.triangles[f].mean(0)
        d = target - point
        length = np.linalg.norm(d)
        if length <= EPS:
            continue
        far = at + min(.5, r / length) * d
        result = qs.open_leg(tf, obs, at, far, radius)
        if result is not None:
            return result
    return None


def vertex(tf, obs, surface, ball, a, b, point, incident):
    """Short great-circle supported arc in the vertex's normal cone."""
    p, q = surface.patches[a], surface.patches[b]
    n, m = p.normal, q.normal
    if not qs.holds(ball, n) or not qs.holds(ball, m):
        return None, 'vertex_normal_outside_hold'
    if max(abs(n@point-p.offset),abs(m@point-q.offset))>EPS:
        return None, 'vertex_not_on_both_fitted_planes'
    # Over the short arc all radial normals are positive combinations of
    # these ends. Checking the ends therefore checks every incident ray.
    rays = surface.triangles[incident].reshape(-1,3) - point
    if np.max(rays @ np.stack((n,m)).T)>EPS:
        return None, 'vertex_outside_incident_normal_cone'
    r = ball.radius + sc.SKIN; inner = ball.radius - sc.RECAPTURE
    A, B = point+r*n, point+r*m
    dot=float(np.clip(n@m,-1.,1.))
    arc=None
    if np.linalg.norm(n-m)>EPS:
        tang=sc.unit(m-dot*n); angle=math.acos(dot)
        # Normal-force margin is strict at a sufficiently small speed.
        speed2=.25*r*ph.GRAVITY*min(n[1],m[1])
        arc=motion.arc(point,r*n,r*tang,0.,angle,np.zeros(3),(speed2,0.,0.))
        verdict,_=rimclear.check(tf,ball,arc,obs,support_vertex=point,stop_on_unproved=True)
        if verdict!='clear':
            return None, 'vertex_arc_'+verdict
    else:
        if qs.open_leg(tf,obs,A,A+EPS*8*n,inner) is None:
            return None, 'vertex_contact_obstructed'
    first = _foot_leg(tf,obs,surface,a,point,n,r,inner,incident)
    last = _foot_leg(tf,obs,surface,b,point,m,r,inner,incident)
    if first is None or last is None:
        return None, 'vertex_approach_or_exit_obstructed'
    return dict(rule='offset_vertex_arc' if arc else 'coplanar_support_vertex',vertex=point,
                normals=(n,m),motion=(motion.line(first,A),*((arc,) if arc else ()),motion.line(B,last))), None


def certify(tf,obs,g,layout,known):
    """Use exact vertex incidence and intersecting planar standing footprints.

    Return witnesses and named unresolved feature candidates. In particular,
    proximity outside a continuous support chart is not proof of a crossing.
    """
    surface=g.info['conformal']; spans=defaultdict(set); vertices=defaultdict(list)
    for f,tri in enumerate(surface.triangles):
        for p in tri:
            vertices[tuple(p)].append(f)
    for i,p in enumerate(surface.patches):
        for _,a,b in p.boundary:
            spans[tuple(a)].add(i);spans[tuple(b)].add(i)
    trials=defaultdict(list)
    for p, ids in spans.items():
        ii=sorted(ids)
        for x,a in enumerate(ii):
            for b in ii[x+1:]:
                trials[a,b].append(p)
    limits=defaultdict(list)
    available=defaultdict(dict)
    guarded={}; holdings={}
    for ball in ph.BALLS.values():
        name=ball.name
        blocked=guarded[name]={k for node,d in layout.roads.items() if node.ball==name for k in d.guarded_patches}
        holdings[name]=[qs.holds(ball,p.normal) for p in surface.patches]
        ranges=g.info['standing_ranges'][name]
        ids=[k for k,s in ranges.items() if not s.is_empty and k not in blocked]
        shapes=[ranges[k] for k in ids]; tree=STRtree(shapes)
        for i,shape in enumerate(shapes):
            a=ids[i];p=surface.patches[a]
            for j in tree.query(shape,predicate='intersects'):
                if j<=i:
                    continue
                b=ids[j];q=surface.patches[b];pair=tuple(sorted((a,b)))
                if (a,b,name) in known and (b,a,name) in known:
                    continue
                if np.linalg.norm(p.normal-q.normal)>EPS:
                    continue
                # Parallel offset planes have constant separation. The
                # entire vertical transition must fit the contact band.
                if abs(p.offset-q.offset)>min(sc.RECAPTURE,conformal.PLANE_ERROR):
                    limits[name,pair].append('parallel_offset_exceeds_contact_band');continue
                overlap=shape.intersection(shapes[j])
                if overlap.area<=EPS**2:
                    continue  # lower dimensional contact handled by incidence
                foot=np.asarray(overlap.representative_point().coords[0]);r=ball.radius+sc.SKIN
                point0=np.array([foot[0],p.height(foot,r),foot[1]]);point1=np.array([foot[0],q.height(foot,r),foot[1]])
                goal=point1 if np.linalg.norm(point1-point0)>EPS else point0+8*EPS*p.normal
                connector=qs.open_leg(tf,obs,point0,goal,ball.radius-sc.RECAPTURE)
                if connector is None or np.linalg.norm(connector-goal)>EPS:
                    limits[name,pair].append('parallel_offset_connector_obstructed');continue
                available[name][pair]=dict(rule='parallel_offset_overlap',offset_polygon=overlap,
                                           centers=(point0,point1),motion=(motion.line(point0,point1),))
    # Geometric clearance for a short spherical arc depends on radius,
    # not mass or thrust. Classify all eligible balls together at each
    # finite feature, after applying their individual holding inequalities.
    for pair,ps in trials.items():
        a,b=pair
        pending=[ball for ball in ph.BALLS.values() if a not in guarded[ball.name] and b not in guarded[ball.name]
                 and pair not in available[ball.name] and not ((a,b,ball.name) in known and (b,a,ball.name) in known)]
        for v in ps:
            if not pending:
                break
            sizes=defaultdict(list)
            for ball in pending:
                h=holdings[ball.name]
                if h[a] and h[b]:
                    sizes[ball.radius].append(ball)
                else:
                    limits[ball.name,pair].append('vertex_normal_outside_hold')
            for group in sizes.values():
                row,reason=vertex(tf,obs,surface,group[0],a,b,np.asarray(v),vertices[v])
                for ball in group:
                    if row is not None:
                        available[ball.name][pair]=row
                        pending.remove(ball)
                    else:
                        limits[ball.name,pair].append(reason)
    return available,dict(limits)
