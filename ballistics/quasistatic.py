"""Quasistatic two-chart road certificates, not literal-input replays."""
import math

import numpy as np
from shapely import Point

from . import conformal, conformalsearch as cs, motion, physics as ph, rimclear, supportcarry as sc, terrain

XZ=[0,2]
EPS=1.e-7


def holds(ball,n):
    """The balancing world force must fit both independent key axes.

    Use the conservative C19 endpoint (catalogue paper approximation as in
    terrain). A strict interior gives a nonzero tangential control margin.
    """
    return n[1]>EPS and np.max(np.abs(n[XZ]/n[1]))<terrain.bounds(ball)[0]-EPS


def pair_center(a,n,m,p,q,r):
    """Centre at the intersection of both offset planes, at this seam station."""
    cos=float(n@m); det=1.-cos*cos
    if det<=EPS**2:return None
    rhs=np.array([p+r-n@a,q+r-m@a]); u,v=np.array([[1.,-cos],[-cos,1.]])@rhs/det
    return a+u*n+v*m


def open_leg(tf,obs,at,far,radius):
    """First obstruction of the ray segment leaving the actual joint.

    A short certified corridor is enough. Clip against feature cells and
    solve their sphere quadratics; do not try a sequence of shorter runs.
    """
    from . import sweep
    lo=np.minimum(at,far)-radius;hi=np.maximum(at,far)+radius
    fs=np.asarray(tf._box_triangles(lo[None],hi[None],obs.lo,obs.hi,obs.mesh._grid),dtype=int)
    d=far-at; first=1.
    # A finite triangle cannot be nearer than its infinite plane. The
    # oriented distance is affine along this leg, so endpoint ranges
    # reject whole triangles before any Voronoi/sphere quadratics.
    normals=obs.mesh.N[fs]; origins=obs.tris[fs,0]
    a=np.einsum('ij,ij->i',at-origins,normals)
    b=np.einsum('ij,ij->i',far-origins,normals)
    fs=fs[np.maximum(np.minimum(a,b),-np.maximum(a,b))<=radius+EPS]
    from .geometry import _closest
    mid=(at+far)*.5
    # Distance to a closed triangle is 1-Lipschitz. A segment lies
    # inside the ball of radius |d|/2 around this centre.
    reach2=(radius+.5*np.linalg.norm(d)+EPS)**2
    for f in fs:
        tri=obs.tris[f]; normal=obs.mesh.N[f]
        delta=mid-_closest(*mid,*tri.ravel())[:3]
        if delta@delta>reach2:
            continue
        for feature in range(7):
            a,b,w,v=sweep.feature_span(tri,normal,feature,at,d)
            a,b=sweep.sphere_span(w,v,radius,max(a,0.),min(b,first))
            if a<=b:first=min(first,a)
        if first<=EPS:return None
    line=motion.line(at,far)
    for a,b in obs.boxes:
        stop=motion.box_hit(line,a,b)
        if stop is not None:first=min(first,stop)
    if first*np.linalg.norm(d)<=4*EPS:return None
    return far if first==1. else at+d*(first*.5)


def concave_stations(tf,obs,p,q,span,r,radius):
    """Admissible two-foot stations along a finite offset-plane intersection.

    Feet are affine in the station. Intersect the finite chart polygons,
    then remove the raw-mesh sphere/feature intervals at the centre itself.
    One interior point of each remaining open cell suffices for its local
    corridor test; this is not a sequence of guessed launch positions.
    """
    from shapely import LineString
    from . import sweep
    from .sharedsearch import _lines
    n,m=p.normal,q.normal
    a,b=(pair_center(x+r*n,n,m,p.offset,q.offset,r) for x in span)
    if a is None or b is None:return
    d=b-a; dx=d[XZ]; z=dx@dx
    if z<=EPS**2:return
    common=LineString((a[XZ],b[XZ]))
    from shapely.affinity import translate
    for patch,v in ((p,n),(q,m)):
        common=common.intersection(translate(patch.polygon,*(r*v[XZ])).buffer(EPS,quad_segs=1))
    intervals=[]
    for line in _lines(common):
        if line.is_empty:continue
        limits=sorted(float((np.asarray(x)-a[XZ])@dx/z) for x in (line.coords[0],line.coords[-1]))
        if limits[1]-limits[0]>EPS:intervals.append(tuple(limits))
    if not intervals:return
    lo=np.minimum(a,b)-radius;hi=np.maximum(a,b)+radius
    fs=tf._box_triangles(lo[None],hi[None],obs.lo,obs.hi,obs.mesh._grid)
    blocked=[]
    for f in fs:
        tri=obs.tris[f]; normal=obs.mesh.N[f]
        for feature in range(7):
            u,v,w,e=sweep.feature_span(tri,normal,feature,a,d)
            u,v=sweep.sphere_span(w,e,radius,max(0.,u),min(1.,v))
            if u<=v:blocked.append((u,v))
    for lo,hi in obs.boxes:
        u,v=0.,1.
        for k in range(3):
            if abs(d[k])<=EPS:
                if not lo[k]<=a[k]<=hi[k]:u,v=1.,0.;break
                continue
            left,right=sorted(((lo[k]-a[k])/d[k],(hi[k]-a[k])/d[k]));u=max(u,left);v=min(v,right)
        if u<=v:blocked.append((u,v))
    for lo,hi in intervals:
        for x,y in sorted(blocked):
            if y<lo:continue
            if x>hi:break
            if x-lo>EPS:yield a+d*((lo+x)*.5)
            lo=max(lo,y)
        if hi-lo>EPS:yield a+d*((lo+hi)*.5)


def certificate(tf,obs,surface,ball,source,target,span,depth_for):
    k,_,a,b,xy=source; j,_,x,y,qxy=target
    p,q=surface.patches[k],surface.patches[j]; n,m=p.normal,q.normal
    r=ball.radius+sc.SKIN; tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    source_hold=holds(ball,n); receiver_hold=holds(ball,m)
    if not source_hold:return None,'quasi_source_requires_momentum_or_gravity_leg'
    u=np.array([xy[0],p.height(xy),xy[1]]); v=np.array([qxy[0],q.height(qxy),qxy[1]])
    if np.linalg.norm(u-v)>tol+EPS:return None,'quasi_support_gap'
    out=sc.unit(np.cross(b-a,n))
    # inc is the opposite of the target's outward (cross(edge,normal)).
    inc=-sc.unit(np.cross(y-x,m))
    dot=float(np.clip(n@m,-1.,1.)); bend=math.acos(dot); turn=sc.unit(m-dot*n)
    if bend<=EPS or abs(out@turn)<=EPS or np.linalg.norm(u+r*n-v-r*m)<=tol+EPS:return None,'quasi_not_a_fold'
    first=turn if turn@out>0. else -turn
    second=sc.unit((n-dot*m) if turn@out<0. else (dot*m-n))
    if not receiver_hold:
        # Once the offset planes meet, gravity may carry on over the
        # receiver. The neutral-key tangent must separate from the old
        # chart as well as enter the new one (not merely lower its height).
        if turn@out>=0. or m[1]<cs.WALK_NY:return None,'quasi_receiver_requires_entry_velocity'
        second=sc.unit(sc.law(ball,m,np.zeros(2)).impulse)
        if second@n<=EPS or second@inc<=EPS:return None,'quasi_receiver_requires_entry_velocity'
    if first@out<=EPS or second@inc<=EPS:return None,'quasi_crossing_orientation'
    arc=None; support_line=sc.unit(np.cross(n,m)); reaction=None
    if turn@out<0.:
        joints=concave_stations(tf,obs,p,q,span,r,ball.radius-sc.RECAPTURE)
        # Both feet must lie on their actual finite charts. The two one-
        # sided corridors meet at the SAME centre, with positive reactions
        # possible at the junction. Neither requires a long single-face run.
        # Positive two-contact reactions with controllable horizontal
        # balance. When only n admits rest, choose a small positive share
        # on m using the proven remaining independent-axis margin on n.
        a_h,b_h=n[XZ]/n[1],m[XZ]/m[1]; margin=terrain.bounds(ball)[0]-np.max(np.abs(a_h))
        delta=np.max(np.abs(b_h-a_h)); share=.5 if receiver_hold or delta<=EPS else min(.5,.5*margin/delta)
        reaction=(ph.GRAVITY*(1-share)/n[1],ph.GRAVITY*share/m[1])
        reason='quasistatic_concave_two_faces' if receiver_hold else 'quasistatic_concave_gravity_receiver'
    else:
        # Every intermediate radial normal is a positive normalized
        # combination of n,m, so its slope ratios stay in their interval.
        # Use any sufficiently small positive speed; this is a spatial
        # certificate under reversible quasistatic control, not a replay.
        if bend>=math.pi/2:return None,'quasi_convex_normal_turn'
        pressure_bound=ph.GRAVITY*min(n[1],m[1]); max_speed=math.sqrt(r*pressure_bound)
        arc=motion.arc(u,r*n,r*turn,0.,bend,np.zeros(3),((max_speed*.5)**2,0.,0.))
        before=u+r*n; z=arc.point(1.); err=float(z@m-q.offset-r)
        if abs(err)>tol+EPS:return None,'quasi_convex_support_offset'
        after=z-err*m
        joints=(before,)
        reason='quasistatic_convex_pressure'
    failure='quasi_no_clear_two_foot_station'
    for joint in joints:
        before=joint
        if arc is None:after=before
        fp=(before-r*n)[XZ]; fq=(after-r*m)[XZ]
        if not (p.polygon.buffer(EPS,quad_segs=1).covers(Point(fp)) and q.polygon.buffer(EPS,quad_segs=1).covers(Point(fq))):
            failure='quasi_contact_foot_outside';continue
        dp=float(np.linalg.norm(first[XZ])); dq=float(np.linalg.norm(second[XZ]))
        if min(dp,dq)<=EPS:failure='quasi_vertical_corridor';continue
        dl=depth_for(p,fp,first[XZ]); dr=depth_for(q,fq,-second[XZ]);
        if min(dl,dr)<=4*EPS:failure='quasi_no_open_corridor';continue
        start=before-first*(min(r,dl*.5)/dp); end=after+second*(min(r,dr*.5)/dq)
        radius=ball.radius-sc.RECAPTURE
        start=open_leg(tf,obs,before,start,radius);end=open_leg(tf,obs,after,end,radius)
        if start is None or end is None:failure='quasi_corridor_obstruction_at_joint';continue
        if arc is not None:
            z=arc.point(1.)
            if cs._capsule_blocked(tf,obs,z,after,radius,set()) or any(
                    motion.box_hit(motion.line(z,after),lo,hi) is not None for lo,hi in obs.boxes):
                return None,'quasi_offset_join_obstruction'
            verdict,_=rimclear.check(tf,ball,arc,obs,(a,b),finite_span=span,incident_supports=((x,y),),stop_on_unproved=True)
            if verdict!='clear':return None,'quasi_convex_'+verdict
        trajectory=(motion.line(start,before),*((arc,motion.line(arc.point(1.),after)) if arc is not None else ()),motion.line(after,end))
        witness=dict(kind='roll',source_patch=k,target_patch=j,origin=start,position=end,mode=reason,quasistatic=True,
                     literal_keys_rechecked=False,reference_key=sc.keys_along(out),motion=trajectory,
                     support_line=support_line,bearing_normals=(n,m),feet=(fp,fq),joint_center=before,
                     bearing_reactions_per_mass=reaction,receiver_gravity_only=not receiver_hold,pressure_speed_bound=None if arc is None else max_speed,
                     witness_reference='v30_finite_quasistatic_certificate')
        return witness,reason
    return None,failure

