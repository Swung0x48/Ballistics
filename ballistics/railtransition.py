"""Finite support-feature transfers between a road lip and a rail chart."""
import math

import numpy as np

from . import conformal, motion, quasistatic, railinlet, railride, rimclear, supportcarry as sc, terrain

XZ=[0,2]
EPS=sc.EPS
TOL=conformal.PLANE_ERROR


def station_roots(a, b, x, y, r):
    """Meet the radius-r cylinder around finite support edge x--y."""
    w=y-x; square=float(w@w)
    if square<=EPS*EPS:return ()
    def transverse(v):return v-(v@w/square)*w
    p=transverse(a-x); d=transverse(b-a)
    aa=float(d@d);bb=float(p@d);cc=float(p@p-r*r)
    if aa<=EPS*EPS:
        # Either no intersection or the entire line is on the bearing
        # cylinder. The latter is considered by the flat contact rule.
        return ()
    disc=bb*bb-aa*cc
    if disc<0:return ()
    q=math.sqrt(disc)
    out=[]
    for s in sorted({(-bb-q)/aa,(-bb+q)/aa}):
        if not -EPS<=s<=1.+EPS:continue
        c=a+min(1.,max(0.,s))*(b-a)
        u=float((c-x)@w/square)
        if -EPS<=u<=1.+EPS:out.append((s,c,x+min(1.,max(0.,u))*w))
    return tuple(out)


def rotation(p,r,n,m):
    dot=float(n@m); turn=math.acos(np.clip(dot,-1.,1.))
    if not EPS<turn<math.pi/2:return None
    return motion.arc(p,r*n,r*sc.unit(m-dot*n),0.,turn,np.zeros(3),(1.,0.,0.))


def reversal(curve):
    _,angle,*_=curve.arc
    c=math.cos(angle);s=math.sin(angle)
    return motion.arc(curve.p,c*curve.v+s*curve.b,s*curve.v-c*curve.b,0.,angle,np.zeros(3),(1.,0.,0.))


def reverse_parts(parts):
    return tuple(reversal(q) if isinstance(q,motion.Piece) else (q[1],q[0]) for q in reversed(parts))


def lip_handoff(tf,obs,ball,patch,seg,pivot,target):
    """Roll around a raw finite road edge until the receiving chart is met."""
    from . import railstatic
    face,x,y=seg; r=ball.radius+sc.SKIN; axis=sc.unit(y-x)
    # The pivot and bearing normal come from the raw triangle, not a
    # near-but-distinct neighbouring plane. Keep its upward orientation.
    n=obs.mesh.N[face]; n=n if n@patch.normal>=0 else -n
    if abs(n@axis)>EPS:return None,'lip_normal_not_perpendicular'
    m=sc.unit(target-pivot)
    if not quasistatic.holds(ball,n) or not quasistatic.holds(ball,m):return None,'lip_requires_momentum'
    arc=rotation(pivot,r,n,m)
    if arc is None:return None,'lip_rotation_unproved'
    tangent=sc.unit(m-(m@n)*n)
    depth=railinlet._approach(patch,pivot,tangent)
    if depth<=EPS:return None,'no_road_approach_to_lip'
    center=arc.point(0.); rest=center-min(depth*.5,r)*tangent
    obstacles=obs.boxes
    if any(motion.box_hit(motion.line(rest,center),a,b) is not None for a,b in obstacles):return None,'road_death_box'
    _,hit=railstatic.segment_features(tf,obs,rest,center,r)
    if hit:return None,'road_approach_obstructed'
    # Only penetration rejects an otherwise continuous bearing path. All
    # actual features remain screened; no rail/guard owner is ignored.
    verdict,_=rimclear.check(tf,ball,arc,obs,(x,y),finite_span=(x,y))
    if verdict not in ('clear','touching'):return None,'lip_'+verdict
    if np.linalg.norm(arc.point(1.)-target)>EPS:return None,'lip_end_not_on_rail'
    return ((rest,center),arc),'supported_lip_to_rail'


def candidates(tf,obs,ball,patch,lane,j,original,safe):
    """Bridge a certified rail interval from a road's finite bearing lip.

    A tilted auxiliary crown is determined by the road normal projected
    into the tube section, not a sweep over guessed angles. Its usable
    longitudinal intervals and all meetings with road lips are solved
    from mesh feature roots. It can only join the existing crown if the
    intervening transverse circular arc has complete raw bearing.
    """
    from . import railstatic
    a,b=original; ra,rb=safe; t=sc.unit(b-a); r=ball.radius+sc.SKIN
    margin=railstatic.axial_margin(ball,t)
    crown=lane.axis is not None; length=np.linalg.norm(b-a)
    up=railride.support_normal(t); rep=(ra+rb)*.5; reasons=set()
    def roots(u,v):
        return [(seg,f,q,p) for seg in patch.boundary for f,q,p in station_roots(u,v,*seg[1:],r)]
    for seg,_,target,pivot in roots(ra,rb):
        pre,reason=lip_handoff(tf,obs,ball,patch,seg,pivot,target)
        if pre is not None:return (*pre,(target,rep)),reason
        reasons.add(reason)
    if crown:
        side=sc.unit(np.cross(up,t)); normal=sc.unit(patch.normal-(patch.normal@t)*t)
        first=math.atan2(float(normal@side),float(normal@up))
        # Derive representatives from the intervals of the force cone.
        # A guard can cut off the nominal crown but not its entire shoulder.
        cap=math.atan(max(0.,margin)); bound=terrain.bounds(ball)[0]
        cuts={-cap,cap,0.}
        if -cap<first<cap:cuts.add(first)
        for index in XZ:
            for sign in (-1.,1.):
                x=bound*up[1]+sign*up[index]; y=bound*side[1]+sign*side[index]
                cuts.update(rimclear.trig_zeros(0.,x,y,-cap,cap))
        cuts=sorted(cuts)
        angles=([first] if -cap<first<cap else [])+[(l+h)*.5 for l,h in zip(cuts,cuts[1:]) if h-l>EPS]
        for angle in angles:
            if abs(angle)<=EPS:continue
            n=up*math.cos(angle)+side*math.sin(angle)
            if not quasistatic.holds(ball,n):continue
            d=(r+lane.tube)*(n-up); intersections=roots(a+d,b+d)
            if not intersections:continue
            cells,blocked=railstatic.segment_features(tf,obs,a+d,b+d,r,full_intervals=True)
            windows=railstatic.unblocked(railstatic.bearing_windows(cells,t,r,0.,lane.tube,margin),blocked)
            for seg,f,target,pivot in intersections:
                window=next((w for w in windows if w[0]<=f<=w[1]),None)
                if window is None:
                    reasons.add('lip_root_outside_rail_bearing');continue
                l,h=window
                lo=max(l,float((ra-a)@t/length));hi=min(h,float((rb-a)@t/length))
                if hi-lo<=EPS:
                    reasons.add('no_joint_axial_support_interval');continue
                pre,reason=lip_handoff(tf,obs,ball,patch,seg,pivot,target)
                if pre is None:reasons.add(reason);continue
                at=a+(.5*(lo+hi))*length*t
                axis=at-(r+lane.tube)*up
                turn=railstatic.crown_arc(tf,obs,ball,axis,t,lane.tube,n,up,
                                         float((lane.axis[j]-axis)@t)-TOL,float((lane.axis[j+1]-axis)@t)+TOL)
                if turn is None:
                    reasons.add('auxiliary_crown_transfer_unproved');continue
                return (*pre,(target,turn.point(0.)),turn,(at,rep)),reason
    return None,('no_lip_cylinder_root' if not reasons else sorted(reasons)[0])
