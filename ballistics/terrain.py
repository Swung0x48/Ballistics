"""Ball-specific ground support and step mechanics on conformal geometry."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from shapely import LineString, Point, Polygon, box, union_all
from shapely.affinity import translate

from . import closedform as cf, conformal, motion, physics as ph

XZ=[0,2]
EPS=1.e-7
# C19 measured single-key pass/fail bracket. Paper has no measured
# bracket: use its spherical catalogue acceleration (ASSUMED for support).
BRACKET={'stone':(16.,17.), 'wood':(30.5,39.75)}


def ratio(ball):
    return ball.push/(ph.GRAVITY*ph.DT)


def bounds(ball):
    if ball.name in BRACKET:
        return tuple(math.tan(math.radians(a)) for a in BRACKET[ball.name])
    return (ratio(ball),)*2


def slope_verdict(ball, grade, horizontal):
    """Return -1 denied, 0 measured interval, or 1 permitted from rest."""
    n=float(np.linalg.norm(horizontal))
    if n<=EPS or grade<=EPS:return 1
    # Independent world axes. A single key gives max(abs(f)); two give L1.
    projection=float(np.sum(np.abs(horizontal)))/n
    lo,hi=bounds(ball)
    return 1 if grade<lo*projection-EPS else -1 if grade>hi*projection+EPS else 0


def support_line(a,b):
    """A concave two-contact bearing line, not a convex ridge."""
    t=np.cross(a,b)
    length=np.linalg.norm(t[XZ])
    if length<=EPS:return None
    trans=np.array([-t[2],t[0]])/length
    if (a[XZ]@trans)*(b[XZ]@trans)>=-EPS:return None
    return t/np.linalg.norm(t)


def slope_for(normals, delta):
    """Support slope and orientation; downhill is unrestricted."""
    for a,b in combinations(normals,2):
        t=support_line(a,b)
        if t is not None:
            signed=(t[XZ]@delta)*t[1]
            return abs(t[1])/np.linalg.norm(t[XZ]), t[XZ], signed, 'support_line'
    a=normals[0]
    return np.linalg.norm(a[XZ])/a[1], a[XZ], -a[XZ]@delta, 'plane'


def torque_ratio(height, radius):
    if height<=0:return 0.
    if height>=radius:return math.inf
    return math.sqrt(height*(2*radius-height))/(radius-height)


def step_verdict(ball,height,radius,forward):
    """Ideal sharp-lip torque, not the C19 sustained-slope bracket."""
    if height<=0.:return -1
    required=torque_ratio(height,radius)
    available=ratio(ball)*float(np.sum(np.abs(forward)))
    return 1 if required<available-EPS else -1 if required>available+EPS else 0


def step_work(height,radius,force):
    theta=math.acos(1-height/radius)
    critical=min(theta,max(0.,math.atan2(force,ph.GRAVITY)))
    potential=lambda a:radius*(ph.GRAVITY*math.cos(a)+force*math.sin(a))
    return theta, potential(critical)-potential(theta), potential(theta)-ph.GRAVITY*radius


def step_entry_speed(ball,height,forward,inertia,*,push=None):
    """Required horizontal rolling approach in the ideal inelastic pivot model."""
    r=ball.radius
    if not 0.<height<r:return math.inf
    force=ratio(ball)*ph.GRAVITY*float(np.sum(np.abs(forward)) if push is None else np.asarray(push)@forward)
    _, barrier, _=step_work(height,r,force)
    return math.sqrt(2*max(0.,barrier)/(1+inertia))/(1-height/(r*(1+inertia)))


def step_energy(ball, height, speed, forward, inertia, *, push=None, radius=None):
    """Ideal angular momentum capture followed by work along a fixed lip."""
    r=ball.radius if radius is None else radius
    if not 0.<height<r:return None
    force=ratio(ball)*ph.GRAVITY*float(np.sum(np.abs(forward)) if push is None else np.asarray(push)@forward)
    _, barrier, crown=step_work(height,r,force)
    pivot_speed=max(0.,speed)*(1-height/(r*(1+inertia)))
    energy=.5*(1+inertia)*pivot_speed**2
    return energy-barrier, math.sqrt(max(0.,2*(energy+crown)/(1+inertia)))


def step_runup(axis, initial_speed, distance, required_speed):
    """A witnessed straight run-up, not an assumed terminal entry speed.

    The caller must establish support, clearance and available input on
    the whole run; a positive result alone does not certify the lip.
    """
    crossing=axis.over(initial_speed,distance)
    return crossing, crossing is not None and crossing[0]>required_speed+EPS


def step_contact_limits(height,radius,force,inertia,pivot_speed,mu):
    """(Impact friction margin, least N/m, least (mu*N-|T|)/m).

    For a horizontal rolling entry (zero speed means no impact). N and
    N +/- T/mu are affine in sin/cos on the constant-force energy arc;
    their extrema are endpoints or stationary angles, not probes.
    """
    theta=math.acos(1-height/radius); g=ph.GRAVITY
    c,s=math.cos(theta),math.sin(theta)
    impact=mu-inertia/(1+inertia)*(1-c)/s if pivot_speed>EPS else mu
    scale=2/(1+inertia)
    offset=-pivot_speed**2/radius-scale*(g*c+force*s)
    nc,ns=(1+scale)*g,(1+scale)*force
    tc,ts=inertia/(1+inertia)*force,-inertia/(1+inertia)*g
    def minimum(a,b,d):
        angle=math.atan2(b,a)
        xs=[0.,theta]+[x for x in (angle,angle+math.pi,angle-math.pi) if 0.<x<theta]
        return min(d+a*math.cos(x)+b*math.sin(x) for x in xs)
    return (impact,minimum(nc,ns,offset),min(minimum(mu*nc+j*tc,mu*ns+j*ts,mu*offset) for j in (-1,1)))


@dataclass(frozen=True)
class StepLip:
    lower: int
    upper: int
    a: np.ndarray
    b: np.ndarray
    forward: np.ndarray
    face: int
    owner: int

    def height(self,surface,point):
        return point[1]-surface.patches[self.lower].height(point[XZ])


def neighbours(surface):
    incident=defaultdict(list)
    for k,p in enumerate(surface.patches):
        for f,a,b in p.boundary:
            incident[conformal.edge_key(a,b)].append((k,a,b,f))
    pairs=defaultdict(list)
    for es in incident.values():
        for (a,x,y,_),(b,_,_,_) in combinations(es,2):
            if a!=b:
                pairs[a,b].append(LineString([x[XZ],y[XZ]]))
                pairs[b,a].append(pairs[a,b][-1])
    return incident,pairs


def _parts(geometry):
    if geometry.is_empty:
        return
    if hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _parts(part)
    else:
        yield geometry


def step_lips(mesh,surface,incident):
    """Find supported intervals of finite horizontal lips on vertical risers.

    The wall is a polygon in (distance along lip, height). Remove the
    projection of its missing area from the lip interval; intersect what
    remains with the lower patch footprint. Neither a single triangle's
    low vertex nor coverage of the complete mesh edge proves local support.
    """
    from shapely import STRtree
    from shapely.affinity import translate as shift
    from shapely.ops import substring
    polygons=[p.polygon for p in surface.patches]
    if not polygons:return {}
    index=STRtree(polygons)
    eps=conformal.PLANE_ERROR
    out=defaultdict(list); seen=set(); by_edge=defaultdict(list)
    vertical=np.flatnonzero((abs(mesh.N[:,1])<=math.sin(math.radians(.3))) & ~mesh.rail & ~mesh.placed)
    wall_edges={}
    for i in vertical:
        tri=surface.triangles[i]
        wall_edges[i]=tuple(conformal.edge_key(a,b) for a,b in zip(tri,np.roll(tri,-1,axis=0)))
        for edge in wall_edges[i]:by_edge[edge].append(i)
    for i in vertical:
        tri=surface.triangles[i]
        for (a,b),edge in zip(zip(tri,np.roll(tri,-1,axis=0)),wall_edges[i]):
            row=incident.get(edge,())
            if not row or abs(a[1]-b[1])>eps:continue
            origin=a[XZ]; span=b[XZ]-origin; length=float(np.linalg.norm(span))
            if length<=EPS:continue
            along=span/length; top=(a+b)*.5
            fwd=np.array([-along[1],along[0]])
            foot=LineString([origin,b[XZ]])
            for upper,_,_,face in row:
                cap=surface.patches[upper]
                interior=surface.triangles[face].mean(0)[XZ]-top[XZ]
                forward=fwd if interior@fwd>0. else -fwd
                candidates=[]
                for lower in index.query(foot.buffer(eps)):
                    lower=int(lower); floor=surface.patches[lower]
                    if lower==upper or floor.normal[1]<conformal.ANGLE:continue
                    level=floor.height(top[XZ]); h=top[1]-level
                    if not eps<h<ph.WOOD.radius:continue
                    if any(abs(floor.height(q[XZ])-level)>eps for q in (a,b)):continue
                    # The lower floor must extend towards the approach, not
                    # merely touch the riser from the upper side.
                    arm=math.sqrt(h*(2*ph.WOOD.radius-h))
                    lower_poly=floor.polygon.buffer(eps)
                    coverage=lower_poly.intersection(foot).intersection(shift(lower_poly,*(arm*forward)))
                    if not coverage.is_empty:candidates.append((lower,level,coverage))
                if not candidates:continue
                wall={i}; queue=[i]
                for f in queue:
                    for e in wall_edges[f]:
                        for j in by_edge[e]:
                            if j in wall or mesh.owner[i]!=mesh.owner[j]:continue
                            if np.max(abs((surface.triangles[j][:,XZ]-origin)@fwd))>eps:continue
                            wall.add(j);queue.append(j)
                profile=union_all([Polygon(np.column_stack(((surface.triangles[j][:,XZ]-origin)@along,
                                                             surface.triangles[j][:,1]))) for j in sorted(wall)]).buffer(eps)
                for lower,level,coverage in candidates:
                    # Test the vertical strip between the two planes, allowing
                    # only the declared mesh tolerance at each end.
                    rectangle=box(0.,level,length,top[1])
                    missing=rectangle.difference(profile)
                    allowed=LineString([(0.,0.),(length,0.)])
                    for gap in _parts(missing):
                        x0,_,x1,_=gap.bounds
                        if x1-x0>EPS:allowed=allowed.difference(LineString([(x0,0.),(x1,0.)]))
                    for interval in _parts(allowed):
                        x0,_,x1,_=interval.bounds
                        if x1-x0<=EPS:continue
                        usable=substring(foot,x0,x1).intersection(coverage)
                        for piece in _parts(usable):
                            if piece.geom_type!='LineString' or piece.length<=EPS:continue
                            t=sorted(float((np.array(q)-origin)@along/length) for q in (piece.coords[0],piece.coords[-1]))
                            x,y=(a+np.clip(q,0.,1.)*(b-a) for q in t)
                            key=(lower,upper,conformal.edge_key(x,y))
                            if key in seen:continue
                            seen.add(key)
                            out[lower].append(StepLip(lower,upper,x,y,forward,int(face),int(mesh.owner[i])))
    return dict(out)


def rebuild(g,mesh,surface,log=lambda *_:None):
    """Replace only road/step adjacency; rails and rail ends unchanged."""
    from . import coarse
    n=g.info['rest_places'];P=g.pos
    balls=(ph.WOOD,ph.STONE,ph.PAPER); full=7
    incident,seams=neighbours(surface)
    support=[]
    for p in P[:n]:
        support.append(tuple(sorted({int(surface.face_patch[c.triangle]) for c in mesh.contacts(p,ph.WOOD.radius,.025)
                      if c.normal[1]>=coarse.WALK_NY and surface.face_patch[c.triangle]>=0})))
    g.info['terrain_support']=tuple(support)
    lips=step_lips(mesh,surface,incident)
    g.info['terrain_lips']=lips
    masks={}; uncertain=[]; reasons={}; counts=Counter()
    polygons=[translate(p.polygon,*(ph.WOOD.radius*p.normal[XZ])).buffer(conformal.PLANE_ERROR) for p in surface.patches]
    solid=mesh.subset(mesh.placed) if mesh.placed.any() else None
    def blocked(u,v):
        if solid is None:return False
        r=ph.WOOD.radius
        return motion.near_hit(motion.line(P[u],P[v]),lambda p:solid.distance(p,r+1.)[0],r-.05) is not None
    # Run-up/lip representatives retain the v12 geometric proposal set.
    # This is not the adjacency used by the state graph or rolling SCCs.
    runup=[tuple(e for e in row if e[2]=='roll') for row in g.adj]
    old=[{v for v,_,_ in row} for row in runup[:n]]
    cells=defaultdict(list)
    for i,ij in enumerate(g.info['rest_cell']):cells[tuple(ij)].append(i)
    for row in g.adj:
        row[:]=[(v,c,k) for v,c,k in row if k not in ('roll','step')]
    for (i,j),ids in cells.items():
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            for u in ids:
                for v in cells.get((i+di,j+dj),()):
                    if not support[u] or not support[v]:continue
                    d=(P[v]-P[u])[XZ]
                    if np.linalg.norm(d)<=EPS:continue
                    shared=tuple(k for k in support[u] if k in support[v])
                    line=None; conditions=[]
                    if shared:
                        candidates=tuple(k for k in shared if polygons[k].covers(line if line is not None else
                                              (line:=LineString([P[u,XZ],P[v,XZ]]))))
                        if candidates:
                            normals=[surface.patches[k].normal for k in candidates]
                            conditions=[slope_for(normals,d)]
                    if not conditions:
                        # A sphere rounds a real shared seam. Its horizontal
                        # centre cannot be more than r from the finite edge.
                        if line is None:line=LineString([P[u,XZ],P[v,XZ]])
                        adjacent=[(a,b) for a in support[u] for b in support[v] if (a,b) in seams and
                                  any(line.distance(e)<=ph.WOOD.radius+EPS for e in seams[a,b])]
                        if not adjacent:
                            if v in old[u]:reasons[u,v,'roll']='no_shared_support'
                            continue
                        a,b=adjacent[0]
                        conditions=[slope_for([surface.patches[k].normal],d) for k in (a,b)]
                    if blocked(u,v):
                        continue
                    cost=float(np.linalg.norm(P[u]-P[v]))
                    bits=0; why='plane'
                    for k,ball in enumerate(balls):
                        failed=[]
                        for grade,f,sign,how in conditions:
                            if sign<-EPS:continue
                            verdict=slope_verdict(ball,grade,f)
                            if verdict<=0:
                                failed.append(how)
                                if verdict==0:uncertain.append((u,v,ball.name,how,math.degrees(math.atan(grade))))
                        if not failed:bits|=1<<k
                        else:why='support_line' if 'support_line' in failed else 'slope'
                    if bits!=full:
                        masks[u,v,'roll']=bits; reasons[u,v,'roll']=why
                    if bits:
                        g.adj[u][:]=[x for x in g.adj[u] if not (x[0]==v and x[2]=='drop')]
                        g.adj[u].append((v,cost,'roll')); counts['roll']+=1
    by_patch=defaultdict(list)
    for i,fields in enumerate(support):
        for k in fields:by_patch[k].append(i)
    step_checks={}
    for lower, steps in lips.items():
        for step in steps:
            a,b=step.a,step.b; f=step.forward
            span=b[XZ]-a[XZ]; span2=float(span@span)
            if span2<=EPS:continue
            mu_surface=float(mesh.entities[step.owner].get('friction',.7)) if step.owner>=0 else .7
            for u in by_patch[lower]:
                p=P[u,XZ]; x=float((p-a[XZ])@f); t=float((p-a[XZ])@span/span2)
                if not (-ph.WOOD.radius-coarse.STEP<=x<=0. and 0.<=t<=1.):continue
                pivot=a+t*(b-a); height=step.height(surface,pivot)
                if not 0.<height<ph.WOOD.radius:continue
                arm=math.sqrt(height*(2*ph.WOOD.radius-height))
                if x > -arm+conformal.PLANE_ERROR:continue
                # Do not mistake one chosen coarse node for the physical
                # crown. Every locally connected cap node can represent it.
                crown=Point(pivot[XZ]); reach=math.sqrt(2)*coarse.STEP+EPS
                for v in by_patch[step.upper]:
                    q=P[v,XZ]; d=q-pivot[XZ]
                    if d@f < -EPS or d@d>reach**2:continue
                    if not polygons[step.upper].covers(LineString([crown,q])):continue
                    if any(j==v and how=='roll' for j,_,how in g.adj[u]):continue
                    key=(u,v,'step'); bits=masks.get(key,0)
                    for k,ball in enumerate(balls):
                        if bits&(1<<k):continue
                        inertia=.277 if ball.name=='paper' else .4
                        state=step_verdict(ball,height,ball.radius,f)
                        force=ratio(ball)*ph.GRAVITY*float(np.sum(np.abs(f)))
                        margins=step_contact_limits(height,ball.radius,force,inertia,0.,ball.friction*mu_surface)
                        why=('needs_momentum' if state!=1 else 'rolling_contact_unproved' if min(margins)<-EPS
                             else 'placed_obstruction' if blocked(u,v) else 'static_torque')
                        if why=='static_torque':bits|=1<<k
                        entry=(why,height,step,step_entry_speed(ball,height,f,inertia))
                        check=(u,v,ball.name)
                        if check not in step_checks or why=='static_torque':step_checks[check]=entry
                    masks[key]=bits; reasons[key]='step_torque'
                    if bits and not any(j==v and how=='step' for j,_,how in g.adj[u]):
                        g.adj[u].append((v,float(np.linalg.norm(P[v]-P[u]))+10.,'step')); counts['step']+=1
    for (u,v,ball),(why,height,_,_) in step_checks.items():
        if why!='static_torque':uncertain.append((u,v,ball,why,height))
    g.info['terrain_step_checks']=step_checks
    g.info.update(terrain_ball_masks=masks,terrain_uncertain=uncertain,terrain_reasons=reasons,
                  launch_runup_adj=runup,terrain_counts=dict(counts))
    log('terrain: %d rolls, %d steps, %d riser lips, %d indeterminate ball edges' %
        (counts['roll'],counts['step'],sum(map(len,lips.values())),len(uncertain)))
    return counts


def climb_turn(law, direction, velocity):
    """First maximum uphill progress on a fixed support and input."""
    axis=law.along(direction)
    speed=float(np.asarray(velocity)@direction)
    if speed<=EPS or axis.impulse>=-EPS:return math.inf
    return cf._turn(speed,axis.impulse,axis.loss)


def meet_step(surface, lips, ball, start, law, end):
    """Exact forward root to the lip's contact line for a horizontal approach.

    A tilted or skew riser is deliberately not certified by this model.
    """
    r=ball.radius+ph.Params().floor
    best=None
    for lip in lips:
        if surface.patches[lip.lower].normal[1]<conformal.ANGLE:continue
        a,b=lip.a,lip.b; forward=np.array([lip.forward[0],0.,lip.forward[1]])
        dy=start.pos[1]-a[1]
        if not 0.<dy<r:continue
        arm=math.sqrt(r*r-dy*dy)
        d=float((a-start.pos)@forward)-arm
        if d<-EPS:continue
        axis=law.along(forward); vel=float(start.vel@forward)
        if vel<=EPS and axis.impulse<=EPS:continue
        lo=cf._turn(vel,axis.impulse,axis.loss) if vel<0<axis.impulse else 0.
        hi=cf._turn(vel,axis.impulse,axis.loss) if axis.impulse<0 else math.inf
        hi=min(hi,end,best[0] if best else math.inf)
        if lo>hi:continue
        t=cf._increasing_root(vel,axis.impulse,axis.loss,max(0.,d),lo,hi)
        if t is None:continue
        dx,v=law.at(start.vel,t)
        q=start.pos+dx
        edge=(b-a)[XZ]; k=float((q-a)[XZ]@edge/(edge@edge))
        if not 0.<=k<=1.:continue
        pivot=a+k*(b-a)
        if abs(np.linalg.norm(q-pivot)-r)>2*conformal.PLANE_ERROR:continue
        best=(t,lip,pivot)
    return best
