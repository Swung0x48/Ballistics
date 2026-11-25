"""Ball-specific ground support and step mechanics on conformal geometry."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from shapely import LineString, Point
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
    return slope_verdict(ball,torque_ratio(height,radius),forward) if height>0 else -1


def step_energy(ball, height, speed, forward, inertia, *, push=None):
    """Specific energy surplus at the work barrier of a sharp step.

    Read the incoming speed before impact. Angular momentum about the lip
    is conserved in an ideal capture; neglect further losses along the arc.
    Returns (surplus, circle speed at the crown); None if outside geometry.
    """
    r=ball.radius
    if not 0.<height<r:return None
    force=ratio(ball)*ph.GRAVITY
    force*=float(np.sum(np.abs(forward)) if push is None else np.asarray(push)@forward)
    theta=math.acos(1-height/r)
    arm=r*math.sin(theta)
    pivot_speed=max(0.,speed)*(1-height/(r*(1+inertia)))
    energy=.5*(1+inertia)*pivot_speed**2
    start=ph.GRAVITY*(r-height)+force*arm
    critical=min(theta,max(0.,math.atan2(force,ph.GRAVITY)))
    maximum=r*(ph.GRAVITY*math.cos(critical)+force*math.sin(critical))
    surplus=energy+start-maximum
    crown=energy+start-r*ph.GRAVITY
    return surplus, math.sqrt(max(0.,2*crown/(1+inertia)))


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


def step_lips(mesh,surface,incident):
    """Vertical risers attached to an upper lip and a lower road at the foot.

    The road can be exported with a T-junction along the foot of the riser,
    so the lower support is identified by its polygon and plane, not by a
    second identical pair of vertex indices.
    """
    from shapely import STRtree
    polygons=[p.polygon for p in surface.patches]
    if not polygons:return {}
    index=STRtree(polygons)
    out=defaultdict(list); seen=set()
    for i in np.flatnonzero((abs(mesh.N[:,1])<=math.sin(math.radians(.3))) & ~mesh.rail & ~mesh.placed):
        tri=surface.triangles[i]
        for a,b in zip(tri,np.roll(tri,-1,axis=0)):
            row=incident.get(conformal.edge_key(a,b),())
            if not row or abs(a[1]-b[1])>conformal.PLANE_ERROR:continue
            top=(a+b)*.5
            if (h:=top[1]-tri[:,1].min())<=conformal.PLANE_ERROR:continue
            fwd=mesh.N[i,XZ]; fwd=fwd/np.linalg.norm(fwd)
            for upper,_,_,face in row:
                # Orient into the upper road, not by a potentially reversed wall winding.
                p=surface.patches[upper]; interior=surface.triangles[face].mean(0)[XZ]-top[XZ]
                forward=fwd if interior@fwd>0. else -fwd
                foot=LineString([a[XZ],b[XZ]])
                for lower in index.query(foot.buffer(conformal.PLANE_ERROR)):
                    lower=int(lower)
                    if lower==upper:continue
                    p0=surface.patches[lower]
                    if p0.normal[1]<conformal.ANGLE:continue
                    if not (.0<top[1]-p0.height(top[XZ])<ph.WOOD.radius+EPS):continue
                    if abs(p0.height(top[XZ])-tri[:,1].min())>conformal.PLANE_ERROR:continue
                    if not p0.polygon.buffer(conformal.PLANE_ERROR).covers(foot):continue
                    key=(lower,upper,conformal.edge_key(a,b))
                    if key in seen:continue
                    seen.add(key)
                    out[lower].append(StepLip(lower,upper,a,b,forward,int(face),int(mesh.owner[i])))
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
    for lower, steps in lips.items():
        for step in steps:
            a,b=step.a,step.b; f=step.forward
            span=b[XZ]-a[XZ]; span2=float(span@span)
            if span2<=EPS:continue
            for u in by_patch[lower]:
                p=P[u,XZ]; x=float((p-a[XZ])@f); t=float((p-a[XZ])@span/span2)
                if not (-ph.WOOD.radius-coarse.STEP<=x<=0. and 0.<=t<=1.):continue
                pivot=a+t*(b-a); height=step.height(surface,pivot)
                if not 0.<height<ph.WOOD.radius:continue
                arm=math.sqrt(height*(2*ph.WOOD.radius-height))
                # Floor-node chord approaches the pivot and exits on its cap.
                if x > -arm+conformal.PLANE_ERROR:continue
                valid=[v for v in by_patch[step.upper] if 0.<=float((P[v,XZ]-pivot[XZ])@f)<=coarse.STEP+
                       ph.WOOD.radius and np.linalg.norm(P[v,XZ]-pivot[XZ])<=coarse.STEP+EPS]
                if not valid:continue
                v=min(valid,key=lambda v:np.linalg.norm(P[v,XZ]-pivot[XZ]))
                if any(q==v and how=='roll' for q,_,how in g.adj[u]):continue
                bits=0
                for k,ball in enumerate(balls):
                    state=step_verdict(ball,height,ball.radius,f)
                    if state==1:bits|=1<<k
                    if state==0:uncertain.append((u,v,ball.name,'step',height))
                masks[u,v,'step']=bits; reasons[u,v,'step']='step_torque'
                if not bits: continue
                if blocked(u,v):continue
                if not any(q==v and how=='step' for q,_,how in g.adj[u]):
                    g.adj[u].append((v,float(np.linalg.norm(P[v]-P[u]))+10.,'step')); counts['step']+=1
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
