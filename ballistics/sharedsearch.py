"""Default node-keyed launch catalogue and literal runway representatives.

A relaxed connection is not inserted into the drawable/state graph as a
witness. Geometry alone also does not establish chaining two witnesses.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product
import math
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from shapely import LineString, MultiPoint, Polygon, STRtree, box, union_all
from shapely.affinity import translate

from . import closedform as cf, conformal, motion, physics as ph, rimclear, supportcarry as sc
from .conformalsearch import Road, Rail, _rail_normal

XZ=[0,2]
EPS=1.e-7
KEYS=np.asarray(tuple(product((-1.,0.,1.),repeat=2)))


@dataclass(frozen=True)
class Chain:
    node: Rail             # piece=-1, distinct from a v23 piece key
    pieces: tuple          # ordered in direction of travel
    knots: np.ndarray      # arc length, no coarse lattice subdivisions
    segments: np.ndarray  # centre lines in the same order


def chain_nodes(layout):
    groups=defaultdict(list); mapped={}; chains={}
    for node in layout.rail_segments:
        chain=Rail(node.ball,node.chain,node.line,-1,node.direction)
        groups[chain].append(node); mapped[node]=chain
    # A v23 chain/line is one oriented, unbranched mesh polyline. Inter-chain
    # junctions remain graph portals; they must not be hidden by this union.
    for node,parts in groups.items():
        members=tuple(sorted(parts,key=lambda k:node.direction*k.piece))
        spans=np.stack([layout.rail_segments[k] for k in members]); lengths=np.linalg.norm(spans[:,1]-spans[:,0],axis=1)
        chains[node]=Chain(node,members,np.r_[0.,np.cumsum(lengths)],spans)
    return chains,mapped


def minkowski_segment(region, displacement):
    """Exact 2D polygon + [0,d], including nonconvex parts and holes.

    Each translated point is inside an endpoint polygon or its translation
    segment crosses a boundary. The latter contribution is the ruled quad of
    one polygon edge. Nothing here samples start positions or the segment.
    """
    if not np.any(displacement):return region
    dx,dz=displacement
    swept=[region,translate(region,dx,dz)]
    for p in conformal.polygon_parts(region):
        for ring in (p.exterior,*p.interiors):
            for a,b in zip(ring.coords,list(ring.coords)[1:]):
                u,v=np.asarray(a),np.asarray(b)
                swept.append(MultiPoint((u,v,u+displacement,v+displacement)).convex_hull)
    return union_all(swept)


def rest_section(patch,ball,ticks,polygon=None):
    """Projected terminal position of the fixed-time joint rest image.

    [DERIVED] p0 is free in the entire finite starting polygon; u has the
    v21 geometric weights. This relaxed image does not assert containment
    at intermediate times. The conditional velocity uses the same u.
    """
    from .runreach import PlaneReach, sums
    n=patch.normal; r=ball.radius+sc.SKIN
    R=PlaneReach.make(ball,patch,np.zeros(3),np.zeros(3)); p,v=R.center(ticks); A,B=sums(ticks,R.loss)
    u,w=(A*R.axes[XZ]).T; drift=(p-R.origin)[XZ]+r*n[XZ]
    poly=patch.polygon if polygon is None else polygon
    poly=minkowski_segment(minkowski_segment(translate(poly,*(drift-u-w)),2*u),2*w)
    return poly,dict(ticks=ticks,drift=drift,position_axes=R.axes*A,velocity_center=v,velocity_axes=R.axes*B)


def _lines(item):
    if item.geom_type=='LineString':
        yield item
    elif hasattr(item,'geoms'):
        for g in item.geoms:yield from _lines(g)


def _lip_track(tf,ball,patch,obs,f,a,b,*,bend=None,on_release=None):
    """One finite literal representative, from a whole rest patch.

    From rest a held key has a fixed tangent. Intersect its backwards ray
    with the patch, choose the midpoint of the connected inlet interval, and
    solve the rolling boundary time. Hold the SAME key through the whole rim
    and flight; no key ever switches during the rim or at a fractional PSI.
    """
    n=patch.normal; r=ball.radius+sc.SKIN; mid=(a+b)*.5; mid[1]=patch.height(mid[XZ]); axis=sc.unit(b-a); out=sc.unit(np.cross(axis,n))
    laws=tuple(sc.law(ball,n,k) for k in KEYS)
    i=int(np.argmax([l.impulse@out for l in laws])); law=laws[i]; strength=np.linalg.norm(law.impulse)
    if strength<=EPS or law.impulse@out<=EPS:return None,'no_outward_key'
    d=law.impulse/strength; scale=np.linalg.norm(d[XZ]); width=float(np.linalg.norm((patch.high-patch.low)[XZ])); start=mid[XZ]
    ray=LineString((start,start-d[XZ]/scale*(width+conformal.PLANE_ERROR)))
    intervals=[]
    for seg in _lines(ray.intersection(patch.polygon.buffer(EPS,quad_segs=1))):
        u=sorted(float((np.asarray(x)-start)@(-d[XZ])/scale) for x in (seg.coords[0],seg.coords[-1]))
        if u[0]<=2*EPS and u[1]>2*EPS:intervals.append(u[1])
    if not intervals:return None,'no_local_runway'
    distance=max(intervals)*.5/scale; origin=mid+r*n-distance*d
    result=law.along(d).over(0.,distance)
    if result is None:return None,'no_crossing'
    speed,seconds=result; ticks=seconds/ph.DT; start=ph.State(origin,np.zeros(3)); push=KEYS[i]
    # Use the fitted face normal but the finite span from the original mesh.
    rim=tf.rim_release(ball,mid,n,out,speed*d,0.,push=push,inertia_ratio=sc.inertia_ratio(ball))
    if rim is None:return None,'no_release'
    if bend is not None and rim.angle>=bend-EPS:return None,'convex_seam_rolls'
    if not obs.support_clear(start,law.impulse,law.loss,ticks,int(obs.mesh.owner[f]),ball.radius-sc.RECAPTURE):
        return None,'runway_blocked'
    state,detail=rimclear.check(tf,ball,rim.motion(),obs,(a,b))
    if state!='clear':return None,'rim_'+state
    st=rim.start
    if not rim.angle:st=ph.State(st.pos+8*EPS*(rim.outward+rim.normal),st.vel,st.omega)
    if on_release is not None:
        on_release(rim,st,ticks+rim.seconds/ph.DT,push,distance,(motion.rolling(start,law.impulse,law.loss,ticks),rim.motion()))
    limit=cf.landing_step(ball,st,min(obs.bottom-r,st.pos[1]-r))
    if not limit:return None,'no_air_horizon'
    air=tf.trace_air(obs,ball,st,((float(limit),push),),rim=rim)
    why,face,flight,at=air.result
    if why!='land' or flight<=EPS:return None,'air_'+why
    all_ticks=ticks+rim.seconds/ph.DT+flight
    geom=(motion.rolling(start,law.impulse,law.loss,ticks),rim.motion(),*(tf.air_geometry(ball,p,k,t) for p,k,t in air.parts))
    return dict(face=int(face),source_face=int(f),origin=origin,position=at,launch=st.pos,velocity=st.vel,
                flight_ticks=flight,ticks=all_ticks,key_table=((math.ceil(all_ticks),push.copy()),),
                motion=geom,contacts=air.contacts,runup=distance,rim_ticks=rim.seconds/ph.DT), 'first_support'


def release_features(surface,layout):
    """Free lips and welded convex lips, in their supported outward chart.

    Shared-edge membership alone does not imply continued support. At a
    convex break the press must reach zero before the next plane's normal.
    At a nonmanifold span this also has to precede *all* bearing neighbours.
    """
    out={k:[(f,a,b,None) for f,a,b in rows] for k,rows in layout.boundary.items()}
    next_normals=defaultdict(list)
    for seam in layout.seams:
        a,b=seam.endpoints; key=conformal.edge_key(a,b)
        i,j=seam.patches; next_normals[i,key].append(j); next_normals[j,key].append(i)
    for k,p in enumerate(surface.patches):
        for f,a,b in p.boundary:
            other=next_normals.get((k,conformal.edge_key(a,b)),())
            if not other:continue
            outward=sc.unit(np.cross(b-a,p.normal)); ns=[surface.patches[j].normal for j in other]
            if all(n@outward>EPS for n in ns):
                angle=min(math.acos(float(np.clip(n@p.normal,-1.,1.))) for n in ns)
                out[k].append((f,a,b,angle))
    return out


def _terrain_speed(ball,normals):
    return max((np.linalg.norm(l.impulse)/l.loss for n in normals for k in KEYS for l in (sc.law(ball,n,k),)),default=0.)


def _air_candidates(tf, ball, lower, upper, speed, index):
    """Safe box outer screen for the declared bounded launch chart.

    y uses the highest origin and upward velocity extreme. A,B are monotone
    on the descending root; both horizontal extrema use the same upper B.
    Actual target footprints (not rest points) intersect the outer square.
    """
    geometry, heights, tree, owners=index
    if not len(heights):return ()
    loss=ball.lin_damp*ph.DT
    # DERIVED: loss opposes upward motion, so the zero-damping apex plus
    # one velocity chord covers the largest sample/continuous apex.
    top=upper[1]+speed**2/(2*ph.GRAVITY)+speed*ph.DT
    valid=np.flatnonzero(heights<=top+EPS)
    if not len(valid):return ()
    st=ph.State(np.array([0.,upper[1],0.]),np.array([0.,speed,0.]))
    # _target_coeff handles only targets actually below the apex; use
    # monotonicity and the launch-height descending root for higher targets.
    A,B=tf._target_coeff(ball,st,np.minimum(upper[1],heights[valid])); span=np.asarray(A)*speed+np.asarray(B)*ball.push
    lo=lower[XZ]-span[:,None]; hi=upper[XZ]+span[:,None]
    ext=geometry[valid]
    valid=valid[np.all((ext[:,:2]<=hi)&(ext[:,2:]>=lo),axis=1)]
    if not len(valid):return ()
    # Geometry is queried only for surviving AABB pairs. Redo the exact
    # coefficient for this subset instead of storing a dense region matrix.
    A,B=tf._target_coeff(ball,st,np.minimum(upper[1],heights[valid])); span=np.asarray(A)*speed+np.asarray(B)*ball.push
    import shapely
    squares=shapely.box(*(np.column_stack((lower[XZ]-span[:,None],upper[XZ]+span[:,None])).T))
    hit=shapely.intersects(squares,tree.geometries[valid])
    return tuple(set(owners[j] for j in valid[hit]))


def _build_index(surface,layout,chains,ball):
    shifts=[]; heights=[]; owners=[]
    r=ball.radius+sc.SKIN
    for k,p in enumerate(surface.patches):
        poly=translate(p.polygon,*(r*p.normal[XZ])); shifts.append(poly)
        heights.append(p.heights(r)[0]); owners.append(layout.patch_nodes[ball.name][k])
    # Dilation bounds the support strip of the fitted centre line; the mesh
    # must still confirm which tube(s) are carrying a real landing.
    for node,c in chains.items():
        if node.ball!=ball.name:continue
        for span in c.segments:
            lo=span.min(0); hi=span.max(0)
            shifts.append(box(*(np.r_[lo[XZ]-r,hi[XZ]+r]))); heights.append(lo[1]-r); owners.append(node)
    return np.asarray([p.bounds for p in shifts]).reshape(-1,4),np.asarray(heights),STRtree(shifts),owners


def _mapping(g,layout,surface):
    # Ground graph membership is consulted only after an actual trajectory.
    by_patch=defaultdict(list)
    for i,ks in enumerate(g.info.get('terrain_support',())):
        for k in ks:by_patch[k].append(i)
    tables={k:(np.asarray(ids),cKDTree(g.pos[ids])) for k,ids in by_patch.items()}
    def pick(patch,pos):
        row=tables.get(patch)
        if row is None:return -1
        ids,tree=row; d,i=tree.query(pos)
        return int(ids[i]) if d<=3 else -1  # same mapping reach as LandingPlaces
    return pick


def display_way(tf,pieces,source,target):
    """Display only: the geometry guards consume pieces, never this polyline."""
    p=[source]
    for item in pieces:
        lo,hi=item.bounds(0.,item.end)
        count=max(1,math.ceil(np.linalg.norm(hi-lo)))
        p.extend(item.point(t) for t in np.linspace(0.,item.end,count+1))
    return tf.densify_way([*p,target])



def add(tf, g,obs,layout, *, log=lambda *_:None, target_priorities=()):
    """Generate and propagate the bounded v24 shared catalogue.

    One union edge per ordered node pair. Unverified edges stay in the
    candidate graph, never become playable edges in the legacy projection.
    """
    surface=g.info['conformal']; t0=perf_counter(); chains,parts=chain_nodes(layout)
    features=release_features(surface,layout)
    from .targetinverse import Priorities
    priority=Priorities(layout,surface,g,target_priorities)
    stats=Counter(); candidates=defaultdict(set); witnesses=defaultdict(list); envelope={}; example=None; momentum_seeds=[]
    rest={node for node,row in layout.roads.items() if row.rest_patches}
    by_ball={b.name:b for b in ph.BALLS.values()}
    # Risky or steep seams are candidate portals, not certified rolling.
    for p in layout.portals:
        a,b=parts.get(p.source,p.source),parts.get(p.target,p.target)
        if a!=b:candidates[a].add(b);stats['geometric_portals']+=1
    for name,ball in by_ball.items():
        index=_build_index(surface,layout,chains,ball)
        for node,row in layout.roads.items():
            if node.ball!=name or not row.rest_patches:continue
            lows=[]; highs=[]
            for k in row.rest_patches:
                p=surface.patches[k]
                for f,a,b,bend in features[k]:
                    # Retain every finite launch span as an outer candidate;
                    # these are mesh boundary features, not coarse headings.
                    r=ball.radius+sc.SKIN; low=np.minimum(a,b)+r*p.normal; high=np.maximum(a,b)+r*p.normal
                    lows.append(low);highs.append(high)
                    stats['lip_features']+=1
                    inverse=(lambda *args:priority.from_release(tf,ball,obs,node,k,*args)) if node in priority.points else None
                    result,why=_lip_track(tf,ball,p,obs,f,a,b,bend=bend,on_release=inverse);stats[why]+=1
                    if result is None:continue
                    result['source_patch']=k
                    momentum_seeds.append((node,result))
                    k2=int(surface.face_patch[result['face']]); other=layout.patch_nodes[name].get(k2)
                    if other is None:
                        stats['unsupported_motion_transfer']+=1
                        continue
                    if other==node:continue
                    result.update(source_patch=k,target_patch=k2)
                    witnesses[node,other].append(result); candidates[node].add(other)
                    if example is None:
                        section,meta=rest_section(p,ball,max(1,math.ceil(result['ticks']-result['flight_ticks']-result['rim_ticks'])))
                        hit=section.intersection(LineString((a[XZ]+r*p.normal[XZ],b[XZ]+r*p.normal[XZ])))
                        from shapely.geometry import mapping
                        example=dict(source=node,patch=k,region_rest_patches=row.rest_patches,origin_polygon=mapping(p.polygon),
                                     finite_span=np.stack((a,b)),position_family=mapping(section),intersection=mapping(hit),joint=meta)
            if not lows:continue
            lo=np.min(lows,axis=0)-2*(ball.radius+sc.SKIN);hi=np.max(highs,axis=0)+2*(ball.radius+sc.SKIN)
            speed=_terrain_speed(ball,(surface.patches[k].normal for k in row.rest_patches))
            # ASSUMED scope of the finite candidate template: local rest
            # run-up and up to one lip rotation. Rim work bound is from its
            # effective potential difference, at most 2r|a|.
            force=math.hypot(ph.GRAVITY,math.sqrt(2)*ball.push/ph.DT)
            speed=math.sqrt(speed**2+4*(ball.radius+sc.SKIN)*force)
            envelope[node]=(lo,hi,speed)
            candidates[node].update(x for x in _air_candidates(tf,ball,lo,hi,speed,index) if x!=node)
        for node,chain in chains.items():
            if node.ball!=name:continue
            # ASSUMED chart for unresolved releases on this chain. This
            # deliberately includes its complete arc span; it is not the
            # v14/v16 inverse and does not certify pressure/side release.
            rail_radius=ball.radius+sc.SKIN; pts=chain.segments.reshape(-1,3)
            lo=pts.min(0)-2*rail_radius; hi=pts.max(0)+2*rail_radius
            normals=(_rail_normal(sc.unit(b-a)) for a,b in chain.segments)
            speed=_terrain_speed(ball,normals)
            lift=math.hypot(ph.GRAVITY,math.sqrt(2)*ball.push/ph.DT)
            speed=math.sqrt(speed**2+4*rail_radius*lift)
            envelope[node]=(lo,hi,speed)
            candidates[node].update(x for x in _air_candidates(tf,ball,lo,hi,speed,index) if x!=node)
        log('shared launches %s: %d candidate pairs, %d first-support witnesses' %
            (name,sum(len(v) for k,v in candidates.items() if k.ball==name),sum(len(v) for (k,_),v in witnesses.items() if k.ball==name)))
    for a,b,row in priority.rows:
        witnesses[a,b].append(row); candidates[a].add(b)
    log('target inverse: %d attempts, %d literal witnesses' % (len(priority.attempts),len(priority.rows)))
    from . import momentum
    entries,continuations,momentum_stats,momentum_example=momentum.wave(tf,g,obs,layout,chains,momentum_seeds,log=log)
    for (a,b),rows in continuations.items():
        witnesses[a,b].extend(rows); candidates[a].add(b)
    # One fixed label per destination support chart in this relaxation.
    # Each arrival is a tag for the outer template, not a realized state.
    # Node+role is the key, never the centre or raw velocity.
    queue=deque(sorted(rest)); expanded=set(); admitted=set(rest); covered=0
    while queue:
        a=queue.popleft();expanded.add(a)
        for b in candidates.get(a,()):
            if b in admitted:covered+=1;continue
            admitted.add(b);queue.append(b)
    pick=_mapping(g,layout,surface); projected=defaultdict(dict)
    for (src,dst),rows in witnesses.items():
        for row in rows:
            u=pick(row['source_patch'],row['origin']);v=pick(row['target_patch'],row['position'])
            if u<0 or v<0 or u==v:continue
            # Mapping is display/route compatibility, not an alternative
            # origin from which to compute the representative again.
            old=projected[u,v].get(src.ball)
            if old is not None and old['ticks']<=row['ticks']:continue
            edge=dict(site=u,ball=src.ball,ticks=row['ticks'],time=row['ticks']/132.,flight_ticks=row['flight_ticks'],
                      verified=True,witness_status='literal_confirmed',representative='shared_literal_momentum' if row.get('continued') else 'shared_literal_inverse' if row.get('inverse') else 'shared_literal_first_support',launch_position=row['launch'],launch_velocity=row['velocity'],
                      launch_speed=float(np.linalg.norm(row['velocity'])),landing_position=row['position'],landing_face=row['face'],
                      first_landing_node=v,motion=row['motion'],contacts=row['contacts'],way=display_way(tf,row['motion'],g.pos[u],g.pos[v]),
                      across=float(np.linalg.norm((row['position']-row['launch'])[XZ])),down=float(row['launch'][1]-row['position'][1]),
                      key_table=row['key_table'],keys_in_air=row['key_table'][-1][1],air_inputs=tuple(x[1] for x in row['key_table']),
                      switch_tick=row['key_table'][0][0] if len(row['key_table'])>1 else 0,
                      runup=row['runup'],rim_ticks=row['rim_ticks'],bounce_count=0,rail_support_prefix_ticks=0.)
            projected[u,v][src.ball]=edge
    launches={}
    for (u,v),by_kind in projected.items():
        rows=tuple(sorted(by_kind.values(),key=lambda e:e['ticks'])); launches[u,v]=dict(rows[0],variants=rows)
        g.adj[u].append((v,tf.cost(rows[0]),'launch'))
    g.info['launches']=launches
    g.info['shared_motion_graph']=dict(chains=chains,segments_to_chain=parts,candidates={a:frozenset(bs) for a,bs in candidates.items()},
                                       witnesses=dict(witnesses),rest=rest,release_outer=envelope,example=example,
                                       momentum_entries=entries,momentum_example=momentum_example,target_inverse=priority.attempts)
    detail=dict(stats,rail_nodes_before=len(parts),rail_nodes_after=len(chains),road_nodes=len(layout.roads),
                candidate_edges=sum(map(len,candidates.values())),witness_edges=len(witnesses),projected_edges=len(launches),
                expanded_abstract_labels=len(expanded),covered_abstract_labels=covered,live_joint_entrance_families=len(entries),momentum=momentum_stats,inverse_attempts=len(priority.attempts),inverse_witnesses=len(priority.rows),seconds=perf_counter()-t0)
    g.info['takeoff_stats']=detail;log('shared takeoff: %s' % detail)
