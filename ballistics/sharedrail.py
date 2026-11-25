"""Held-key v14/v16 rail transfers for a finite shared seed catalogue."""
from collections import Counter,defaultdict
import math
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import closedform as cf, physics as ph, railride, supportcarry as sc
from .conformalsearch import Rail

EPS=1.e-7


def _trace(tf,paths,ball,dep,key,clock,*,rail_arrival=False):
    st=dep.body
    if dep.kind in ('foldback','crown_twin'):return None,'unresolved_rail_transition'
    limit=cf.landing_step(ball,st,min(paths.obs.bottom-ball.radius-sc.SKIN,st.pos[1]-ball.radius))
    if not limit:return None,'rail_no_air_horizon'
    q=paths.air(tf,ball,dep,controls=lambda _:((float(limit),key),),clock=clock)
    if q is None:return None,'rail_no_result'
    if q['reason']!='land':return None,'rail_'+q['reason']
    if not dep.pieces and dep.kind!='rail_rebound':return None,'rail_empty_ride'
    if q['ticks']<=EPS:return None,'rail_zero_contact'
    if paths.obs.mesh.rail[q['face']]:return (q if rail_arrival else None),'rail_next_support_not_road'
    return q,'rail_road_landing'


def _append(row,dep,hit,key):
    ride=dep.pieces; flight=hit['segments']; suffix=(*ride,*flight); time0=row['ticks']; duration=dep.tick+hit['ticks']; end=time0+duration
    origin={} if row['contacts'] else dict(launch=dep.body.pos,velocity=dep.body.vel)
    return dict(dict(row,**origin),face=hit['face'],position=hit['state'].pos,ticks=end,flight_ticks=row['flight_ticks']+hit['ticks'],
                key_table=((math.ceil(end),key.copy()),),rail_parts=suffix,rail_mode=dep.kind,continued=True,
                motion=(*row['motion'],*(x for s in suffix for x in s['motion'])),
                contacts=(*row['contacts'],*(dict(x,tick=x['tick']+time0+dep.tick) for x in hit['contacts'])))


def road_inlets(tf, paths, surface, layout, chains, balls, stats):
    """Supported end-on road entry, restricted to aligned constant-key runs.

    The road ray ends on its finite boundary at the fitted bearing station.
    The radius-offset road centre and rail centre must agree in the export
    band, and the original mesh must provide an upward rail reaction there.
    A skew approach or an unproved gap remains a candidate, not an edge.
    """
    from shapely import LineString, STRtree
    from . import conformal, conformalsearch as cs, motion, supportededges as se
    spans=[(k,a,b) for k,p in enumerate(surface.patches) for _,a,b in p.boundary
           if np.linalg.norm((a-b)[se.XZ])>EPS]
    lines=[LineString((a[se.XZ],b[se.XZ])) for _,a,b in spans]; index=STRtree(lines)
    obs=paths.obs; tol=min(conformal.PLANE_ERROR,sc.RECAPTURE)
    for (lane_id,direction,_end) in paths.endpoint:
        lane=paths.lanes[lane_id]; j=0 if direction>0 else len(lane.length)-1
        t=direction*lane.tangent[j]; up=railride.support_normal(t)
        for ball in balls.values():
            rail=Rail(ball.name,lane.parent,lane.line,-1,direction)
            if rail not in chains:continue
            r=ball.radius+sc.SKIN
            end=j if direction>0 else j+1
            center=lane.p[end] if lane.axis is None else lane.axis[end]+(r+lane.tube)*up
            foot=center-r*up
            # Nearby ends are only broad candidates; squared distance and
            # support are checked below. One edge can be tested by all balls.
            from shapely import Point
            for i in index.query(Point(foot[se.XZ]).buffer(tol)):
                k,a,b=spans[i]; p=surface.patches[k]; n=p.normal
                node=layout.patch_nodes[ball.name][k]
                if k not in layout.roads[node].rest_patches or n@up<conformal.ANGLE:
                    stats['inlet_road_support_unproved']+=1;continue
                xy=(center-r*n)[se.XZ]; d=(b-a)[se.XZ]; z=float(np.clip((xy-a[se.XZ])@d/(d@d),0.,1.))
                xy=a[se.XZ]+z*d; at=np.array([xy[0],p.height(xy),xy[1]])+r*n
                if np.linalg.norm(at-center)>tol+EPS:
                    stats['inlet_bearing_gap']+=1;continue
                outward=sc.unit(np.cross(b-a,n)); key=sc.keys_along(t); law=sc.law(ball,n,key); heading=sc.unit(law.impulse)
                # Avoid unmodeled transverse capture. The one allowed
                # normal projection is the supported plane-to-axis transfer.
                if heading@outward<=EPS or heading@t<1.-EPS:
                    stats['inlet_oblique_or_outward']+=1;continue
                depth=se._xy_depth(p,xy,heading[se.XZ]); scale=np.linalg.norm(heading[se.XZ]);
                if depth<=EPS or scale<=EPS:stats['inlet_no_runway']+=1;continue
                dist=min(depth*.5,ball.radius)/scale; start=ph.State(at-dist*heading,np.zeros(3))
                caught=law.along(heading).over(0.,dist)
                face=paths.geometry_support(center,ball,lane.double,t)
                if caught is None or face<0:stats['inlet_mesh_contact_unproved']+=1;continue
                legs=((start.pos,at),(at,center)); radius=ball.radius-sc.RECAPTURE
                if any(cs._capsule_blocked(tf,obs,a0,b0,radius,set(p.faces)) or any(
                            motion.box_hit(motion.line(a0,b0),lo,hi) is not None for lo,hi in obs.boxes)
                       for a0,b0 in legs):
                    stats['inlet_obstruction']+=1;continue
                speed,dt=caught; tick=dt/ph.DT; v=(heading@t)*speed*t
                crown=None if lane.axis is None else dict(supported=True,line=lane.line,contact='road_end',
                                                          angle=0.,lateral_speed=0.,lateral_offset=0.)
                c=railride.Catch(lane_id,j,direction,ph.State(center,v,ph.rolling_spin(v,up,ball.radius)),0.,face,crown=crown)
                seed=dict(source_patch=k,origin=start.pos,launch=center,velocity=v,face=face,ticks=tick,flight_ticks=0.,
                          runup=dist,rim_ticks=0.,motion=(motion.rolling(start,law.impulse,law.loss,tick),motion.line(at,center)),contacts=())
                stats['inlet_supported']+=1
                yield node,ball,seed,c,key,(0.,0.),'supported_road_end'


def mark_ring_witnesses(g, layout, witnesses):
    """Attach checked crossings to actual consecutive segment portals.

    An interior chain is a search node, not a certification of every ring.
    This pass records only uninterrupted, already swept rail motion on both
    sides of the same fitted ring. A ring with no such trace stays unknown.
    """
    from scipy.spatial import cKDTree
    from . import conformal
    indexed=defaultdict(list)
    for row in g.info.get('portal_verdicts',()):
        if row['kind']=='rail_ring':indexed[row['source'].ball].append(row)
    trees={name:cKDTree([r['segment'].mean(axis=0) for r in rows]) for name,rows in indexed.items()}
    total=0; rtol=conformal.PLANE_ERROR
    for (source,_),rows in witnesses.items():
        if source.ball not in trees:continue
        for row in rows:
            parts=row['rail_parts']
            for left,right in zip(parts,parts[1:]):
                if left['kind'] not in ('rail','rail_crown') or right['kind'] not in ('rail','rail_crown'):continue
                if left['object']!=right['object'] or left.get('rail_line')!=right.get('rail_line'):continue
                p=left['end_position']; q=right['start_position']
                if np.linalg.norm(p-q)>EPS:continue
                line=left.get('rail_line'); matches=[]
                for i in trees[source.ball].query_ball_point(p,rtol):
                    candidate=indexed[source.ball][i]; a,b=candidate['source'],candidate['target']
                    if isinstance(source,Rail) and (source.chain!=a.chain or source.line!=a.line or source.direction!=a.direction):continue
                    if line is not None and a.line!=line:continue
                    if line is None and a.line not in ('single','groove'):continue
                    if np.max(np.linalg.norm(candidate['segment']-p,axis=1))>rtol:continue
                    u,v=(sc.unit(np.diff(layout.rail_segments[k],axis=0)[0]) for k in (a,b))
                    if left['end_velocity']@u<=EPS or right['start_velocity']@v<=EPS:continue
                    if np.linalg.norm(np.cross(sc.unit(left['end_velocity']),u))>math.sqrt(1.-conformal.ANGLE**2):continue
                    if np.linalg.norm(np.cross(sc.unit(right['start_velocity']),v))>math.sqrt(1.-conformal.ANGLE**2):continue
                    matches.append(candidate)
                # Nearby overlapping rails can be indistinguishable at one
                # ring; do not claim the wrong tube/chain or both of them.
                if len(matches)==1 and matches[0]['status']!='verified':
                    matches[0].update(status='verified',reason='checked_rail_ring',
                                      witness=dict(incoming=left,outgoing=right,key_table=row['key_table']))
                    total+=1
    return total


def wave(tf,g,obs,layout,chains,seeds,*,log):
    """One rail carry per exact admitted contact; no recursive relay search.

    Also admit one supported terminal-piece representative per directed
    groove, from rest. It tests the outlet independently of how the groove
    was reached. No new rest equilibrium is asserted on a single crown.
    """
    t0=perf_counter(); surface=g.info['conformal']; out=defaultdict(list); stats=Counter(); example=None; entries=[]
    if not chains:return out,dict(entries=0,edges=0),entries,None
    # Landing nodes are projection only. RailPaths uses a nonnegative
    # receipt for a road landing; the actual road patch is taken from face.
    near=lambda f,_:0 if surface.face_patch[f]>=0 else -1
    paths=railride.RailPaths(g,obs,SimpleNamespace(near=near))
    balls={b.name:b for b in ph.BALLS.values()}
    def run(src,ball,seed,c,key,interval,kind):
        nonlocal example
        entry=dict(source=src,ball=ball.name,kind=kind,interval=interval,position=c.body.pos,velocity=c.body.vel,
                   rail_chain=paths.lanes[c.lane].parent,rail_line=paths.line_for(c.lane),direction=c.direction,
                   piece=c.j,key=key,outcomes=[])
        entries.append(entry)
        # Existing closed-form groove/crown walk, constant true key through
        # capture, release and flight. Do not invoke the old recursive search.
        departures=paths.walk(tf,ball,c,controls=lambda _:((math.inf,key),),clock=seed['ticks'],search_side=False,require_groove_stability=True)
        hits=0
        for dep in departures:
            hits+=1; q,why=_trace(tf,paths,ball,dep,key,seed['ticks']);stats[why]+=1
            entry['outcomes'].append(dict(reason=why,release=dep.body.pos,velocity=dep.body.vel,ticks=dep.tick,
                                          mode=dep.kind,landing=None if q is None else q['state'].pos,
                                          landing_face=None if q is None else q['face']))
            if q is None:continue
            k=int(surface.face_patch[q['face']]); dest=layout.patch_nodes[ball.name].get(k)
            if dest is None or dest==src:continue
            row=_append(seed,dep,q,key);row['target_patch']=k;out[src,dest].append(row)
            if example is None and any(x['kind'] in ('rail','rail_crown') for x in dep.pieces):
                example=dict(entry=entries[-1],trajectory=row['rail_parts'],landing=row['position'],keys=row['key_table'])
        if not hits:
            stats['rail_no_supported_release']+=1
            entry['outcomes'].append(dict(reason='rail_no_supported_release'))
    for src,row in seeds:
        if not obs.mesh.rail[row['face']]:continue
        ball=balls[src.ball]; contact=row['contacts'][-1]; c=paths.capture(ball,contact,-1)
        if c is None:stats['rail_capture_unproved']+=1;continue
        lane=paths.lanes[c.lane]; j,s=c.j,c.direction; left=np.r_[0.,np.cumsum(lane.length)]
        u=float(np.clip((c.body.pos-lane.p[j])@lane.tangent[j],0.,lane.length[j])); arc=float(left[j]+u)
        if s<0:arc=float(left[-1]-arc)
        run(src,ball,row,c,row['key_table'][0][1],(arc,arc),'first_impact')
    for args in road_inlets(tf,paths,surface,layout,chains,balls,stats):
        run(*args)
    for src,chain in chains.items():
        if src.line!='groove':continue
        ball=balls[src.ball]; lane=paths.lanes.get(src.chain)
        if lane is None or not lane.double:continue
        j=len(lane.length)-1 if src.direction>0 else 0
        a,b=lane.p[j:j+2]; point=(a+b)*.5+sc.SKIN*railride.support_normal(lane.tangent[j]);t=src.direction*lane.tangent[j]
        face=paths.geometry_support(point,ball,True,t)
        if face<0:stats['rail_rest_not_supported']+=1;continue
        key=sc.keys_along(t); up=railride.support_normal(t)
        if sc.law(ball,up,key).impulse@t<=EPS:stats['rail_rest_no_forward_thrust']+=1;continue
        arc=float(chain.knots[-1]-lane.length[j]*.5); c=railride.Catch(src.chain,j,src.direction,ph.State(point,np.zeros(3)),0.,face)
        # This seed is a local road-graph anchor, not an invented landing
        # impulse. Mapping to the coarse groove happens only on publication.
        ix=np.asarray(lane.ids); n=int(ix[np.linalg.norm(g.pos[ix]-point,axis=1).argmin()])
        if np.linalg.norm(g.pos[n]-point)>3.:
            stats['rail_anchor_unmapped']+=1;n=-1  # mapping is not a condition on support
        seed=dict(source_patch=-1,source_node=n,origin=point,launch=point,velocity=np.zeros(3),face=face,ticks=0.,flight_ticks=0.,
                  runup=0.,rim_ticks=0.,motion=(),contacts=())
        run(src,ball,seed,c,key,(arc,arc),'supported_groove_rest')
    stats.update(entries=len(entries),edges=len(out),seconds=perf_counter()-t0);log('shared rail: %s' % dict(stats))
    return out,dict(stats),entries,example


def inverse_outlets(tf,g,obs,layout,chains,targets):
    """Try target polygons from the entire final straight groove, not a midpoint.

    Hold a literal key on the rail. Another key is allowed only if the
    inverse admits an integer-tick outlet. Moving inlets remain separate.
    """
    from . import railoutlet, supportededges as se
    surface=g.info['conformal']; stats=Counter(); found=defaultdict(list); attempts=[]
    if not targets:return found,attempts,{}
    paths=railride.RailPaths(g,obs,SimpleNamespace(near=lambda f,_:0 if surface.face_patch[f]>=0 else -1))
    balls={b.name:b for b in ph.BALLS.values()}
    for src,patches in targets.items():
        if src not in chains or src.line!='groove':continue
        lane=paths.lanes.get(src.chain)
        if lane is None or not lane.double:continue
        ball=balls[src.ball]; j=len(lane.length)-1 if src.direction>0 else 0
        tangent=src.direction*lane.tangent[j]; up=railride.support_normal(tangent)
        end=lane.p[j+1 if src.direction>0 else j]+sc.SKIN*up
        length=lane.length[j]; total=float(chains[src].knots[-1]); arc_j=(total-length,total)
        for key in se.KEYS:
            law=sc.law(ball,up,key).along(tangent); top=law.over(0.,length)
            if top is None:continue
            for release_key in (*se.KEYS,np.zeros(2)):
                switch=not np.array_equal(key,release_key)
                for k in patches:
                    patch=surface.patches[k]; stats['target_polygons']+=1
                    for speeds,_,xz in railoutlet.band(ball,end,tangent,release_key,patch,(0.,top[0])):
                        dist=railoutlet.runway_interval(law,length,speeds)
                        if dist is None:continue
                        lo,hi,run=dist; arc=(total-hi,total-lo); switch_tick=None
                        if switch:
                            representative=railoutlet.integer_runway(law,speeds,length)
                            if representative is None:
                                stats['outlet_no_integer_switch']+=1;continue
                            run,switch_tick=representative
                        p=end-run*tangent
                        outcome=dict(source=src,target_patch=k,arc_interval=arc,speed_interval=speeds,
                                     finite_segment=arc_j,landing_segment=xz,key=key,release_key=release_key,switch_tick=switch_tick,origin=p,reason='outlet_runway_support_unproved')
                        attempts.append(outcome); stats['outlet_intervals']+=1
                        face=paths.geometry_support(p,ball,True,tangent)
                        if face<0:continue
                        body=ph.State(p,np.zeros(3)); catch=railride.Catch(src.chain,j,src.direction,body,0.,face)
                        outcome['reason']='outlet_held_support_unproved'
                        for dep in paths.walk(tf,ball,catch,controls=lambda _:((math.inf,key),),search_side=False,require_groove_stability=True):
                            if switch and (dep.kind!='chain_end' or abs(dep.tick-switch_tick)>EPS):
                                outcome['reason']='outlet_integer_event_mismatch';continue
                            hit,why=_trace(tf,paths,ball,dep,release_key,0.,rail_arrival=True); stats[why]+=1
                            outcome.update(reason=why,release=dep.body.pos,velocity=dep.body.vel,release_tick=dep.tick)
                            if hit is None:continue
                            seed=dict(source_patch=-1,source_node=-1,origin=p,launch=end,velocity=dep.body.vel,
                                      face=face,ticks=0.,flight_ticks=0.,runup=run,rim_ticks=0.,motion=(),contacts=())
                            prefix=_append(seed,dep,hit,release_key)
                            def keep(road,trace):
                                actual=int(surface.face_patch[road['face']]); dest=layout.patch_nodes[ball.name][actual]
                                if switch:trace['key_table']=((switch_tick,key.copy()),(math.ceil(trace['ticks']),release_key.copy()))
                                trace.update(target_patch=actual,arc_interval=arc,inverse=True)
                                found[src,dest].append(trace)
                                outcome.update(reason='literal_target' if actual==k else 'literal_other_patch',actual_patch=actual,landing=road['state'].pos)
                            if why=='rail_road_landing':keep(hit,prefix);continue
                            # The solved outlet may end on a different rail.
                            # Preserve the actual contact and try one direct,
                            # same-key carry from that *same* captured state.
                            other=hit.get('catch'); outcome.update(first_rail_face=hit['face'],first_rail_contact=hit['state'].pos)
                            if other is None:continue
                            outcome['next_rail_line']=paths.line_for(other.lane)
                            outcome['second_outlets']=[]
                            for nxt in paths.walk(tf,ball,other,controls=lambda _:((math.inf,release_key),),clock=prefix['ticks'],
                                                  search_side=False,require_groove_stability=True):
                                road,code=_trace(tf,paths,ball,nxt,release_key,prefix['ticks']); stats['relay_'+code]+=1
                                outcome['second_outlets'].append(dict(reason=code,position=nxt.body.pos,velocity=nxt.body.vel))
                                if road is None:continue
                                trace=_append(prefix,nxt,road,release_key)
                                trace['rail_parts']=(*prefix['rail_parts'],*trace['rail_parts'])
                                keep(road,trace)
    return found,attempts,dict(stats)


def outlet_priorities(g,chains,points):
    """Resolve optional project target anchors; the inverse uses polygons."""
    targets=defaultdict(set)
    if not points:return targets
    support=g.info['terrain_support']
    for row in points:
        u=g.nearest(np.asarray(row['near']),3.,floor_only=False); v=g.nearest(np.asarray(row['far']),3.)
        if u<0 or not 0<=v<len(support) or g.lane[u]<0:continue
        for node in chains:
            if node.chain==g.lane[u] and node.ball==row.get('ball',node.ball):targets[node].update(support[v])
    return targets
