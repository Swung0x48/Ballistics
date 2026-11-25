"""Held-key v14/v16 rail transfers for a finite shared seed catalogue."""
from collections import Counter,defaultdict
import math
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import closedform as cf, physics as ph, railride, supportcarry as sc
from .conformalsearch import Rail

EPS=1.e-7


def _trace(tf,paths,ball,dep,key,clock):
    st=dep.body
    if dep.kind in ('foldback','crown_twin'):return None,'unresolved_rail_transition'
    limit=cf.landing_step(ball,st,min(paths.obs.bottom-ball.radius-sc.SKIN,st.pos[1]-ball.radius))
    if not limit:return None,'rail_no_air_horizon'
    q=paths.air(tf,ball,dep,controls=lambda _:((float(limit),key),),clock=clock)
    if q is None:return None,'rail_no_result'
    if q['reason']!='land':return None,'rail_'+q['reason']
    if not dep.pieces and dep.kind!='rail_rebound':return None,'rail_empty_ride'
    if q['ticks']<=EPS or paths.obs.mesh.rail[q['face']]:return None,'rail_next_support_not_road'
    return q,'rail_road_landing'


def _append(row,dep,hit,key):
    ride=dep.pieces; flight=hit['segments']; suffix=(*ride,*flight); time0=row['ticks']; duration=dep.tick+hit['ticks']; end=time0+duration
    origin={} if 'source_node' not in row else dict(launch=dep.body.pos,velocity=dep.body.vel)
    return dict(dict(row,**origin),face=hit['face'],position=hit['state'].pos,ticks=end,flight_ticks=row['flight_ticks']+hit['ticks'],
                key_table=((math.ceil(end),key.copy()),),rail_parts=suffix,rail_mode=dep.kind,continued=True,
                motion=(*row['motion'],*(x for s in suffix for x in s['motion'])),
                contacts=(*row['contacts'],*(dict(x,tick=x['tick']+time0+dep.tick) for x in hit['contacts'])))


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
        entries.append(dict(source=src,ball=ball.name,kind=kind,interval=interval,position=c.body.pos,velocity=c.body.vel,
                            rail_chain=paths.lanes[c.lane].parent,rail_line=paths.line_for(c.lane),direction=c.direction))
        # Existing closed-form groove/crown walk, constant true key through
        # capture, release and flight. Do not invoke the old recursive search.
        departures=paths.walk(tf,ball,c,controls=lambda _:((math.inf,key),),clock=seed['ticks'],search_side=False,require_groove_stability=True)
        hits=0
        for dep in departures:
            hits+=1; q,why=_trace(tf,paths,ball,dep,key,seed['ticks']);stats[why]+=1
            if q is None:continue
            k=int(surface.face_patch[q['face']]); dest=layout.patch_nodes[ball.name].get(k)
            if dest is None or dest==src:continue
            row=_append(seed,dep,q,key);row['target_patch']=k;out[src,dest].append(row)
            if example is None and any(x['kind'] in ('rail','rail_crown') for x in dep.pieces):
                example=dict(entry=entries[-1],trajectory=row['rail_parts'],landing=row['position'],keys=row['key_table'])
        if not hits:stats['rail_no_supported_release']+=1
    for src,row in seeds:
        if not obs.mesh.rail[row['face']]:continue
        ball=balls[src.ball]; contact=row['contacts'][-1]; c=paths.capture(ball,contact,-1)
        if c is None:stats['rail_capture_unproved']+=1;continue
        lane=paths.lanes[c.lane]; j,s=c.j,c.direction; left=np.r_[0.,np.cumsum(lane.length)]
        u=float(np.clip((c.body.pos-lane.p[j])@lane.tangent[j],0.,lane.length[j])); arc=float(left[j]+u)
        if s<0:arc=float(left[-1]-arc)
        run(src,ball,row,c,row['key_table'][0][1],(arc,arc),'first_impact')
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
        if np.linalg.norm(g.pos[n]-point)>3.:stats['rail_anchor_unmapped']+=1;continue
        seed=dict(source_patch=-1,source_node=n,origin=point,launch=point,velocity=np.zeros(3),face=face,ticks=0.,flight_ticks=0.,
                  runup=0.,rim_ticks=0.,motion=(),contacts=())
        run(src,ball,seed,c,key,(arc,arc),'supported_groove_rest')
    stats.update(entries=len(entries),edges=len(out),seconds=perf_counter()-t0);log('shared rail: %s' % dict(stats))
    return out,dict(stats),entries,example
