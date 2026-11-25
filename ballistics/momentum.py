"""Finite, replay-backed momentum catalogue for the shared motion graph."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
import math

import numpy as np

from . import closedform as cf, conformal, physics as ph, rimclear, supportcarry as sc
from .conformalsearch import Rail

EPS=1.e-7


@dataclass(frozen=True)
class Entry:
    node: object
    patch: int
    source: object
    state: np.ndarray     # one coupled p,v point, never two independent boxes
    tick: float
    keys: np.ndarray
    support_interval: tuple | None = None


def _span(curve, surface):
    best=None
    for p in surface.patches:
        if abs(p.normal@curve.v/np.linalg.norm(curve.v)-1.)>EPS or np.any(curve.p<p.low-conformal.PLANE_ERROR) or np.any(curve.p>p.high+conformal.PLANE_ERROR):continue
        for _,a,b in p.boundary:
            d=b-a; z=d@d
            if z<=EPS**2:continue
            at=a+np.clip((curve.p-a)@d/z,0,1)*d
            dist=float(np.linalg.norm(at-curve.p))
            if best is None or dist<best[0]:best=(dist,a,b)
    return None if best is None or best[0]>conformal.PLANE_ERROR else best[1:]


def _rail_entry(ball, chains, c):
    p=np.asarray(c['position']); v=np.asarray(c['incoming']); best=None
    for node,s in chains.items():
        if node.ball!=ball.name:continue
        a,b=s.segments[:,0],s.segments[:,1]; d=b-a
        norm=np.einsum('ij,ij->i',d,d)
        z=np.clip(np.einsum('ij,ij->i',p-a,d)/np.maximum(norm,EPS**2),0.,1.)
        q=a+d*z[:,None]; dist=np.linalg.norm(q-p,axis=1)
        dist[d@v<=0.]=np.inf
        j=int(np.argmin(dist))
        if dist[j]<=ball.radius and (best is None or dist[j]<best[0]):
            s0=s.knots[j]+z[j]*(s.knots[j+1]-s.knots[j]); best=(float(dist[j]),node,(float(s0),float(s0)))
    return None if best is None else best[1:]


def _road_continuation(tf,paths, ball, row, surface):
    """One supported event sequence and at most one additional free flight.

    The same literal key is held from the original runway. All sub-tick
    events are geometric contacts, never key changes. Do not infer that
    another representative with a similar incoming velocity can replay it.
    """
    hit=row['contacts'][-1]; key=row['key_table'][0][1]
    support=paths.impact(tf,ball,hit,-1,chosen_push=key)
    if support.reason not in ('floor_end','convex','rebound'):
        return None,'support_'+support.reason,support
    if support.reason!='rebound' and support.rim is None:return None,'no_supported_release',support
    for seg in support.parts:
        for curve in seg['motion']:
            if curve.kind!='arc':continue
            span=_span(curve,surface)
            if span is None:return None,'carry_rim_support_unmatched',support
            code,_=rimclear.check(tf,ball,curve,paths.obs,span)
            if code!='clear':return None,'carry_rim_'+code,support
    if support.rim is not None and not support.rim.angle:
        p=support.rim.start
        st=ph.State(p.pos+8*EPS*(support.rim.normal+support.rim.outward),p.vel,p.omega)
    else:st=support.body
    lim=cf.landing_step(ball,st,min(paths.obs.bottom-ball.radius-sc.SKIN,st.pos[1]-ball.radius))
    if not lim:return None,'carry_no_air_horizon',support
    air=tf.trace_air(paths.obs,ball,st,((float(lim),key),),rim=support.rim,incoming_only=True)
    why,f,dt,p=air.result
    if why!='land' or dt<=EPS:return None,'carry_air_'+why,support
    elapsed=sum(x['ticks'] for x in support.parts); end=row['ticks']+elapsed+dt
    body=tuple(x for s in support.parts for x in s['motion'])
    arcs=tuple(tf.air_geometry(ball,st,k,t) for st,k,t in air.parts)
    return dict(row,face=int(f),position=p,ticks=end,flight_ticks=row['flight_ticks']+dt,
                key_table=((math.ceil(end),key.copy()),), motion=(*row['motion'],*body,*arcs),
                contacts=(*row['contacts'],*(dict(x,tick=x['tick']+row['ticks']+elapsed) for x in air.contacts)),
                continued=True),'second_support',support


def wave(tf,g,obs,layout,chains,seeds,*,log):
    """Each member of the frozen first-support catalogue is admitted once.

    Finite because this wave does not enqueue its own output. A newly
    produced exact state is distinct even at the same patch/velocity box.
    """
    start=perf_counter(); surface=g.info['conformal']; paths=sc.SurfacePaths(g,obs,SimpleNamespace(near=lambda *_:-1))
    balls={b.name:b for b in ph.BALLS.values()}; stats=Counter(); entries=[]; new=defaultdict(list); example=None
    for src,row in seeds:
        ball=balls[src.ball]; hit=row['contacts'][-1]; k=int(surface.face_patch[hit['triangle']]); rail=None
        if k<0 and obs.mesh.rail[hit['triangle']]:rail=_rail_entry(ball,chains,hit)
        node=layout.patch_nodes[src.ball].get(k) if rail is None else rail[0]
        if node is None:stats['unmapped_entry']+=1;continue
        family=Entry(node,k,src,np.stack((hit['position'],hit['incoming'])),row['ticks'],row['key_table'][0][1],None if rail is None else rail[1]); entries.append(family)
        if isinstance(node,Rail):stats['rail_entry_unexpanded']+=1;continue
        result,why,support=_road_continuation(tf,paths,ball,row,surface); stats[why]+=1
        if example is None and support.parts:
            example=dict(entry=family,scope='exact_literal_prefix_singleton',reason=why,supported=support.parts,
                         landing=None if result is None else result['position'])
        if result is None:continue
        k=int(surface.face_patch[result['face']]); dest=layout.patch_nodes[src.ball].get(k)
        if dest is None or dest==src:continue
        result['target_patch']=k; new[src,dest].append(result)
    stats.update(entries=len(entries),seeds=len(seeds),literal_second_support_edges=len(new),seconds=perf_counter()-start)
    log('momentum: %s' % dict(stats))
    return entries,new,dict(stats),example
