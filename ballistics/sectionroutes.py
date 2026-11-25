"""Checkpoint kind continuity and conservatively checked transformer transfers."""
from collections import defaultdict

import numpy as np
from shapely import Point

from . import closedform as cf, conformal, conformalsearch as cs, motion, physics as ph, supportcarry as sc, trafo

XZ=[0,2]
EPS=1.e-7


def link_kinds(sections, initial):
    """Select a verified prefix; a checkpoint never generates a new kind."""
    front={initial:(0,0,[])}; rows=[]; blocked=None; prefix=[]
    for k,matrix in enumerate(sections,1):
        next_front={}
        for (before,after),route in matrix.items():
            if before not in front or not route['nodes']:continue
            old,hops,path=front[before]
            score=(old+sum(e['status']!='verified' for e in route['edges']),hops+len(route['edges']))
            if after not in next_front or score<next_front[after][:2]:
                next_front[after]=(*score,[*path,dict(sector=k,entry_ball=before,exit_ball=after,**route)])
        rows.append(dict(sector=k,reachable_entry_kinds=sorted(front),reachable_exit_kinds=sorted(next_front),
                         local_pairs=[list(pair) for pair,r in matrix.items() if r['nodes']]))
        if not next_front and blocked is None:
            blocked=k; prefix=min(front.values(),key=lambda x:x[:2])[2]
        front=next_front
    return dict(status='unresolved' if blocked is not None else 'checked_node_sequence',first_missing_sector=blocked,
                checked_prefix=prefix if blocked is not None else min(front.values(),key=lambda x:x[:2])[2],boundary_states=rows)


def transformer_edges(tf,g,obs,items):
    """Admit rest-chart trigger witnesses, not overlaps of coarse regions.

    The trigger PSI sees the stationary centre. Its replacement follows the
    catalogue hover state and an original-mesh neutral-input first contact.
    Guarded patches and unproved/non-road spawn contacts stay unknown.
    """
    S=g.info['conformal']; L=g.info['conformal_search']; rules=trafo.rules()
    balls={b.name:b for b in ph.BALLS.values()}; edges=[]; audit=[]; key=np.zeros(2)
    for item in items:
        out=balls[item['kind']]; p,v,w=trafo.new_state(item,rules); state=ph.State(p,v,w)
        fall=[]; contacts=[]; elapsed=0.; settled=None; patch=-1; landing=p; why='spawn_bounce_budget'
        # Re-use the event impact rule. A neutral hover drop need not be
        # captured on its first downward contact.
        for _ in range(tf.MAX_BOUNCES+1):
            dur=cf.landing_step(out,state,min(state.pos[1]-out.radius,obs.bottom-out.radius-sc.SKIN))
            if not dur:why='spawn_no_air_horizon';break
            air=tf.trace_air(obs,out,state,((float(dur),key),),follow_below=True,incoming_only=True)
            event,face,t,landing=air.result; patch=int(S.face_patch[face]) if face>=0 else -1
            fall.extend(tf.air_geometry(out,st,k,dt) for st,k,dt in air.parts)
            contacts.extend(dict(hit,tick=hit['tick']+elapsed) for hit in air.contacts)
            elapsed+=t
            if event!='land' or patch<0 or not air.contacts:why=event if event!='land' else 'spawn_nonroad_support';break
            target=L.patch_nodes[out.name][patch]; hit=air.contacts[-1]
            n=hit.get('support_normal',hit['normal']); material=obs.mesh.entities[obs.mesh.owner[face]]
            incoming=ph.State(landing,hit['incoming'],hit['incoming_omega'])
            rebound,_=cf.collision_impulse(out,incoming,n,float(material.get('elasticity',tf.DEFAULT_ELASTICITY)))
            if sc.apex(out,rebound.vel,n,key)<=sc.RECAPTURE:
                why='spawn_supported' if patch in L.roads[target].rest_patches else 'spawn_arrival_not_rest_chart'
                if why=='spawn_supported':settled=target
                break
            r=out.radius+sc.SKIN+8*EPS
            state=ph.State(hit['point']+r*n,rebound.vel,rebound.omega)
        audit.append(dict(name=item['name'],ball=out.name,reason=why,first_patch=patch,hover=p,landing=landing,contacts=contacts))
        if settled is None:continue
        target=settled
        if any(other is not item and other['kind']!=out.name and any(motion.sphere_hit(x,other['pos'],rules.radius) for x in fall)
               for other in items):continue
        for ball in balls.values():
            if ball.name==out.name:continue
            radius=ball.radius+sc.SKIN; point=item['pos']; index=L.patch_nodes[ball.name]
            for k,q in enumerate(S.patches):
                if q.normal[1]<conformal.ANGLE or np.any(point<q.low-rules.radius-radius) or np.any(point>q.high+rules.radius+radius):continue
                a=index[k]
                if k not in L.roads[a].rest_patches:continue
                x=point+q.normal*(q.offset+radius-point@q.normal)
                if not q.polygon.contains(Point((x-radius*q.normal)[XZ])) or trafo.triggered(x,ball.name,items,rules) is not item:continue
                if cs._capsule_blocked(tf,obs,x,x,ball.radius-sc.RECAPTURE,set(q.faces)) or any(
                        motion.box_hit(motion.line(x,x),lo,hi) is not None for lo,hi in obs.boxes):continue
                edges.append((a,target,dict(origin=x,position=landing,source_patch=k,target_patch=patch,transformer=item['name'],
                    input_ball=ball.name,output_ball=out.name,ticks=elapsed+rules.body_after,motion=tuple(fall),contacts=tuple(contacts),
                    witness_reference='catalogue_rest_trigger_and_mesh_spawn')))
    return edges,audit


def crosses_foreign_transformer(source,row,items):
    """Conservative screen on an individual ordinary witness, not a region split."""
    if row.get('transformer'):return False
    radius=trafo.rules().radius
    return any(t['kind']!=source.ball and any(motion.sphere_hit(x,t['pos'],radius) for x in row.get('motion',())) for t in items)
