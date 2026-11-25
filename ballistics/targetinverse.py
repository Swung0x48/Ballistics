"""Target-side inverse for a fixed already checked release, not a global proof."""
import math

import numpy as np
from shapely import LineString,box
from shapely.affinity import translate

from . import closedform as cf, physics as ph,runfamily,supportcarry as sc

XZ=[0,2]
EPS=1.e-7


def polygon_slice(tf,ball,start,patch):
    r=ball.radius+sc.SKIN; n=patch.normal
    poly=translate(patch.polygon,*(r*n[XZ])); aim=np.asarray(poly.representative_point().coords[0])
    y=float(patch.height(aim,r)); a,b=tf._target_coeff(ball,start,y)
    width=float(b)*ball.push
    if not math.isfinite(width) or width<=EPS:return None
    center=start.pos[XZ]+a*start.vel[XZ]; q=poly.intersection(box(*(np.r_[center-width,center+width])))
    if q.is_empty:return None
    if np.linalg.norm(n[XZ])>EPS:
        # At this descending height the footprint is a finite intersection
        # with a plane line, not a constant-height approximation to a ramp.
        axis=sc.unit(np.array([-n[2],n[0]])); extent=np.linalg.norm(np.asarray(poly.bounds)[2:]-np.asarray(poly.bounds)[:2])+1.
        q=q.intersection(LineString([aim-extent*axis,aim+extent*axis]));
        if q.is_empty:return None
    xy=np.asarray(q.representative_point().coords[0]); push=(xy-center)/width
    t=tf._time_at_b(float(b),ball.lin_damp*ph.DT)
    return q,np.array([xy[0],y,xy[1]]),np.clip(push,-1,1),float(t)


class Priorities:
    def __init__(self,layout,surface,g,points):
        self.layout=layout;self.surface=surface;self.points={};self.attempts=[];self.tried=set(); self.rows=[]; self.rail_seeds=[]; self.carry_points={}
        for item in points:
            u=g.nearest(np.asarray(item['near']),3.);v=g.nearest(np.asarray(item['far']),3.)
            support=g.info['terrain_support']; ballots=ph.BALLS.values()
            if not (0<=u<len(support) and 0<=v<len(support)):continue
            for b in ballots:
                if item.get('carry') and item.get('ball') not in (None,b.name):continue
                for node in layout.mapped(b.name,support[u]):
                    self.points.setdefault(node,set()).update(support[v])
                    if item.get('carry'):
                        self.carry_points.setdefault(node,set()).update(support[u])

    def from_release(self,tf,ball,obs,source,k,rim,st,clock,key,runup,geometry,feature=None):
        if source not in self.points:return
        switch=math.ceil(clock); delta=switch-clock; loss=ball.lin_damp*ph.DT
        if delta>EPS:
            impulse=cf.air_impulse(ball,push=key); dp,v=cf._motion(delta,st.vel,impulse,loss)
            after=ph.State(st.pos+dp,v,st.omega)
        else:after=st
        for j in sorted(self.points[source]):
            identity=(source,k,j,tuple(key),feature)
            if identity in self.tried:continue
            q=polygon_slice(tf,ball,after,self.surface.patches[j])
            if q is None:continue
            self.tried.add(identity); geom,aim,push,t=q
            lim=cf.landing_step(ball,after,min(obs.bottom-ball.radius-sc.SKIN,after.pos[1]-ball.radius))
            if not lim:continue
            schedule=((switch,key),(switch+max(t,lim),push)) if switch else ((max(t,lim),push),)
            table=runfamily.literal_schedule(schedule)
            actual=tuple((b-clock,k) for b,k in table if b>clock+EPS)
            air=tf.trace_air(obs,ball,st,actual,rim=rim)
            why,f,dt,p=air.result; found=int(self.surface.face_patch[f]) if f>=0 else -1
            dest=self.layout.patch_nodes[ball.name].get(j); landed=self.layout.patch_nodes[ball.name].get(found)
            from shapely.geometry import mapping
            match=why=='land' and dt>EPS and landed==dest and dest!=source
            self.attempts.append(dict(source=source,target=dest,switch_tick=switch,target_point=aim,target_slice=mapping(geom),
                                      status='literal_confirmed' if match else 'unresolved',first_event=why,first_patch=found,
                                      release_patch=k, first_face=int(f), actual_target=landed,
                                      first_object=obs.mesh.entities[int(obs.mesh.owner[f])]['name'] if f>=0 else None,
                                      key_table=table))
            if why!='land' or dt<=EPS:continue
            last=math.ceil(clock+dt)
            used=[]
            for end, held in table:
                used.append((min(last,end),held))
                if end>=last:break
            row=dict(face=int(f),source_patch=k,target_patch=found,origin=geometry[0].p,position=p,
               launch=st.pos,velocity=st.vel,flight_ticks=dt,ticks=clock+dt,key_table=tuple(used),
               motion=(*geometry,*(tf.air_geometry(ball,s,p,n) for s,p,n in air.parts)),contacts=air.contacts,
               runup=runup,rim_ticks=rim.seconds/ph.DT,inverse=True)
            if landed is not None and landed!=source:
                self.rows.append((source,landed,row))
            elif obs.mesh.rail[f]:
                self.rail_seeds.append((source,row))


    def carried_releases(self,tf,g,obs):
        """From a nominated rest block, follow actual conformal support events.

        This is a finite representative family (one interior position for
        each support polygon and each true input), not a velocity envelope.
        Receiving a different block does not reset speed to rest.
        """
        from collections import Counter
        from types import SimpleNamespace
        from . import momentum, rimclear
        from .sharedsearch import KEYS
        surface=self.surface
        paths=sc.SurfacePaths(g,obs,SimpleNamespace(near=lambda face,_:0 if surface.face_patch[face]>=0 else -1))
        balls={b.name:b for b in ph.BALLS.values()}
        counts=Counter()
        for source,ids in self.carry_points.items():
            ball=balls[source.ball]
            r=ball.radius+sc.SKIN
            for k in sorted(ids.intersection(self.layout.roads[source].rest_patches)):
                patch=surface.patches[k]
                xy=np.asarray(patch.polygon.representative_point().coords[0])
                origin=np.array([xy[0],patch.height(xy),xy[1]])+r*patch.normal
                begin=ph.State(origin,np.zeros(3))
                for key in KEYS:
                    carry=paths.floor(tf,ball,patch.faces[0],-1,begin,patch.normal,chosen_push=key,follow_support=True)
                    counts[carry.reason]+=1
                    if carry.reason not in ('floor_end','convex') or carry.rim is None:continue
                    geometry=tuple(x for leg in carry.parts for x in leg['motion'])
                    arcs=[(x,momentum._span(x,surface)) for x in geometry if x.kind=='arc']
                    if any(span is None or rimclear.check(tf,ball,x,obs,span)[0]!='clear' for x,span in arcs):
                        counts['runway_rim_not_clear']+=1; continue
                    duration=sum(x['ticks'] for x in carry.parts)
                    first,rails=len(self.rows),len(self.rail_seeds)
                    self.from_release(tf,ball,obs,source,k,carry.rim,carry.body,duration,key,
                        np.linalg.norm(carry.body.pos-origin),geometry,
                        feature=('carry',tuple((x.get('patch',-1),x['kind']) for x in carry.parts)))
                    chain=tuple(x['patch'] for x in carry.parts if 'patch' in x)
                    for item in (*self.rows[first:],*self.rail_seeds[rails:]):
                        item[-1]['carried_runup']=chain
        return dict(counts)
