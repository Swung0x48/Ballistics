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
        self.layout=layout;self.surface=surface;self.points={};self.attempts=[];self.tried=set(); self.rows=[]; self.rail_seeds=[]
        for item in points:
            u=g.nearest(np.asarray(item['near']),3.);v=g.nearest(np.asarray(item['far']),3.)
            support=g.info['terrain_support']; ballots=ph.BALLS.values()
            if not (0<=u<len(support) and 0<=v<len(support)):continue
            for b in ballots:
                for node in layout.mapped(b.name,support[u]):
                    self.points.setdefault(node,set()).update(support[v])

    def from_release(self,tf,ball,obs,source,k,rim,st,clock,key,runup,geometry):
        if source not in self.points:return
        switch=math.ceil(clock); delta=switch-clock; loss=ball.lin_damp*ph.DT
        if delta>EPS:
            impulse=cf.air_impulse(ball,push=key); dp,v=cf._motion(delta,st.vel,impulse,loss)
            after=ph.State(st.pos+dp,v,st.omega)
        else:after=st
        for j in sorted(self.points[source]):
            if (source,k,j) in self.tried:continue
            q=polygon_slice(tf,ball,after,self.surface.patches[j])
            if q is None:continue
            self.tried.add((source,k,j)); geom,aim,push,t=q
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
            row=dict(face=int(f),source_patch=k,target_patch=found,origin=geometry[0].p,position=p,
               launch=st.pos,velocity=st.vel,flight_ticks=dt,ticks=clock+dt,key_table=table,
               motion=(*geometry,*(tf.air_geometry(ball,s,p,n) for s,p,n in air.parts)),contacts=air.contacts,
               runup=runup,rim_ticks=rim.seconds/ph.DT,inverse=True)
            if landed is not None and landed!=source:
                self.rows.append((source,landed,row))
            elif obs.mesh.rail[f]:
                self.rail_seeds.append((source,row))
