"""Inverse of a held-key, immediate-pressure-release runway to a crown."""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from . import closedform as cf, motion, physics as ph, railinlet, railride, supportcarry as sc, sweep

EPS=sc.EPS


def entrance_charts(g,ball):
    r=ball.radius+sc.SKIN; charts=[]
    for i,lane in enumerate(g.lanes):
        if lane['double'] and not lane.get('axes'):continue
        tubes=lane.get('axes',()) if lane['double'] else (np.asarray(lane['segments'])-(ph.WOOD.radius+lane['radius'])*sc.UP,)
        for tube in tubes:
            if len(tube)<2:continue
            for u,v in ((tube[0],tube[1]),(tube[-1],tube[-2])):
                length=np.linalg.norm(v-u)
                if length<=EPS:continue
                t=(v-u)/length; a=u+(r+lane['radius'])*railride.support_normal(t)
                charts.append((i,a,a+min(length,r)*t))
    return cKDTree([a for _,a,b in charts]) if charts else None,charts


def invert(ball,n,a,b,d,q,push):
    """At fixed target station, separate flight normal and tangent equations."""
    e=b-a; imp=cf.air_impulse(ball,push=push)
    normal=float(imp@n); height=float((q-a)@n)
    if normal>=-EPS or height>=-EPS:return None
    loss=ball.lin_damp*ph.DT; log=math.log1p(-loss); g=height/normal
    hi=3+loss*g/ph.DT+2/loss
    tick=brentq(lambda x:sweep.chord(x,loss,log)[1]-g,0.,hi)
    A,B=sweep.chord(tick,loss,log)
    m=np.stack((e,A*d,n),axis=1)
    if abs(np.linalg.det(m))<=EPS:return None
    z,s,err=np.linalg.solve(m,q-a-B*imp)
    if not EPS<z<1.-EPS or s<=EPS or abs(err)>EPS:return None
    return float(z),float(s),tick


def seed(tf,obs,ball,p,f,a,b,push,goal):
    n=p.normal; r=ball.radius+sc.SKIN
    law=sc.law(ball,n,push); d=sc.unit(law.impulse); out=sc.unit(np.cross(b-a,n))
    if d@out<=EPS:return None,'key_not_outward'
    a,b=a.copy(),b.copy()
    for endpoint in (a,b):endpoint[1]=p.height(endpoint[[0,2]])
    proposal=invert(ball,n,a+r*n,b+r*n,d,goal,push)
    if proposal is None:return None,'inverse_no_finite_lip_speed'
    z,v,_=proposal; foot=a+z*(b-a); foot[1]=p.height(foot[[0,2]])
    axis=law.along(d); end=axis.terminal_speed
    if not EPS<v<end-EPS:return None,'inverse_runway_speed_limit'
    ticks=math.log1p(-v/end)/math.log1p(-axis.loss)
    distance=axis.at(0.,ticks)[0]; depth=railinlet._approach(p,foot,d)
    if not EPS<distance<depth-EPS:return None,'inverse_runway_too_short'
    rim=tf.rim_release(ball,foot,n,out,v*d,0.,push=push,inertia_ratio=sc.inertia_ratio(ball))
    if rim is None or rim.angle>EPS:return None,'inverse_not_immediate_release'
    origin=ph.State(foot+r*n-distance*d,np.zeros(3)); start=rim.start
    if not obs.support_clear(origin,law.impulse,law.loss,ticks,int(obs.mesh.owner[f]),ball.radius-sc.RECAPTURE):
        return None,'inverse_runway_obstructed'
    horizon=cf.landing_step(ball,start,min(obs.bottom-r,start.pos[1]-r))
    if horizon is None:return None,'inverse_no_air_horizon'
    air=tf.trace_air(obs,ball,start,((float(horizon),push),),rim=rim)
    why,face,time,at=air.result
    if why!='land' or time<=EPS or not obs.mesh.rail[face] or not air.contacts:
        return None,'inverse_first_route_support_not_rail'
    finish=ticks+time
    return dict(face=int(face),source_face=int(f),origin=origin.pos,position=at,launch=start.pos,velocity=start.vel,
                flight_ticks=time,ticks=finish,key_table=((math.ceil(finish),push.copy()),),contacts=air.contacts,
                motion=(motion.rolling(origin,law.impulse,law.loss,ticks),*(tf.air_geometry(ball,s,k,t) for s,k,t in air.parts)),
                runup=distance,rim_ticks=0.,inlet_kind='crown_end_inverse',crown_inverse=True,
                episode_bounces=air.bounces),'inverse_crown_contact'


def seeds(tf,g,obs,layout,features,balls,stats):
    from .sharedsearch import KEYS,rail_end_keys
    surf=g.info['conformal']; out=[]
    for name,ball in balls.items():
        index,charts=entrance_charts(g,ball)
        if index is None:continue
        r=ball.radius+sc.SKIN
        for node,region in layout.roads.items():
            if node.ball!=name:continue
            for k in region.rest_patches:
                p=surf.patches[k]
                for f,a,b,bend in features[k]:
                    if bend is not None:continue
                    base=(a+b)*.5+r*p.normal; edge=b-a; length=np.linalg.norm(edge)
                    if length<=EPS:continue
                    for j in index.query_ball_point(base,3*r+.5*length):
                        chain,u,v=charts[j]; near=u-(a+r*p.normal)
                        delta=near-np.clip(near@edge/(edge@edge),0.,1.)*edge
                        if delta@delta>(3*r)**2:continue
                        # Fix a representative strictly inside the finite
                        # receiving interval. No tick or station scan.
                        goal=(u+v)*.5
                        if (goal-base)@p.normal>=-EPS:continue
                        for idx in rail_end_keys(ball,p.normal,edge,(goal-base,)):
                            if idx is None:continue
                            row,why=seed(tf,obs,ball,p,f,a,b,KEYS[idx],goal);stats[why]+=1
                            if row is not None:
                                row.update(source_patch=k,aimed_chain=chain,aimed_entrance_interval=(u,v))
                                out.append((node,row))
    return out
