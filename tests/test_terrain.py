import math
from types import SimpleNamespace

import numpy as np
import pytest

from ballistics import coarse, conformal, physics as ph, regions, supportcarry as sc, takeoff as tf, terrain as T
from test_coarse import mesh, quad


def test_force_projection_and_c19_interval():
    for b,angles in [(ph.WOOD,(36.8,46.6)),(ph.STONE,(16.9,23.2)),(ph.PAPER,(47.0,56.6))]:
        np.testing.assert_allclose(np.degrees(np.arctan(T.ratio(b)*np.array([1,math.sqrt(2)]))),angles,atol=.12)
    assert T.slope_verdict(ph.STONE,math.tan(math.radians(16.5)),[1.,0.])==0
    assert T.slope_verdict(ph.STONE,math.tan(math.radians(21)),[1.,0.])==-1
    assert T.slope_verdict(ph.STONE,math.tan(math.radians(21)),[1.,1.])==1


def test_v_bearing_line_uses_common_grade_not_plane_or_chord_grade():
    a=sc.unit(np.array([-.576,.787,.22])); b=a*np.array([1.,1.,-1.])
    grade,h,sign,mode=T.slope_for([a,b],np.array([1.,1.]))
    assert sign>0 and mode=='support_line'
    assert math.degrees(math.atan(grade))==pytest.approx(36.2,abs=.1)
    assert T.slope_verdict(ph.STONE,grade,h)==-1
    assert T.slope_for([a,b],np.array([-1.,1.]))[2]<0


def ramp(angle=36.):
    h=math.tan(math.radians(angle))
    tris=[tuple(np.array([x,h*x,z]) for x,_,z in t) for t in quad(0,12,-4,4,0)]
    m=mesh([(tris,False)]); g=coarse.build(m);s=conformal.build(m)
    g.info['conformal']=s; T.rebuild(g,m,s)
    return m,g,s


def test_direction_and_kind_apply_to_both_state_searches_and_rolling_regions():
    m,g,s=ramp(21.)
    u=g.nearest((3,3.3,0)); v=g.nearest((4,3.7,1))
    assert g.info['terrain_ball_masks'][u,v,'roll']==5 # wood and paper, no zigzag discount
    rel=SimpleNamespace(fwd=g.adj)
    sg=coarse.StateGraph(rel,g,np.ones(len(g.pos),bool),np.full(len(g.pos),-1),0.,[],[],0.)
    assert any(e[1]==v for e in sg.step(coarse.WOOD,u))
    assert not any(e[1]==v for e in sg.step(coarse.STONE,u))
    assert any(e[1]==u for e in sg.step(coarse.STONE,v))
    labels=regions.roll_labels(sg)
    assert labels[coarse.WOOD,u]==labels[coarse.WOOD,v]
    assert labels[coarse.STONE,u]!=labels[coarse.STONE,v]
    assert not np.isfinite(sg.togo(v)[coarse.STONE,u])


def test_finite_uphill_run_stalls_or_reaches_crest_by_closed_form():
    n=sc.unit(np.array([-.39,1.,0.]));d=sc.unit(np.array([1.,.39,0.])); b=ph.STONE
    law=sc.law(b,n,np.array([1.,0.])); turn=T.climb_turn(law,d,d*7)
    distance,v=law.at(d*7,turn)
    assert (law.at(d*7,turn+.001)[0]-law.at(d*7,turn-.001)[0])@d==pytest.approx(0.,abs=1e-9)
    assert law.along(d).over(7.,float(distance@d)-.1)
    assert law.along(d).over(7.,float(distance@d)+.1) is None
    assert law.at(np.zeros(3),1.)[1]@d<0.


def test_riser_torque_and_moving_wood_record_case():
    assert T.step_verdict(ph.WOOD,.21,2.,[1.,0.])==1
    assert T.step_verdict(ph.WOOD,1.45,2.,[1.,0.])==-1
    assert T.step_verdict(ph.STONE,.21,2.,[1.,0.])==-1
    assert T.step_energy(ph.WOOD,1.45,0.,[1.,0.],.4)[0]<0.
    assert T.step_energy(ph.WOOD,1.45,15.,[1.,0.],.4)[0]>0.
    assert T.step_energy(ph.WOOD,2.01,30.,[1.,0.],.4) is None


def test_step_is_a_real_riser_and_keeps_ball_mask():
    h=.24
    a,b,c,d=map(np.asarray,[(8,0,-4),(8,0,4),(8,h,4),(8,h,-4)])
    mesh0=mesh([(quad(0,8,-4,4,0)+quad(8,16,-4,4,h)+[(a,b,c),(a,c,d)],False)])
    g=coarse.build(mesh0);s=conformal.build(mesh0);g.info['conformal']=s;T.rebuild(g,mesh0,s)
    assert sum(map(len,g.info['terrain_lips'].values()))==1
    edges=[(u,v) for u,row in enumerate(g.adj) for v,_,k in row if k=='step']
    assert edges and all(not g.info['terrain_ball_masks'][u,v,'step']&2 for u,v in edges)


def test_overspeed_riser_is_checked_without_a_static_step():
    h=.8
    a,b,c,d=map(np.asarray,[(8,0,-4),(8,0,4),(8,h,4),(8,h,-4)])
    m=mesh([(quad(0,8,-4,4,0)+quad(8,18,-4,4,h)+[(a,b,c),(a,c,d)],False)])
    g=coarse.build(m);s=conformal.build(m);g.info['conformal']=s;T.rebuild(g,m,s)
    class Landing:
        def near(self,*a):return 0
    paths=sc.SurfacePaths(g,tf.Obstacles(m,[]),Landing())
    state=ph.State(np.array([3.,2.+sc.SKIN,0.]),np.array([15.,0.,0.]))
    r=paths.floor(tf,ph.WOOD,0,0,state,np.array([0.,1.,0.]),chosen_push=np.array([1.,0.]))
    assert any(p['kind']=='step' for p in r.parts)
    assert next(p for p in r.parts if p['kind']=='step')['end_position'][1]==pytest.approx(h+2.+sc.SKIN)


def test_slope_carry_returns_down_after_analytic_turn():
    m,g,s=ramp(36.)
    n=s.patches[0].normal
    d=sc.unit(np.array([1.,math.tan(math.radians(36)),0.]))
    start=np.array([3.,3.*math.tan(math.radians(36)),0.])+(2.+sc.SKIN)*n
    class Landing:
        def near(self,*a):return 0
    paths=sc.SurfacePaths(g,tf.Obstacles(m,[]),Landing())
    r=paths.floor(tf,ph.STONE,0,0,ph.State(start,d*3),n,chosen_push=np.array([1.,0.]))
    assert r.parts[0]['reason']=='uphill_turn'
    assert r.parts[0]['end_position'][0]>start[0]
    assert r.parts[1]['end_position'][0]<start[0]
