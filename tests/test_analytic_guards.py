"""Analytic route and simultaneous-contact regressions for v10."""
from types import SimpleNamespace

import numpy as np

from ballistics import closedform as cf, coarse, motion, solve, takeoff as tf
from ballistics.physics import WOOD, DT, State
from test_coarse import mesh, quad


def test_two_walls_enter_one_manifold_and_one_bounce():
    a,b,c,d = [(0.,y,z) for y,z in [(-10.,-10.),(20.,-10.),(20.,20.),(-10.,20.)]]
    faces=np.asarray([(a,b,c),(a,c,d)])
    m=mesh([(faces,False),(faces[:,:,[2,1,0]],False)])
    start=State(np.array([3.,3.,3.]),np.array([-10.,0.,-10.]))
    result = tf.trace_air(tf.Obstacles(m,()),WOOD,start,[(20.,np.zeros(2))],max_bounces=1)
    assert result.bounces==1 and result.result[0]=='no_contact'
    assert len(result.contacts)==1
    h=result.contacts[0]
    assert len(h['manifold'])==2
    assert h['outgoing'][0]>0 and h['outgoing'][2]>0


def test_rail_seed_keeps_source_direction_and_speed_interval_distinct():
    state=State(np.zeros(3),np.array([9.,0.,0.]))
    a=SimpleNamespace(lane=2,direction=1,j=5,body=state,lateral=.1)
    b=SimpleNamespace(**vars(a)); b.direction=-1
    c=SimpleNamespace(**vars(a)); c.body=State(state.pos,np.array([12.,0.,0.]))
    keys={tf.rail_seed_key(10,u,20,WOOD,cap) for u,cap in [(0,a),(1,a),(0,b),(0,c)]}
    assert len(keys)==4


def test_air_geometry_gate_box_and_transformer_detect_arc_missed_by_end_chord():
    start=State(np.array([0.,0.,0.]),np.array([8.,20.,0.]))
    push=np.zeros(2); flight=tf.air_geometry(WOOD,start,push,90.)
    q=flight.point(45.)
    chord=motion.line(start.pos,flight.point(90.))
    r=.025
    assert motion.box_hit(chord,q-r,q+r) is None
    assert motion.box_hit(flight,q-r,q+r) is not None
    assert not motion.sphere_hit(chord,q,r)
    assert motion.sphere_hit(flight,q,r)
    edge=dict(ball='wood',motion=(flight,),way=np.array([chord.p,chord.point(1.)]))
    assert not tf.valid_in_sector(edge,chord.point(1.),[(q,'stone')],r)
    # shrink the effective collision radius to distinguish the bowed path
    part=(q,np.full(3,r),np.eye(3),None); clear=solve.BALL_R-r
    assert not solve._hits((chord,),[part],clear)
    assert solve._hits((flight,),[part],clear)


def test_support_prism_union_finds_narrow_unsupported_gap():
    a=np.array([0.,2.,0.]); b=np.array([4.,2.,0.])
    without_gap=mesh([(quad(-1.,5.,-2.,2.,0.),False)])
    gap=mesh([(quad(-1.,2.10,-2.,2.,0.),False),(quad(2.11,5.,-2.,2.,0.),False)])
    assert solve._FixedSupport(without_gap).line(a,b)
    assert not solve._FixedSupport(gap).line(a,b)


def test_horizontal_balance_can_find_a_narrow_unbracketed_gap():
    h=np.array([[1.,0.],[-1.,.002]])
    assert coarse.horizontal_balance(h,tol=.0011)
    assert not coarse.horizontal_balance(h,tol=.0009)


def test_sloping_dual_rail_balance_uses_local_cross_section():
    from ballistics.contactset import opposed_support
    a=np.array([.28,.496,-.822]); b=a*np.array([1.,1.,-1.]); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
    assert (a[[0,2]]@b[[0,2]])/(np.linalg.norm(a[[0,2]])*np.linalg.norm(b[[0,2]])) > -.9
    assert opposed_support(a,b)[1] >= coarse.WALK_NY
