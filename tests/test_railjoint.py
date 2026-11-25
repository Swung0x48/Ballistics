from collections import Counter
import math

import numpy as np
import pytest

from ballistics import motion, physics as ph, rail_inverse as ri, railjoint as j, takeoff as tf
from test_coarse import mesh, quad


def tube(x0,x1,y=0.):
    return np.array([x0,y,0.,x1,y,0.,.35])


def test_finite_tube_prefix_ends_at_radial_root_after_tip():
    a=tube(0,10)
    piece=motion.flight(ph.State(np.array([10.,2.,0.]),np.array([12.,-1.,0.])),np.array([0.,-.025,0.]),.02,80.)
    t=j.support_end(piece,(a,),2.)
    assert 0.<t<80.
    assert np.linalg.norm(j.tube_radial(piece.point(t),a[:3],a[3:6]))==pytest.approx(2.35,abs=1.e-6)
    assert piece.point(t)[0]>10.


def test_departure_expires_before_return_to_same_tube():
    a=tube(-20,20)
    piece=motion.flight(ph.State(np.array([0.,2.2,0.]),np.array([1.,4.,0.])),np.array([0.,-.2,0.]),.01,43.)
    t=j.support_end(piece,(a,),2.)
    # Later re-entry does not lengthen the contiguous exemption.
    assert 0.<t<10. and piece.point(40.)[1]<2.35
    assert piece.point(t)[1]==pytest.approx(2.35)


def test_downward_radial_vector_terminates_support_while_still_in_capsule():
    a=tube(-20,20)
    piece=motion.flight(ph.State(np.array([0.,.5,1.]),np.array([1.,-2.,0.])),np.array([0.,-.2,0.]),.01,40.)
    t=j.support_end(piece,(a,),2.)
    q=piece.point(t)
    assert 0.<t<40. and q[1]==pytest.approx(0.,abs=1.e-6)
    assert np.linalg.norm(j.tube_radial(q,a[:3],a[3:6]))<2.35


def test_rail_face_filter_does_not_waive_a_disconnected_tube_of_same_object():
    m=mesh([(quad(-1,1,-1,1,0)+quad(6,9,-3,3,0),False)])
    m.rail[:]=True
    obs=tf.Obstacles(m,[])
    st=ph.State(np.array([7.,2.8,0.]),np.array([0.,-4.,0.]))
    dep=tf.RailSupport(0,np.array([0,1]),(tube(-1,1),))
    kind,i,t,p=obs.first_contact(ph.WOOD,st,np.zeros(2),40.,departure=dep,clearance=35.,incoming_only=True)
    assert kind=='land' and i>=2 and t<35.
    # The genuinely departing faces can be collided with after the prefix.
    st=ph.State(np.array([0.,3.6,0.]),np.array([0.,-4.,0.]))
    kind,i,t,p=obs.first_contact(ph.WOOD,st,np.zeros(2),40.,departure=dep,clearance=3.,incoming_only=True)
    assert kind=='land' and i<2 and t>3.


def test_rail_strip_inversion_uses_continuous_axial_and_transverse_support():
    z=np.zeros(3)
    law=ri.ReleaseLaw(np.array([0.,12.,0.]),np.array([8.,0.,0.]),np.array([.3,0.,0.]),z,z,-.02)
    goal=j.Sleeve(1,0,1,np.array([18.,-1.,1.]),np.array([31.,-6.,1.]),1.,42)
    result=list(j.invert_sleeve(tf,ph.PAPER,law,0.,37.,goal))
    assert result
    assert any(hi>lo for _,(lo,hi) in result)
    for q,(lo,hi) in result:
        assert 18.-1.e-6<=q[0]<=31.+1.e-6 and abs(q[2]-1.)<=1.+1.e-6
        push,_=ri.objective(tf,ph.PAPER,law.at((lo+hi)/2),q)
        assert max(abs(push))<=1.+1.e-6
    far=j.Sleeve(1,0,1,np.array([660.,1.,1.]),np.array([676.,0.,1.]),1.,43)
    assert not list(j.invert_sleeve(tf,ph.PAPER,law,0.,37.,far))


def test_prefix_stays_active_across_a_key_event():
    m=mesh([(quad(-5,5,-5,5,0),False)]); m.rail[:]=True
    obs=tf.Obstacles(m,[])
    ball=ph.WOOD
    dep=tf.RailSupport(0,np.array([0,1]),(tube(-4,4),))
    st=ph.State(np.array([0.,2.4,0.]),np.array([0.,-4.,0.]))
    tr=tf.trace_air(obs,ball,st,((1.,np.zeros(2)),(30.,np.zeros(2))),departure=dep,clearance=12.,incoming_only=True)
    assert not any(x['tick']<12. for x in tr.contacts)
    baseline=tf.trace_air(obs,ball,st,((1.,np.zeros(2)),(30.,np.zeros(2))),incoming_only=True)
    assert baseline.contacts[0]['tick']<12.


def test_attached_mitred_cap_is_bound_but_not_a_disconnected_triangle():
    a=np.array([0.,0.,0.]); b=np.array([4.,-1.,0.]); rt=.35
    face=np.array([[0.,0.,rt],[4.,-1.,rt],[4.,-.65,0.]])
    cap=np.array([[4.,-1.,rt],[4.,-.65,0.],[4.,-1.,0.]])
    unrelated=cap+np.array([0.,0.,1.])
    near=np.array([3.9,.6,.2]); triangles=np.array([face,cap,unrelated]); axes,ids=j.bind_axes(np.array([np.r_[a,b,rt]]),triangles,near,2.)
    assert len(axes)==1 and ids.tolist()==[0,1]
