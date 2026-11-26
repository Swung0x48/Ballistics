import math
from types import SimpleNamespace

import numpy as np
import pytest

from ballistics import physics as ph, railstatic as qs


def test_axis_balance_needs_transverse_thrust_reserve():
    t=np.array([math.cos(.4),math.sin(.4),0.])
    assert qs.axial_margin(ph.WOOD,t)>0
    assert qs.axial_margin(ph.STONE,t)<0
    assert qs.axial_margin(ph.WOOD,-t)==pytest.approx(qs.axial_margin(ph.WOOD,t))


def test_groove_support_needs_two_opposite_reactions_across_whole_interval():
    t=np.array([1.,0.,0.]); a=np.array([0.,1.6,-1.2]); b=np.array([0.,1.6,1.2]); z=np.zeros(3)
    assert qs.bearing_intervals([(1,0.,1.,a,z),(2,0.,1.,b,z)],t,2.,2.8,.36,.3)
    assert not qs.bearing_intervals([(1,0.,1.,a,z),(2,0.,.9,b,z)],t,2.,2.8,.36,.3)
    assert not qs.bearing_intervals([(1,0.,1.,a,z),(2,0.,1.,a,z)],t,2.,2.8,.36,.3)
    cap=np.array([.5,1.6,1.2]); assert not qs.bearing_intervals([(1,0.,1.,a,z),(2,0.,1.,cap,z)],t,2.,2.8,.36,.3)


def test_feature_interval_catches_penetration_inside_clear_endpoints():
    tri=np.array([[[0.,-4.,-4.],[0.,4.,-4.],[0.,0.,5.]]]); n=np.array([[1.,0.,0.]])
    obs=SimpleNamespace(mesh=SimpleNamespace(N=n,rail=np.array([False]),_grid=None),
                        tris=tri,lo=tri.min(1),hi=tri.max(1))
    tf=SimpleNamespace(_box_triangles=lambda *args:[0])
    a,b=np.array([-3.,0.,0.]),np.array([3.,0.,0.]);_,hit=qs.segment_features(tf,obs,a,b,2.)
    assert hit


def test_miter_must_join_two_finite_charts():
    a,b,c,d=(np.array(p,float) for p in ((0,0,0),(2,0,0),(1.5,0,-.5),(1.5,0,1.5)))
    np.testing.assert_allclose(qs.miter_station(a,b,c,d),[1.5,0.,0.])
    lift=np.array([0.,.1,0.]); assert qs.miter_station(a,b,c+lift,d+lift) is None
    shift=np.array([1.,0.,0.]); assert qs.miter_station(a,b,c+shift,d+shift) is None


def test_arc_band_change_is_found_by_closed_polynomial_roots():
    D=np.array([[1.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
    assert qs.norm_roots(D,1.5,1.5)==pytest.approx((math.pi/3,))


def test_coverage_keeps_a_hole_cut_by_a_penetrating_feature():
    windows=qs.unblocked([(0.,1.)],[(.2,.3),(.6,.8)])
    assert len(windows)==3
    assert windows[0][1]<.2 and windows[1][0]>.3 and windows[1][1]<.6 and windows[2][0]>.8
    assert not qs.covers(windows)
