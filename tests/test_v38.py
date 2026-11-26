import math
from types import SimpleNamespace

import numpy as np

from ballistics import motion, regions, rimclear


def test_departure_from_fitted_support_waives_only_initial_bearing_face():
    tri=np.array([[0.,0.,-3.],[-3.,0.,0.],[0.,0.,3.]])
    tilt=.0002
    curve=motion.arc(np.zeros(3),2*np.array([-math.sin(tilt),math.cos(tilt),0.]),
                     2*np.array([math.cos(tilt),math.sin(tilt),0.]),0.,.4,
                     np.array([0.,0.,2.]),(9.,0.,0.))
    feature=next(f for f in rimclear.features(tri) if f[0]=='face')
    initial=rimclear.feature_minimum(curve,tri,feature)
    assert initial['parameter']==0. and not initial['boundary_limit']
    assert rimclear.leaving_face(curve,tri,tri[[0,2]],initial)
    assert not rimclear.leaving_face(curve,tri,tri[[0,1]],initial)
    assert not rimclear.leaving_face(curve,tri,tri[[0,2]],dict(initial,parameter=.5))


def test_proven_uphill_is_not_labelled_downhill_or_one_way():
    sg=SimpleNamespace(g=SimpleNamespace(info={'native_geometry':True}))
    key=('plain',3,4,'roll',0,0,0)
    res=dict(sg=sg,links={key:{}})
    points=np.array([[0.,1.,0.],[1.,2.,0.]])
    assert regions._describe_connection(res,key,{},geometry=(points,None))=='roll uphill; reverse not certified'


def test_fixed_vertex_bounds_do_not_integrate_axial_time(monkeypatch):
    def fail(*args):
        raise AssertionError('fixed pivot has no axial displacement')
    monkeypatch.setattr(motion.Piece,'axial',fail)
    curve=motion.arc(np.zeros(3),np.array([0.,2.,0.]),np.array([2.,0.,0.]),
                     0.,math.pi/4,np.zeros(3),(9.,0.,0.))
    low,high=curve.bounds(0.,1.)
    np.testing.assert_allclose(low,[0.,math.sqrt(2),0.])
    np.testing.assert_allclose(high,[math.sqrt(2),2.,0.])
    tri=np.array([[4.,0.,-2.],[4.,-2.,0.],[4.,0.,2.]])
    assert any(rimclear.feature_minimum(curve,tri,f) is not None for f in rimclear.features(tri))
