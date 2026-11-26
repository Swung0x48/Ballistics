from types import SimpleNamespace

import numpy as np
import pytest
from shapely import Polygon, box

from ballistics import physics as ph, railinlet, railride, supportcarry as sc


def flat(pol):
    return SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.,polygon=pol)


def test_flat_road_joins_a_finite_coplanar_rail_interval():
    p=flat(box(1.,-1.,3.,1.))
    x=np.array([0.,2.,0.]); t=np.array([1.,0.,0.])
    assert railinlet.bearing_intervals(p,x,t,4.,2.,.02)==((1.,3.),)
    assert railinlet.bearing_intervals(p,x+[0.,.03,0.],t,4.,2.,.02)==()


def test_bearing_station_clips_by_plane_and_polygon_hole():
    p=flat(Polygon([(0.,-2.),(6.,-2.),(6.,2.),(0.,2.)],holes=[[(2.,-1.),(3.,-1.),(3.,1.),(2.,1.)]]))
    t=np.array([1.,.01,0.]); t=t/np.linalg.norm(t)
    xs=railinlet.bearing_intervals(p,np.array([0.,1.98,0.]),t,6.,2.,.02)
    assert np.asarray(xs)*t[0]==pytest.approx(np.asarray([(0.,2.),(3.,4.)]))


def test_held_key_need_not_reach_ring_if_an_integer_switch_is_earlier():
    ball=ph.WOOD; shape=np.array([[0.,2.,0.],[5.,2.,0.]])
    t=np.array([1.,0.,0.]); n=np.array([0.,1.,0.]); v=.9
    brake,go=np.array([-1.,0.]),np.array([1.,0.]); law=sc.law(ball,n,brake).along(t)
    assert law.over(v,5.) is None
    paths=railride.RailPaths.__new__(railride.RailPaths)
    paths.lanes={0:railride.Lane(np.array((0,1)),shape,np.array([5.]),t[None],False,.3,0.)}
    paths.obs=SimpleNamespace(mesh=SimpleNamespace(owner=[0]))
    paths.surface=SimpleNamespace(name=lambda _: 'test',clear=lambda *a:True)
    c=railride.Catch(0,0,1,ph.State(shape[0],v*t),0.,0)
    controls=lambda clock: ((1-clock,brake),) if clock<1. else ((float('inf'),go),)
    dep,=paths.walk(None,ball,c,controls,search_side=False)
    assert dep.kind=='chain_end'
    assert dep.pieces[0]['ticks']==1.
    np.testing.assert_allclose(dep.pieces[0]['end_position'],shape[0]+t*law.at(v,1.)[0])
    np.testing.assert_allclose(dep.body.pos,shape[1])


def test_rail_end_aim_retains_ordinary_outward_key_without_replaying_ground():
    from ballistics.sharedsearch import KEYS, rail_end_keys
    out=rail_end_keys(ph.WOOD,np.array([0.,1.,0.]),np.array([0.,0.,-3.]),
                      np.array([[2.,0.,0.],[-2.,0.,1.]]))
    assert {tuple(KEYS[i]) for i in out}=={(1.,-1.),(1.,0.)}
