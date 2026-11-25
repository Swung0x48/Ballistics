from types import SimpleNamespace
import math
import numpy as np

from ballistics import motion, physics as ph, railcrown, railride, supportcarry as sc


def test_inner_miter_is_not_the_old_axis_end_and_outer_does_not_join():
    old=np.array([1.,0.,0.]); new=np.array([0.,0.,1.]); at=np.zeros(3)
    inner=motion.line(np.array([-4.,1.,.5]),np.array([0.,1.,.5]))
    t=railcrown.miter_root(inner,at,old,new)
    np.testing.assert_allclose(inner.point(t),[-.5,1.,.5])
    outer=motion.line(np.array([-4.,1.,-.5]),np.array([0.,1.,-.5]))
    assert railcrown.miter_root(outer,at,old,new) is None
    forward=motion.line(np.array([-1.,2.3,0.]),[0.,2.3,0.])
    assert railcrown.miter_root(forward,at,old,new)==1.


def test_miter_support_preserves_position_and_does_not_make_energy():
    r=ph.WOOD.radius+.35+sc.SKIN; q=np.array([-.4,math.sqrt(r*r-.4**2),.4]); n=sc.unit(q-[0.,0.,.4]);
    v=np.array([2.,0.,2.]); shell=np.array([0.,0.,.4])+.35*n
    c=SimpleNamespace(triangle=0,distance=ph.WOOD.radius+sc.SKIN,point=shell,normal=n)
    mesh=SimpleNamespace(rail=[True],owner=[0],entities=[{'elasticity':.3}],contacts=lambda *_:[c])
    paths=railride.RailPaths.__new__(railride.RailPaths); paths.obs=SimpleNamespace(mesh=mesh)
    lane=SimpleNamespace(axis=np.array([[-4.,0.,0.],[0.,0.,0.],[0.,0.,4.]]),twin=np.array([[-4.,0.,-4.],[4.,0.,-4.],[4.,0.,4.]]),
                         length=[4.,4.],tangent=np.array([[1.,0.,0.],[0.,0.,1.]]),tube=.35)
    state=ph.State(q,v)
    out,face,load=paths.crown_miter(ph.WOOD,lane,0,1,state,0,np.zeros(2))
    np.testing.assert_array_equal(out.pos,state.pos)
    assert face==0 and load>0. and np.linalg.norm(out.vel)<=np.linalg.norm(v)
    assert abs(out.vel@n)<1.e-10
    assert paths.crown_miter(ph.WOOD,lane,0,1,ph.State(q,[-8.,0.,2.]),0,np.zeros(2)) is None
    mesh.contacts=lambda *_:[]
    assert paths.crown_miter(ph.WOOD,lane,0,1,state,0,np.zeros(2)) is None


def test_exposed_downhill_cap_has_axial_curvature_pressure_term():
    n=np.array([0.,1.,0.]); force=-20.*n
    assert railcrown.exposed_load(n,np.array([4.,0.,0.]),force,2.)==12.
    assert railcrown.exposed_load(n,np.array([8.,0.,0.]),force,2.)==-12.
