import numpy as np

from ballistics import conformal,conformalsearch,physics as ph,roadway,sharedrail,supportededges,takeoff as tf
from test_coarse import mesh,quad


def test_finite_near_t_seam_has_supported_literal_crossing():
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(quad(10.01,20.,-2.,2.,0.),False)])
    s=conformal.build(m); obs=tf.Obstacles(m,()); (a,b,span),=list(supportededges.overlaps(roadway.Topology(s)))
    np.testing.assert_allclose(np.sort(span[:,1]),[-2.,2.])
    row,why=supportededges.prove(tf,obs,s,ph.WOOD,a,b)
    assert why=='literal_roll' and row['position'][0]>10.01
    assert all(p.kind!='air' for p in row['motion'])
    assert all(int(k)==k and set(u)<= {-1.,0.,1.} for k,u in row['key_table'])


def test_bridge_beyond_export_band_is_not_a_supported_seam():
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(quad(10.1,20.,-2.,2.,0.),False)])
    assert not list(supportededges.overlaps(roadway.Topology(conformal.build(m))))


def test_ground_portal_checks_wall_between_the_two_charts():
    parts=[(quad(0.,10.,-5.,5.,0.),False),(quad(10.01,20.,-2.,2.,0.),False)]
    a,b,c,d=((10.,y,z) for y,z in ((0.,-4.),(5.,-4.),(5.,4.),(0.,4.)))
    m=mesh([*parts,([(a,b,c),(a,c,d)],False)]);s=conformal.build(m);(a,b,_),=list(supportededges.overlaps(roadway.Topology(s)))
    row,why=supportededges.prove(tf,tf.Obstacles(m,()),s,ph.WOOD,a,b)
    assert row is None and 'obstruction' in why


def test_supported_rail_seed_from_rest_reaches_finite_outlet():
    from types import SimpleNamespace
    from ballistics import railride
    p=np.array([[0.,2.,0.],[10.,2.,0.]])
    lane=railride.Lane(np.array([0]),p,np.array([10.]),np.array([[1.,0.,0.]]),True,.35,3.8)
    path=SimpleNamespace(lanes={0:lane},obs=SimpleNamespace(mesh=SimpleNamespace(owner=[0])),
                         surface=SimpleNamespace(name=lambda _: 'rail',clear=lambda *a: True))
    start=railride.Catch(0,0,1,ph.State(p[0],np.zeros(3)),0.,0); key=np.array([1.,0.])
    end,=railride.RailPaths.walk(path,tf,ph.WOOD,start,controls=lambda _:((np.inf,key),),
                                search_side=False,require_groove_stability=True)
    assert end.kind=='chain_end' and end.tick>0.
    np.testing.assert_allclose(end.body.pos,p[1]); assert end.body.vel[0]>0.


def test_small_convex_seam_can_remain_supported_without_an_air_leg():
    ds=[tuple((x,-.05*(x-10.),z) for x,y,z in f) for f in quad(10.,20.,-5.,5.,0.)]
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(ds,False)]); s=conformal.build(m)
    a,b,_=next(supportededges.overlaps(roadway.Topology(s)))
    row,why=supportededges.prove(tf,tf.Obstacles(m,()),s,ph.WOOD,a,b)
    assert why=='literal_roll' and row['mode']=='convex_supported'
    assert [x.kind for x in row['motion']]==['rolling','arc','line','line']
