import numpy as np

from ballistics import conformal, physics as ph, quasistatic as qs, roadway, supportededges as se, takeoff as tf
from test_coarse import mesh,quad


def test_short_bevel_needs_no_literal_runup_to_switch_support():
    upper=[tuple((x,.4*(x-10.),z) for x,_,z in f) for f in quad(10.,10.57,-3.,3.,0.)]
    m=mesh([(quad(0.,10.,-3.,3.,0.),False),(upper,False)]); s=conformal.build(m); obs=tf.Obstacles(m,())
    a,b,sp=next(se.overlaps(roadway.Topology(s)))
    for u,v in ((a,b),(b,a)):
        n=s.patches[u[0]]
        ends=np.column_stack((sp[:,0],n.height(sp),sp[:,1]))
        row,reason=qs.certificate(tf,obs,s,ph.WOOD,u,v,ends,se._xy_depth)
        assert reason=='quasistatic_concave_two_faces'
        at=row['joint_center']
        for i in (u[0],v[0]):
            patch=s.patches[i]
            assert np.isclose(at@patch.normal-patch.offset,ph.WOOD.radius+ph.Params().floor)
        assert np.array_equal(row['motion'][0].point(1.),row['motion'][1].point(0.)) or np.allclose(row['motion'][0].point(1.),row['motion'][1].point(0.))
        assert min(row['bearing_reactions_per_mass'])>0. and not row['literal_keys_rechecked']


def test_concave_certificate_does_not_waive_a_third_wall():
    upper=[tuple((x,.4*(x-10.),z) for x,y,z in f) for f in quad(10.,10.57,-3.,3.,0.)]
    a,b,c,d=((10.,y,z) for y,z in ((.5,-2.),(4.,-2.),(4.,2.),(.5,2.)))
    m=mesh([(quad(0.,10.,-3.,3.,0.),False),(upper,False),([(a,b,c),(a,c,d)],False)]);s=conformal.build(m)
    a,b,sp=next(se.overlaps(roadway.Topology(s))); p=s.patches[a[0]]
    row,reason=qs.certificate(tf,tf.Obstacles(m,()),s,ph.WOOD,a,b,np.column_stack((sp[:,0],p.height(sp),sp[:,1])),se._xy_depth)
    assert row is None and reason=='quasi_no_clear_two_foot_station'


def test_steep_source_can_roll_down_but_not_start_crossing_uphill():
    ds=[tuple((x,.5*(10.-x),z) for x,y,z in f) for f in quad(0.,10.,-5.,5.,0.)]
    m=mesh([(ds,False),(quad(10.,20.,-5.,5.,0.),False)]); s=conformal.build(m)
    a,b,_=next(se.overlaps(roadway.Topology(s)))
    if s.patches[a[0]].normal[1]>s.patches[b[0]].normal[1]:a,b=b,a
    row,reason=se.prove(tf,tf.Obstacles(m,()),s,ph.STONE,a,b)
    assert reason=='literal_gravity_downhill' and row['end_velocity'][0]>0.
    reverse=tuple((x if i not in (2,3) else a[5-i]) for i,x in enumerate(a))
    assert se.prove(tf,tf.Obstacles(m,()),s,ph.STONE,reverse,b)[1]=='source_uphill_or_crossing_requires_entry_velocity'


def test_station_interval_excludes_central_wall_without_rejecting_entire_seam():
    upper=[tuple((x,.4*(x-10.),z) for x,y,z in f) for f in quad(10.,10.57,-7.,7.,0.)]
    a,b,c,d=((10.,y,z) for y,z in ((.5,-.1),(4.,-.1),(4.,.1),(.5,.1)))
    m=mesh([(quad(0.,10.,-7.,7.,0.),False),(upper,False),([(a,b,c),(a,c,d)],False)]);s=conformal.build(m)
    a,b,span=next(se.overlaps(roadway.Topology(s)));p=s.patches[a[0]]
    row,reason=qs.certificate(tf,tf.Obstacles(m,()),s,ph.WOOD,a,b,np.column_stack((span[:,0],p.height(span),span[:,1])),se._xy_depth)
    assert reason=='quasistatic_concave_two_faces'
    assert abs(row['joint_center'][2])>2.
