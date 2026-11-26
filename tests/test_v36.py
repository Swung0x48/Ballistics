from types import SimpleNamespace

import numpy as np
from shapely import Point
from ballistics import blockroute, coarse, conformal, conformalsearch, physics as ph, railride, supportcarry as sc, takeoff
from test_coarse import mesh, quad
from test_v35 import feature, pair


def test_standing_range_clips_wall_without_a_lattice():
    floor = quad(0, 12, 0, 12, 0.)
    wall = np.array([[[0.,0.,0.],[0.,0.,12.],[0.,5.,12.]],[[0.,0.,0.],[0.,5.,12.],[0.,5.,0.]]])
    m = mesh([(floor,False),(wall,False)])
    s = conformal.build(m)
    ranges = blockroute.stand_ranges(s,m,None)['wood']
    p = ranges[0]
    assert not p.covers(Point(.5,6.))
    assert p.covers(Point(3.,6.))
    g = blockroute.geometry_graph(s,[],{})
    assert g.nearest(np.array([11.9,2.01,11.9]),.1)==0


def test_native_crown_impact_does_not_cancel_tangential_speed_due_to_spin():
    n = np.array([0., .8, .6]); r = ph.WOOD.radius+.35+sc.SKIN
    h = feature(-1.,n,1); v=np.array([14.,-.4,1.])
    event = dict(h,position=np.array([5.,0.,-1.])+r*n,incoming=v,
                 first_rail_feature=h,coactive_rail_features=(h,))
    c = pair().paired_capture(ph.WOOD,event,np.array([0.,-5.,1.]),0,(h,))
    np.testing.assert_allclose(c.body.vel,v-(v@n)*n,atol=1e-12)


def test_mechanism_bridge_does_not_copy_ordinary_host_roll():
    m=mesh([(quad(0,4,0,4,0.),False),(quad(6,10,0,4,0.),False)])
    s=conformal.build(m); base=blockroute.geometry_graph(s,[],{})
    host=coarse.Graph(base.pos.copy(),base.kind.copy(),base.lane.copy(),
                     [[(1,5.,'roll')],[]],[],dict(terrain_support=((0,),(1,))))
    old=SimpleNamespace(glob=np.arange(2),_base=[(0,1,5.,-1,0,0,-1)],_base_kind=['roll'],
                        _lift=[],_cond=[],_cond_kind=[],mechs=[],tcost=0.,bridges=[],floats=[],pendulums=[],timed=[],float_row_names=set())
    info=dict(start=base.pos[0],reset_points=[base.pos[0]],checkpoints=[],airship=base.pos[-1],placements=[])
    sg=blockroute.mechanism_state(host,old,base,info,1,[],4.,0.,original_host_size=2)
    assert not any(k=='roll' for _,_,k,_,_ in sg.fwd[0])
    assert sg.mechanism_bridge['mapped_host_anchors']==2


def test_retained_shared_edge_is_routable_but_not_current_certification(tmp_path):
    import json
    from ballistics import retained
    m=mesh([(quad(0,4,0,4,0.),False),(quad(6,10,0,4,0.),False)])
    s=conformal.build(m);g=blockroute.geometry_graph(s,[],{})
    a,b=g.pos.tolist();p=tmp_path/'archive.json'
    witness=dict(source_patch=0,target_patch=1,origin=a,launch=a,position=b,velocity=[1.,0.,0.],
                 ticks=12.,flight_ticks=12.,motion=[['line',a,[b[i]-a[i] for i in range(3)],[0,0,0],0.,1.,[]]])
    p.write_text(json.dumps(dict(map='synthetic',geometry=retained.signature(s.triangles),version='old',
                                 rows=[dict(ball='wood',witness=witness)])),encoding='ascii')
    out=blockroute.retain_shared(g,dict(map='synthetic',retained_shared_archive=p))
    assert out['retained']==1 and g.edge_kind(0,1)=='launch'
    row,=takeoff.variants(g,0,1)
    assert row['witness_status']=='retained_verified_shared' and not row['verified'] and row['retained_reference']=='old'
