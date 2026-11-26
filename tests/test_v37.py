from types import SimpleNamespace
import numpy as np

from ballistics import blockcontacts, blockroute, conformal, conformalsearch, motion, physics as ph, regions, takeoff as tf
from test_coarse import mesh, quad


def test_exact_support_vertex_can_join_two_point_touching_blocks():
    m=mesh([(quad(-5,0,-5,0,0.),False),(quad(0,5,0,5,0.),False)])
    surface=conformal.build(m); obs=tf.Obstacles(m,[])
    assert len(surface.patches)==2
    layout=conformalsearch.build(tf,obs,surface)
    g=blockroute.geometry_graph(surface,[],{})
    g.info['standing_ranges']=blockroute.stand_ranges(surface,m,layout)
    certs,_=blockcontacts.certify(tf,obs,g,layout,set())
    assert certs['wood'][0,1]['rule']=='coplanar_support_vertex'
    np.testing.assert_allclose(certs['wood'][0,1]['vertex'],np.zeros(3))


def test_convex_vertex_needs_full_normal_cone_and_feature_clearance():
    a=np.array([[[0.,0.,0.],[-3.,0.,-1.],[-3.,0.,1.]]]);b=np.array([[[0.,0.,0.],[3.,-.6,1.],[3.,-.6,-1.]]]);
    m=mesh([(a,False),(b,False)]); surface=conformal.build(m)
    obs=tf.Obstacles(m,[]); point=np.zeros(3)
    result,why=blockcontacts.vertex(tf,obs,surface,ph.WOOD,0,1,point,[0,1])
    assert why is None and result['rule']=='offset_vertex_arc'
    assert result['motion'][1].kind=='arc'
    obs=tf.Obstacles(m,[dict(lo=[-.1,1.9,-.1],hi=[.1,2.1,.1])])
    assert blockcontacts.vertex(tf,obs,surface,ph.WOOD,0,1,point,[0,1])[0] is None


def test_portal_flight_uses_checked_destination_without_claiming_roll():
    m=mesh([(quad(u,u+3,0,3,0.),False) for u in (0,6,12)])
    surface=conformal.build(m); obs=tf.Obstacles(m,[]);layout=conformalsearch.build(tf,obs,surface)
    g=blockroute.geometry_graph(surface,[],{}); p,q=g.pos[0],g.pos[2]
    w=dict(kind='air',source_patch=0,target_patch=2,origin=p,position=q,motion=(motion.line(p,q),))
    g.info.update(shared_motion_graph={'witnesses':{}},portal_verdicts=[dict(source=layout.patch_nodes['wood'][0],
                target=layout.patch_nodes['wood'][1],patch_pair=(0,1),status='verified',kind='finite_seam',witness=w)])
    blockroute.attach(g,layout)
    assert g.edge_kind(0,2)=='launch' and g.edge_kind(0,1)=='?'
    assert not g.info['block_roll_proofs']


def test_region_rail_roll_masks_are_per_ball():
    sg=SimpleNamespace(n=2,tk=[-1,-1],guards={},terrain_masks={(0,1,'rail'):1,(1,0,'rail'):1},
                       fwd=[[(1,1.,'rail',-1,0)],[(0,1.,'rail',-1,0)]])
    labs=regions.roll_labels(sg)
    assert labs[0,0]==labs[0,1]
    assert labs[1,0]!=labs[1,1] and labs[2,0]!=labs[2,1]


def test_overlapping_coplanar_objects_connect_without_welded_edges():
    m=mesh([(quad(0,6,0,6,0.),False),(quad(2,8,1,7,0.),False)])
    surface=conformal.build(m);obs=tf.Obstacles(m,[]);layout=conformalsearch.build(tf,obs,surface)
    assert len(surface.patches)==2 and not layout.seams
    g=blockroute.geometry_graph(surface,[],{});g.info['standing_ranges']=blockroute.stand_ranges(surface,m,layout)
    certs,_=blockcontacts.certify(tf,obs,g,layout,set())
    assert certs['wood'][0,1]['rule']=='parallel_offset_overlap'



def test_regional_links_do_not_offer_a_wood_only_transfer_to_other_balls():
    sg=SimpleNamespace(n=2,tk=np.array([-1,-1]),guards={},terrain_masks={(0,1,'rail'):1},
                       fwd=[[(1,1.,'rail',-1,0)],[]],lfwd=[[],[]],cfwd=[[],[]])
    R=regions.Regions(regions.roll_labels(sg),np.array([[0.,2.,0.],[1.,2.,0.]]))
    checked=list(regions.links_of(sg,R))
    assert len(checked)==1 and checked[0][5]==0
