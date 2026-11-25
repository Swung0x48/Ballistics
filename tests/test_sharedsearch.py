from types import SimpleNamespace

import numpy as np
from shapely.geometry import Point, Polygon, box

from ballistics import coarse, conformal, conformalsearch, physics as ph, sharedsearch, takeoff as tf, terrain
from test_coarse import mesh, quad


def test_rest_polygon_sweep_preserves_holes_until_they_are_covered():
    p=Polygon(box(0,0,8,8).exterior.coords,[box(2,2,6,6).exterior.coords])
    q=sharedsearch.minkowski_segment(p,np.array([2.,0.]))
    assert q.area==72. and not q.covers(Point(5.,4.))
    s=sharedsearch.minkowski_segment(q,np.array([0.,4.]))
    assert s.equals(box(0,0,10,12))


def test_single_block_rest_family_contains_exact_no_keys_projection():
    p=SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.,polygon=box(-2,-1,3,1))
    q,meta=sharedsearch.rest_section(p,ph.WOOD,30)
    assert q.covers(p.polygon) and q.area>p.polygon.area
    np.testing.assert_array_equal(meta['velocity_center'],np.zeros(3))


def test_chain_key_orders_pieces_without_a_node_for_every_joint():
    first,second=[conformalsearch.Rail('wood',1,'left_crown',i,-1) for i in range(2)]
    a=np.array([[1.,0,0],[0,0,0]]); b=np.array([[3.,0,0],[1,0,0]])
    groups,index=sharedsearch.chain_nodes(SimpleNamespace(rail_segments={first:a,second:b}))
    assert len(groups)==1 and index[first]==index[second]
    track=groups[index[first]]
    assert track.pieces==(second,first)
    np.testing.assert_array_equal(track.knots,[0.,2.,3.])


def test_shared_graph_does_not_project_unchecked_candidates():
    m=mesh([(quad(0,9,0,5,0),False),(quad(12,50,-10,15,-10),False)])
    g=coarse.build(m); surface=conformal.build(m); g.info['conformal']=surface; terrain.rebuild(g,m,surface)
    obs=tf.Obstacles(m,()); layout=conformalsearch.build(tf,obs,surface,lips=g.info['terrain_lips'])
    sharedsearch.add(tf,g,obs,layout)
    stats=g.info['takeoff_stats']; assert stats['candidate_edges']>0
    assert g.info['launches']
    assert stats['projected_edges']<=stats['witness_edges']<stats['candidate_edges']
    for u,v in g.info['launches']:
        for e in tf.variants(g,u,v):
            assert e['verified'] and e['flight_ticks']>0 and e['across']>0 and np.isfinite(e['down'])
            for psi,k in e['key_table']:
                assert isinstance(psi,int) and set(k)<= {-1,0,1}


def test_convex_welded_lip_kept_but_concave_side_is_not_a_release():
    g=quad(0,10,-5,5,0)
    desc=[tuple((x,-.14*(x-10),z) for x,_,z in row) for row in quad(10,20,-5,5,0)]
    m=mesh([(g+desc,False)]); s=conformal.build(m)
    l=conformalsearch.build(tf,tf.Obstacles(m,()),s)
    features=sharedsearch.release_features(s,l)
    assert [round(t,8) for rows in features.values() for _,_,_,t in rows if t is not None]==[round(np.arctan(.14),8)]*2
    # Raising the next plane produces a concave seam from either direction.
    asc=[tuple((x,.14*(x-10),z) for x,_,z in row) for row in quad(10,20,-5,5,0)]
    m=mesh([(g+asc,False)]); s=conformal.build(m); l=conformalsearch.build(tf,tf.Obstacles(m,()),s)
    assert all(t is None for rows in sharedsearch.release_features(s,l).values() for _,_,_,t in rows)
