import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import box

from ballistics import conformalsearch as cs,momentum,motion,physics as ph,retained,sharedsearch,takeoff as tf,targetinverse


def test_archive_keeps_witnesses_without_promoting_cone_candidates(tmp_path):
    path=tmp_path/'reference.json'; g=SimpleNamespace(pos=np.array([[0.,2.,0.],[3.,0.,0.]]),adj=[[],[]],info={'launches':{}})
    p,v=g.pos
    e=dict(ball='wood',ticks=10.,flight_ticks=9.,motion=[retained.encode_piece(motion.line(p,v))],way=[p,v],
           launch_position=p,launch_velocity=[1,0,0],landing_position=v,keys_in_air=[1.,.12])
    def plain(x):return x.tolist()
    data=dict(version='v21',map='Demo.NMO',mapping=retained.signature(g.pos),variants=[{'from':0,'to':1,'edge':e,'was_witness':True},
         {'from':1,'to':0,'edge':e,'was_witness':False}]);path.write_text(json.dumps(data,default=plain),encoding='ascii')
    result=retained.attach(tf,g,path,{'map':'Demo.NMO'})
    assert result==dict(variants=2,playable=1,unresolved=1,pairs=1)
    assert g.adj[0] and not g.adj[1]
    got=tf.variants(g,0,1)[0]; assert not got['verified'] and got['witness_status']=='retained_legacy'
    assert motion.path_for(got)[0].kind=='line'
    with pytest.raises(ValueError):retained.attach(tf,g,path,{'map':'Other.NMO'})


def test_inverse_intersects_sloping_plane_at_same_descending_height():
    n=np.array([-.11,1.,0.]);n/=np.linalg.norm(n)
    p=SimpleNamespace(normal=n,offset=0.,polygon=box(2.,-3.,5.,3.),height=lambda x,r:(r-x[0]*n[0])/n[1])
    start=ph.State(np.array([0.,18.,0.]),np.array([3.,0.,0.]))
    shape,xyz,push,t=targetinverse.polygon_slice(tf,ph.WOOD,start,p)
    assert shape.geom_type=='LineString' and t>1.
    assert xyz@n==pytest.approx(ph.WOOD.radius+ph.Params().floor)
    assert np.max(abs(push))<=1.
    a,b=tf._target_coeff(ph.WOOD,start,xyz[1]);np.testing.assert_allclose((start.pos+a*start.vel)[[0,2]]+b*ph.WOOD.push*push,xyz[[0,2]])


def test_rail_entry_does_not_erase_direction_and_arc_position():
    a,b=[cs.Rail('wood',0,'groove',0,d) for d in (1,-1)]
    seg=np.array([[0.,2.,0.],[10.,2.,0.]])
    pool,_=sharedsearch.chain_nodes(SimpleNamespace(rail_segments={a:seg,b:seg[::-1]}))
    node,interval=momentum._rail_entry(ph.WOOD,pool,dict(position=[7.,2.,0.],incoming=[4.,0.,0.]))
    assert node.direction==1 and interval==(7.,7.)
