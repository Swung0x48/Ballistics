import json
from types import SimpleNamespace

import numpy as np

from ballistics import (conformal, conformalsearch as cs, motion, physics as ph,
                        railoutlet, regions, retained, roadway, supportededges as se, takeoff as tf)
from test_coarse import mesh,quad


def test_split_crest_releases_to_checked_receiving_patch():
    left=[tuple((x,.4*(x-10.),z) for x,y,z in f) for f in quad(0.,10.,-6.,6.,0.)]
    def right(z0,z1):return [tuple((x,-.5*(x-10.),z) for x,y,z in f) for f in quad(10.,25.,z0,z1,0.)]
    m=mesh([(left,False),(right(-6.,0.)+right(0.,6.),False)])
    s=conformal.build(m); obs=tf.Obstacles(m,()); count=0
    for a,b,span in se.overlaps(roadway.Topology(s)):
        if a[-1][0]>10.000001:a,b=b,a
        if s.patches[a[0]].normal[0]>0.:a,b=b,a
        d=[]; row,why=se.prove(tf,obs,s,ph.WOOD,a,b,details=d)
        if row is None:continue
        count+=1
        assert why=='literal_convex_flight' and row['target_patch']==b[0]
        assert row['flight_ticks']>0. and row['contacts'][-1]['tick']>row['release_tick']
        assert any(piece.kind=='air' for piece in row['motion'])
    assert count==2


def test_outlet_band_intersects_polygon_and_returns_real_runway_interval():
    s=conformal.build(mesh([(quad(0.,12.,-3.,3.,0.),False)])); p=s.patches[0]
    tangent=np.array([1.,0.,0.]); key=np.array([1.,0.]); origin=np.array([-9.,6.,0.])
    (speeds,rep,segment),=railoutlet.band(ph.WOOD,origin,tangent,key,p,(0.,20.))
    assert speeds[0]<rep<speeds[1]
    assert 0.-1.e-6<=segment[0][0]<=segment[-1][0]<=12.+1.e-6
    from ballistics import supportcarry as sc
    axis=sc.law(ph.WOOD,sc.UP,key).along(tangent)
    lo,hi,mid=railoutlet.runway_interval(axis,30.,speeds)
    assert 0.<=lo<mid<hi<=30.
    assert speeds[0]<axis.over(0.,mid)[0]<speeds[1]
    run,tick=railoutlet.integer_runway(axis,speeds,30.)
    assert int(tick)==tick and 0.<run<30.
    assert speeds[0]<axis.at(0.,tick)[1]<speeds[1]


def test_v26_rolling_certificate_is_route_usable_and_upgraded(tmp_path):
    a,b=cs.Road('wood',0),cs.Road('wood',1); layout=SimpleNamespace(patch_nodes={'wood':{0:a,1:b}})
    surface=SimpleNamespace(triangles=np.zeros((2,3,3))); shared=dict(rolling={},witnesses={},candidates={})
    g=SimpleNamespace(info=dict(conformal=surface,conformal_search=layout,shared_motion_graph=shared))
    p=tmp_path/'archive.json'; arc=motion.line(np.zeros(3),np.ones(3))
    row=dict(ball='wood',source_patch=0,target_patch=1,motion=[retained.encode_piece(arc)])
    p.write_text(json.dumps(dict(map='test',geometry=retained.signature(surface.triangles),version='v26',rows=[row]),
                            default=lambda x:x.tolist()),encoding='ascii')
    assert retained.attach_rolling(g,p,{'map':'test'})['retained']==1
    assert regions.shared_route(g,[a],[b])['edges'][0]['status']=='retained_verified_roll'
    shared['rolling'][a,b]=dict(motion=(arc,))
    assert retained.attach_rolling(g,p,{'map':'test'})==dict(retained=0,rechecked=1)
    assert regions.shared_route(g,[a],[b])['edges'][0]['status']=='verified'
