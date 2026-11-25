import numpy as np
import pytest
from shapely.geometry import Point

from ballistics import conformal as c, coarse, takeoff as tf, rail_inverse as ri, supportcarry as sc
from ballistics import physics as ph
from test_coarse import mesh, quad


def test_merge_keeps_polygon_hole_and_separate_deck():
    tiles=[t for x,z in ((0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)) for t in quad(x,x+1,z,z+1,0)]
    m=mesh([(tiles+quad(0,3,0,3,3),False)])
    s=c.build(m)
    assert len(s.patches)==2 and sorted(np.bincount(s.face_patch))==[2,16]
    floor=s.patches[s.face_patch[0]]
    assert floor.polygon.area==pytest.approx(8.)
    assert len(floor.polygon.interiors)==1 and not floor.polygon.covers(Point(1.5,1.5))
    assert sum(np.linalg.norm(a-b) for _,a,b in floor.boundary)==pytest.approx(16.)
    assert not any(not p.is_empty for p in c.footprint(floor.polygon,[(np.r_[np.eye(2),-np.eye(2)],[1.8,1.8,-1.2,-1.2])]))


def test_shared_plane_is_split_into_rolling_regions():
    m=mesh([(quad(0,10,0,4,0),False)])
    s=c.build(m)
    pos=np.array([[2.,2.,2.],[8.,2.,2.]])
    targets=c.Targets(s,pos,{0:[0,1],1:[0,1]},{'wood':np.array([10,20])},2)
    a,b=[r[-1] for r in targets.groups['wood']]
    assert a.area==b.area==pytest.approx(20.)
    assert a.intersection(b).area==0.
    assert s.node_patch.tolist()==[0,0]


def test_plane_tolerance_requires_common_plane_and_adjacent_faces():
    floor=quad(0,5,0,5,0)
    def tilted(offset):
        return [tuple(np.array([x,y+(x-5)*offset,z]) for x,y,z in t) for t in quad(5,10,0,5,0)]
    assert len(c.build(mesh([(floor+tilted(.002),False)])).patches)==1
    assert len(c.build(mesh([(floor+tilted(.008),False)])).patches)==2
    assert len(c.build(mesh([(floor+quad(6,9,0,5,0),False)])).patches)==2


def test_sloped_drop_halfplanes_keep_reachable_points():
    floor=[tuple(np.array([x,y+.2*x,z]) for x,y,z in t) for t in quad(0,10,0,7,-20)]
    patch=c.build(mesh([(floor,False)])).patches[0]
    o=np.array([0.,0.,0.]); d=np.array([1.,0.]); b=ph.WOOD; radius=b.radius
    s=.3; vmax=13.; ratio=b.push/tf.GRAVITY_IMPULSE
    xy=np.array([6.,3.]); p=np.array([xy[0],patch.height(xy,radius),xy[1]])
    lo,hi=tf.reachable(b,p-o,d,s,vmax)
    assert lo[0]<=hi[0]
    n,h=c.drift_planes(patch,radius,o,d,s,ratio,25.)
    assert np.all(n@xy<=h+c.EPS)
    tile=next(c.footprint(patch.polygon,[(n,h)]))
    witnesses=c.speed_witnesses(patch,[tile],radius,o,d,s,ratio,25.,vmax,b.lin_damp*ph.DT,ph.DT,tf.GRAVITY_IMPULSE,tf._coeff)
    assert any((lambda ab:ab[0][0]<=ab[1][0]) (tf.reachable(b,q-o,d,s,vmax)) for q in witnesses)


def test_rail_sweep_does_not_fill_diagonal_aabb_corners():
    z=np.zeros(3)
    law=ri.ReleaseLaw(np.array([0.,10.,0.]),np.array([4.,0.,4.]),np.array([1.,0.,1.]),z,z,-.12)
    b=ph.PAPER; lo,hi,(n,h)=ri.reach_planes(tf,b,law,0.,30.,0.)
    assert np.any(n@np.array([hi[0],lo[1]])>h+c.EPS)
    a,B,_=ri.contact_time(tf,b,law.at(10.),np.zeros(3))
    center=law.at(10.).pos[c.XZ]+a*law.at(10.).vel[c.XZ]
    assert np.all(n@(center+B*b.push)<=h+c.EPS)


def test_rolling_does_not_stop_at_internal_triangle_diagonal():
    m=mesh([(quad(0,5,0,4,0)+quad(5,10,0,4,0),False)])
    g=coarse.build(m); g.info['conformal']=c.build(m)
    obs=tf.Obstacles(m,[])
    class Landing:
        def near(self,*a):return 0
    paths=sc.SurfacePaths(g,obs,Landing())
    pos=np.array([1.,2.+sc.SKIN,1.]); v=np.array([15.,0.,0.]); body=ph.State(pos,v)
    r=paths.floor(tf,ph.WOOD,1,0,body,np.array([0.,1.,0.]),chosen_push=np.array([1.,0.]))
    assert r.reason=='floor_end'
    assert len([x for x in r.parts if x['kind']=='floor'])==1
    assert r.parts[0]['end_position'][0]==pytest.approx(10.)


def test_ball_centre_footprint_shifts_horizontally_on_a_slope():
    floor=[tuple(np.array([x,.25*x,z]) for x,_,z in t) for t in quad(0,10,0,5,0)]
    s=c.build(mesh([(floor,False)])); p=s.patches[0]
    r=2.
    expected=p.normal*r
    nodes=np.array([np.array([1.,.25,1.])+expected])
    t=c.Targets(s,nodes,{0:[0],1:[0]},{'wood':np.array([0])},1)
    mn=np.array([expected[0]+.01,2.]); mx=mn+.03
    cuts=[(np.r_[np.eye(2),-np.eye(2)],np.r_[mx,-mn])]
    from collections import Counter
    q=list(t.representatives('wood',r,mn,mx,lambda p: cuts,lambda q: True,stats=Counter(),prefix='test'))
    assert len(q)==1 and q[0][1][0]<0.
    assert (q[0][1]@p.normal)==pytest.approx(p.offset+r)


def test_radial_contact_outside_patch_crosses_the_touching_seam_at_entry():
    left=quad(0,6,-4,0,0)
    right=[tuple(np.array([x,.12*z,z]) for x,_,z in t) for t in quad(0,6,0,8,0)]
    m=mesh([(left+right,False)]);g=coarse.build(m);g.info['conformal']=c.build(m)
    obs=tf.Obstacles(m,[])
    class Landing:
        def near(self,*a):return 0
    paths=sc.SurfacePaths(g,obs,Landing())
    r=ph.WOOD.radius+sc.SKIN; z=.2
    body=ph.State(np.array([3.,np.sqrt(r*r-z*z),z]),np.array([0.,0.,15.]))
    carry=paths.floor(tf,ph.WOOD,0,0,body,np.array([0.,1.,0.]),chosen_push=np.array([0.,1.]))
    assert carry.reason=='floor_end'
    assert carry.parts[0]['reason']=='patch_edge' and carry.parts[0]['ticks']==0.
    assert carry.parts[1]['end_position'][2]>7.
