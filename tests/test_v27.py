from types import SimpleNamespace
import numpy as np

from ballistics import conformal, conformalsearch as cs, physics as ph, regions, roadway, supportededges as se, takeoff as tf
from test_coarse import mesh, quad


def test_overlap_clips_export_error_to_actual_span():
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),
            ([tuple((x+.001*z,z*0.,z) for x,_,z in tri) for tri in quad(10.,20.,-100.,100.,0.)],False)])
    topo=roadway.Topology(conformal.build(m)); a,b,span=next(se.overlaps(topo))
    assert np.linalg.norm(a[-1]-b[-1])<conformal.PLANE_ERROR
    np.testing.assert_allclose(np.sort(span[:,1]),[-5.,5.],atol=.001)


def test_corner_touch_is_not_a_seam():
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(quad(10.,20.,5.,15.,0.),False)])
    assert list(se.overlaps(roadway.Topology(conformal.build(m))))==[]


def test_catalogue_records_unresolved_rail_portal():
    s=SimpleNamespace(patches=[]); rail=cs.Rail('wood',0,'groove',0,1)
    portal=cs.Portal(rail,cs.Rail('wood',0,'groove',1,1),'rail_ring',None,np.zeros((2,3)))
    layout=SimpleNamespace(portals=(portal,), patch_nodes={},roads={})
    g=SimpleNamespace(info=dict(conformal=s));se.build(tf,g,None,layout,lambda *_:-1)
    row,=g.info['portal_verdicts']
    assert row['portal_index']==0 and row['status']=='unresolved' and row['reason']=='rail_ring_requires_supported_inlet'


def test_route_uses_unmapped_witness_but_not_unproved_portal():
    a,b,c=[cs.Road('wood',k) for k in range(3)]
    g=SimpleNamespace(info=dict(shared_motion_graph=dict(rolling={(a,b):{}},witnesses={},candidates={b:{c}}),
                                conformal_search=None))
    assert regions.shared_route(g,[a],[b])['nodes']==[a,b]
    assert regions.shared_route(g,[a],[c])['unresolved_frontier']==[(b,c)]


def test_uphill_entry_requirement_reaches_the_stationary_point():
    from ballistics import closedform as cf, supportcarry as sc
    normal=sc.unit(np.array([-1.,1.,0.])); forward=sc.unit(np.array([1.,1.,0.]))
    patch=SimpleNamespace(normal=normal)
    v=se.slope_requirement(ph.STONE,patch,forward,8.)
    axis=sc.law(ph.STONE,normal,np.array([1.,0.])).along(forward)
    stop=cf._turn(v,axis.impulse,axis.loss)
    assert v>0
    np.testing.assert_allclose(axis.at(v,stop)[0],8.,atol=1.e-8)
    assert axis.over(v*.99,8.) is None
    assert se.slope_requirement(ph.WOOD,SimpleNamespace(normal=np.array([0.,1.,0.])),np.array([1.,0.,0.]),8.)==0.


def test_concave_crossing_switches_at_two_offset_planes():
    ds=[tuple((x,.12*(x-10.),z) for x,y,z in f) for f in quad(10.,20.,-5.,5.,0.)]
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(ds,False)]); surface=conformal.build(m)
    a,b,_=next(se.overlaps(roadway.Topology(surface)))
    if surface.patches[a[0]].normal[1]<surface.patches[b[0]].normal[1]:a,b=b,a
    row,reason=se.prove(tf,tf.Obstacles(m,()),surface,ph.WOOD,a,b)
    assert reason=='literal_roll'
    run,transfer,_=row['motion']; end=run.point(run.end)
    assert end[0]<10.
    np.testing.assert_allclose(transfer.point(transfer.end),end,atol=1.e-7)
    for k in (a[0],b[0]):
        p=surface.patches[k]
        np.testing.assert_allclose(end@p.normal-p.offset,ph.WOOD.radius+ph.Params().floor,atol=1.e-7)


def test_microscopic_normal_noise_is_not_a_deep_concave_intersection():
    ds=[tuple((x,.003+1.e-6*(x-10.),z) for x,y,z in f) for f in quad(10.,20.,-5.,5.,0.)]
    m=mesh([(quad(0.,10.,-5.,5.,0.),False),(ds,False)]); surface=conformal.build(m)
    a,b,_=next(se.overlaps(roadway.Topology(surface)))
    if a[-1][0]>b[-1][0]:a,b=b,a
    row,reason=se.prove(tf,tf.Obstacles(m,()),surface,ph.WOOD,a,b)
    assert reason=='literal_roll'
    assert row['mode']=='concave_or_flat'


def test_aligned_road_inlet_uses_finite_chord_and_rail_support():
    from collections import Counter
    from ballistics import railride, sharedrail, sharedsearch
    m=mesh([(quad(-10.,0.,-4.,4.,0.),False),(quad(0.,10.,-.3,.3,0.),False)])
    m.rail[2:]=True
    obs=tf.Obstacles(m,()); surface=conformal.build(m); r=ph.WOOD.radius+ph.Params().floor
    lane=railride.Lane(np.array([0,1]),np.array([[0.,r,0.],[10.,r,0.]]),np.array([10.]),
                       np.array([[1.,0.,0.]]),False,.35,0.,'single',0)
    paths=SimpleNamespace(lanes={0:lane},endpoint=[(0,1,lane.p[0])],obs=obs)
    paths.geometry_support=lambda *args:railride.RailPaths.geometry_support(paths,*args)
    road=cs.Road('wood',0); rail=cs.Rail('wood',0,'single',-1,1)
    layout=SimpleNamespace(patch_nodes={'wood':{0:road}},roads={road:SimpleNamespace(rest_patches=(0,))})
    hits=list(sharedrail.road_inlets(tf,paths,surface,layout,{rail:None},{'wood':ph.WOOD},Counter()))
    src,ball,row,c,key,interval,kind=hits[0]
    assert src==road and kind=='supported_road_end' and row['origin'][0]<0.<c.body.vel[0]
    np.testing.assert_allclose(c.body.pos,lane.p[0])
    assert interval==(0.,0.) and tuple(key)==(1.,0.) and m.rail[c.triangle]


def test_step_speed_is_selected_inside_pressure_and_friction_bounds():
    from ballistics import supportcarry as sc, terrain
    h=.46
    a,b,c,d=[(10.,y,z) for y,z in [(0.,-4.),(h,-4.),(h,4.),(0.,4.)]]
    m=mesh([(quad(0.,10.,-4.,4.,0.),False),([(a,b,c),(a,c,d),*quad(10.,20.,-4.,4.,h)],False)])
    s=conformal.build(m);lips=terrain.step_lips(m,s,terrain.neighbours(s)[0]); lip,=next(iter(lips.values()))
    paths=SimpleNamespace(g=SimpleNamespace(info={'conformal':s}),obs=tf.Obstacles(m,()))
    paths.climb_lip=lambda *args:sc.SurfacePaths.climb_lip(paths,*args)
    row,reason,detail=se.step_proposal(tf,paths,ph.WOOD,lip)
    assert reason=='ideal_step_roll'
    assert 0.<detail['required_entry_speed']<detail['entry_speed']<detail['max_contact_speed']
    assert min(detail['contact_margins'])>=0.
    assert row['motion'][1].kind=='arc'


def test_rail_ring_only_certified_by_continuous_two_sided_checked_trace():
    from ballistics import sharedrail, supportcarry as sc
    a=cs.Rail('wood',0,'groove',0,1); b=cs.Rail('wood',0,'groove',1,1)
    p=np.array([0.,2.,0.]); d=np.array([1.,0.,0.]); key=np.array([1.,0.]); start=ph.State(p-d,d)
    mid=ph.State(p,d); end=ph.State(p+d,d)
    portal=dict(kind='rail_ring',source=a,target=b,segment=np.stack((p,p)),status='unresolved')
    g=SimpleNamespace(info={'portal_verdicts':[portal]})
    layout=SimpleNamespace(rail_segments={a:np.stack((p-d,p)),b:np.stack((p,p+d))})
    left=sc.segment('rail','tube',start,mid,1,key)
    right=sc.segment('rail','tube',mid,end,1,key)
    source=cs.Rail('wood',0,'groove',-1,1)
    traces=lambda sides:{(source,cs.Road('wood',0)):[dict(rail_parts=sides,key_table=((2,key),))]}
    assert not sharedrail.mark_ring_witnesses(g,layout,traces((left,)))
    assert sharedrail.mark_ring_witnesses(g,layout,traces((left,right)))==1
    assert portal['reason']=='checked_rail_ring'


def test_crown_inlet_has_supported_cross_section_state():
    from collections import Counter
    from ballistics import railride, sharedrail
    m=mesh([(quad(-10.,0.,-4.,4.,0.),False),(quad(0.,10.,-.3,.3,0.),False)])
    m.rail[2:]=True
    surface=conformal.build(m); obs=tf.Obstacles(m,()); r=ph.WOOD.radius+ph.Params().floor
    center=np.array([[0.,r,0.],[10.,r,0.]])
    axis=center-[0.,r+.35,0.]
    lane=railride.Lane(np.array([0,1]),center,np.array([10.]),np.array([[1.,0.,0.]]),
                       False,.35,3.8,'right_crown',0,axis,axis+[0.,0.,3.8])
    paths=SimpleNamespace(lanes={0:lane},endpoint=[(0,1,center[0])],obs=obs)
    paths.geometry_support=lambda *args:railride.RailPaths.geometry_support(paths,*args)
    road=cs.Road('wood',0); rail=cs.Rail('wood',0,'right_crown',-1,1)
    layout=SimpleNamespace(patch_nodes={'wood':{0:road}},roads={road:SimpleNamespace(rest_patches=(0,))})
    (_,_,_,catch,_,_,_),=sharedrail.road_inlets(tf,paths,surface,layout,{rail:None},{'wood':ph.WOOD},Counter())
    assert catch.crown['supported'] and catch.crown['angle']==0.
    _,_,_,radial,radius,_=railride.RailPaths.cross_section(paths,lane,catch.j,catch.body.pos,ph.WOOD)
    assert abs(catch.body.vel@radial)<1.e-7
    assert np.isclose(np.linalg.norm(catch.body.pos-axis[0]),radius)
