from types import SimpleNamespace
import numpy as np

from ballistics import sharedrail, targetinverse, takeoff, physics, motion


def test_crown_twin_accepts_only_checked_continuous_groove():
    dep=SimpleNamespace(kind='crown_twin',pieces=(),body=None)
    transfer=dict(ticks=0.,contacts=(),catch=object())
    paths=SimpleNamespace(twin_transfer=lambda *_:transfer)
    assert sharedrail._trace(takeoff,paths,physics.WOOD,dep,np.zeros(2),0.)==(None,'unresolved_rail_transition')
    diagnostics={}
    q,why=sharedrail._trace(takeoff,paths,physics.WOOD,dep,np.zeros(2),0.,rail_arrival=True,diagnostic=diagnostics)
    assert why=='rail_support_continues' and q['catch'] is transfer['catch']
    assert q['flight_ticks']==0. and diagnostics['trajectory']==()
    paths.twin_transfer=lambda *_:None
    assert sharedrail._trace(takeoff,paths,physics.WOOD,dep,np.zeros(2),0.,rail_arrival=True)==(None,'unresolved_rail_transition')


def test_unpaired_pipe_keeps_separating_rebound_but_supported_groove_does_not():
    paths=SimpleNamespace(lanes={0:SimpleNamespace(line='single'),1:SimpleNamespace(line='groove')})
    c=SimpleNamespace(lane=0,rebound=object(),crown=None)
    assert sharedrail._free_rebound(paths,c)
    c.rebound=None
    assert not sharedrail._free_rebound(paths,c)
    c.rebound=object(); c.lane=1
    assert not sharedrail._free_rebound(paths,c)


def test_priority_stops_literal_keys_at_actual_contact(monkeypatch):
    from shapely import Point
    priority=targetinverse.Priorities.__new__(targetinverse.Priorities)
    src,dst=0,1
    priority.points={src:{1}};priority.tried=set();priority.attempts=[];priority.rail_seeds=[];priority.rows=[]
    priority.surface=SimpleNamespace(patches=[None,None],face_patch=[-1])
    priority.layout=SimpleNamespace(patch_nodes={'wood':{1:dst}})
    monkeypatch.setattr(targetinverse,'polygon_slice',lambda *args:(Point(0.,0.),np.zeros(3),np.array([1.,0.]),6.))
    monkeypatch.setattr(targetinverse.cf,'landing_step',lambda *_:16.)
    pos=np.array([4.,0.,0.]); obs=SimpleNamespace(bottom=-10.,mesh=SimpleNamespace(owner=[0],rail=[True],entities=[{'name':'rail'}]))
    air=SimpleNamespace(result=('land',0,2.5,pos),parts=(),contacts=())
    tf=SimpleNamespace(trace_air=lambda *a,**k:air)
    seed=physics.State(np.zeros(3),np.zeros(3)); rim=SimpleNamespace(seconds=0.)
    priority.from_release(tf,physics.WOOD,obs,src,0,rim,seed,3.25,np.zeros(2),0.,(motion.line(seed.pos,seed.pos),),feature='A')
    _,w=priority.rail_seeds[0]
    assert w['ticks']==5.75 and w['key_table'][-1][0]==6
    assert np.array_equal(w['key_table'][-1][1],np.array([1.,0.]))
    priority.from_release(tf,physics.WOOD,obs,src,0,rim,seed,3.25,np.zeros(2),0.,(motion.line(seed.pos,seed.pos),),feature='B')
    assert len(priority.rail_seeds)==2



def test_zero_time_transfer_does_not_overwrite_completed_key():
    key0=np.array([1.,0.]); key1=np.array([0.,1.]); p=np.zeros(3)
    seed=dict(ticks=5.,flight_ticks=1.,contacts=({'kind':'land'},),motion=(),key_table=((5,key0),))
    dep=SimpleNamespace(kind='chain_end',pieces=(),tick=2.)
    hit=dict(ticks=0.,flight_ticks=0.,face=0,state=physics.State(p,p),segments=(),contacts=())
    row=sharedrail._held_prefix(seed,dep,hit,key1)
    assert [t for t,_ in row['key_table']]==[5,7]
    assert np.array_equal(row['key_table'][0][1],key0)
    assert np.array_equal(row['key_table'][1][1],key1)


def test_carried_release_passes_changed_flat_velocity_and_real_duration(monkeypatch):
    from shapely import box
    from ballistics.conformalsearch import Road
    k=0; source=Road('stone',k)
    patch=SimpleNamespace(polygon=box(0.,0.,2.,2.),normal=np.array([0.,1.,0.]),faces=(3,),height=lambda _:5.)
    P=targetinverse.Priorities.__new__(targetinverse.Priorities)
    P.carry_points={source:{k}}; P.rows=[]; P.rail_seeds=[]; P.surface=SimpleNamespace(patches=[patch],face_patch=[0])
    P.layout=SimpleNamespace(roads={source:SimpleNamespace(rest_patches=(k,))})
    out=physics.State(np.array([3.,7.,2.]),np.array([9.,-1.,0.]))
    leg=motion.line(np.array([1.,7.,1.]),out.pos)
    carry=SimpleNamespace(reason='floor_end',body=out,rim=object(),parts=[dict(ticks=23.,motion=(leg,),patch=0,kind='floor')])
    monkeypatch.setattr(targetinverse.sc,'SurfacePaths',lambda *_:SimpleNamespace(floor=lambda *a,**kw:carry))
    released=[]; P.from_release=lambda *args,**kw:released.append(args)
    assert P.carried_releases(takeoff,SimpleNamespace(),SimpleNamespace())=={'floor_end':9}
    assert all(args[6] is out and args[7]==23. for args in released)


def test_carried_runway_can_attach_to_same_patch_by_real_edge_contact():
    from shapely import Polygon
    from ballistics import routeprojection
    tri=np.array([[0.,0.,0.],[2.,0.,0.],[0.,0.,2.]])
    patch=SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.,faces=(0,),polygon=Polygon(tri[:,[0,2]]))
    pos=np.array([[-.2,np.sqrt(4.-.2**2),.5]])
    g=SimpleNamespace(pos=pos,lane=np.array([-1]),info=dict(terrain_support=((0,),),
            conformal=SimpleNamespace(patches=[patch],triangles=np.array([tri]))))
    A=routeprojection.Anchors(g); origin=np.array([.3,2.0099,.6]);r=2.0099
    assert A.road(0,origin,r)[0]==-1
    assert A.road(0,origin,r,bearing_anchor=True)==(0,'actual_coarse_bearing_on_patch')
    g.pos[0,1]+=.1
    assert A.road(0,origin,r,bearing_anchor=True)[0]==-1
