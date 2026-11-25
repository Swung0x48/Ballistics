from types import SimpleNamespace

import numpy as np
from shapely import box

from ballistics import coarse, motion, regions, routeprojection, takeoff
from ballistics.conformalsearch import Rail, Road


def graph():
    pos=np.array([[0.,2.,0.],[1.,2.,0.],[3.,2.,0.],[5.,5.,0.],[5.,1.,0.]])
    patches=[SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.,polygon=box(-1,-1,2,1)),
             SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.,polygon=box(2,-1,4,1))]
    return SimpleNamespace(pos=pos,lane=np.array([-1,-1,-1,0,1]),lanes=[dict(double=True),dict(double=False)],
        adj=[[(1,1.,'roll')],[(0,1.,'roll')],[],[],[]],
        info=dict(terrain_support=((0,),(0,),(1,)), conformal=SimpleNamespace(patches=patches),
                  shared_motion_graph=dict(witnesses={},retained_rolling={})))


def test_shared_certificate_joins_coarse_roll_without_promoting_outer_candidate():
    g=graph(); a,b,c=Road('wood',0),Road('wood',1),Road('wood',2); p,q=g.pos[[1,2]]
    w=dict(kind='roll',source_patch=0,target_patch=1,origin=p,position=q,motion=(motion.line(p,q),),quasistatic=True)
    g.info['portal_verdicts']=[dict(source=a,target=b,status='verified',witness=w),
                              dict(source=b,target=c,status='unresolved',witness=w)]
    assert routeprojection.attach(takeoff,g)=={'quasistatic_roll:mapped':1}
    rel=SimpleNamespace(fwd=g.adj)
    sg=coarse.StateGraph(rel,g,np.ones(len(g.pos),bool),np.full(len(g.pos),-1),0.,[],[],0.)
    R=regions.Regions(regions.roll_labels(sg),g.pos)
    assert R.of(coarse.WOOD,0)==R.of(coarse.WOOD,1)!=R.of(coarse.WOOD,2)
    links=regions.links_of(sg,R)
    assert regions.connect(links,(R.of(coarse.WOOD,0),coarse.WOOD,0),[R.of(coarse.WOOD,2)])[0]==0
    assert regions.connect(links,(R.of(coarse.STONE,0),coarse.STONE,0),[R.of(coarse.STONE,2)])[0] is None


def test_rail_outlet_recovers_anchor_without_source_node_and_uses_actual_contact():
    g=graph(); a,b=Rail('wood',0,'groove',-1,1),Road('wood',1)
    q=np.array([1.9,2.,0.]); p=g.pos[3]; foot=np.array([2.01,0.,0.])
    row=dict(source_patch=-1,target_patch=1,source_node=-1,origin=p,position=q,face=4,
             motion=(motion.line(p,q),),flight_ticks=20.,rail_parts=('ride',),
             contacts=(dict(triangle=4,kind='land',position=q,point=foot),))
    g.info['shared_motion_graph']['witnesses']={(a,b):[row]}
    assert routeprojection.attach(takeoff,g)=={'shared_rail:mapped':1}
    assert g.adj[3][0][0]==2 and not g.adj[4]
    anchors=routeprojection.Anchors(g)
    assert anchors.rail(Rail('wood',0,'left_crown',-1,1),p)[0]==-1
    assert anchors.road(1,q,2.)[0]==-1


def test_zero_flight_works_for_supported_certificate_not_old_unknown_edge():
    e=dict(ball='wood',flight_ticks=0.,motion=(motion.line(np.zeros(3),np.array([4.,0.,0.])),))
    assert not takeoff.valid_in_sector(e,np.array([4.,0.,0.]),[],.25)
    e['shared_supported_transfer']=True
    assert takeoff.valid_in_sector(e,np.array([4.,0.,0.]),[],.25)
    assert not takeoff.valid_in_sector(e,np.array([4.,0.,0.]),[(np.array([2.,0.,0.]),'stone')],.25)


def test_gate_clearance_is_not_shared_with_another_ball_variant():
    from ballistics import solve
    g=graph(); p=np.array([-4.,2.,0.]); q=np.array([4.,2.,0.]); g.pos[:2]=p,q
    g.nearest=lambda *_: 0
    def edge(ball,points):
        return dict(ball=ball,ticks=20.,flight_ticks=20.,motion=motion.lines(np.array(points)))
    clear=edge('wood',[p,[-4.,2.,5.],[4.,2.,5.],q]); blocked=edge('stone',[p,q])
    g.info['launches']={(0,1):dict(clear,variants=(clear,blocked))}; g.adj[0]=[(1,20.,'launch')]
    part=dict(kind='P_Modul_01_Pusher',body='gate/P_Modul_01_Pusher',placed=[0.,2.,0.],
              half=[.2,.7,.5],axes=np.eye(3))
    info=dict(placements=[dict(name='gate',sector=1,x_axis=[1.,0.,0.])])
    # More than one unit along the push axis is behind the relieved face.
    part['half'][0]=1.
    gates=solve.gate_edges(g,info,[part]); assert not clear.get('route_gates')
    assert blocked['route_gates']
    assert g.adj[0]==[(1,20.,'launch')]
    assert gates['gate']['edges']==[]  # the translated gate still intersects the stone trajectory
