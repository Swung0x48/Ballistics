import numpy as np
import pytest

from ballistics import blockroute, boxmotion as bx, coarse, conformal, physics as ph, props, regions
from test_coarse import mesh, quad

SPEC=props.bodies()['box']

@pytest.mark.parametrize('q,kind', [([5.,0.,0.],'face'),([4.,4.,0.],'edge'),([4.,4.,4.],'corner')])
def test_finite_cube_feature(q,kind):
    c=np.zeros(3); q=np.array(q); p,n,k=bx.feature(q,c,np.array(SPEC['half']))
    assert k==kind
    assert n@ (q-p)==pytest.approx(np.linalg.norm(q-p))


def test_free_box_impact_obeys_momentum_and_catalogue_restitution():
    n=np.array([1.,0.,0.]); v=np.array([-8.,0.,3.]); u=np.zeros(3)
    a=bx.impact(ph.WOOD,v,u,n,SPEC)
    assert ph.WOOD.mass*a['ball_velocity']+SPEC['mass']*a['box_velocity']==pytest.approx(ph.WOOD.mass*v)
    assert (a['ball_velocity']-a['box_velocity'])@n==pytest.approx(8*np.sqrt(ph.WOOD.elasticity*SPEC['elasticity']))
    assert a['ball_velocity'][2]==v[2]
    assert bx.impact(ph.WOOD,-v,u,n,SPEC) is None


def test_ground_constraint_and_coulomb_variant():
    n=bx.sc.unit(np.array([1.,1.,0.]));v=np.array([-8.,-3.,3.]); u=np.zeros(3)
    a=bx.impact(ph.WOOD,v,u,n,SPEC,supported=True); b=bx.impact(ph.WOOD,v,u,n,SPEC,supported=True,friction=True)
    assert a['box_velocity'][1]==b['box_velocity'][1]==0.
    assert a['ball_velocity'][2]!=b['ball_velocity'][2]


def test_slide_and_wall_are_analytic_events():
    floor=quad(-15,15,-10,10,0.)
    wall=np.array([[[9.,0.,-10.],[9.,0.,10.],[9.,8.,10.]],[[9.,0.,-10.],[9.,8.,10.],[9.,8.,-10.]]])
    m=mesh([(floor,False),(wall,False)]); f=bx.Floor(conformal.build(m),m,SPEC)
    p=np.array([4.,2.5,0.]); end,parts,case=f.move(p,np.array([14.,0.,0.]),0)
    assert parts[0].at(parts[0].ticks)[0][0]==pytest.approx(9.-2.5)
    assert end[0]<6.5
    assert case=='wall_rebound_rest'


def test_fall_selects_actual_lower_polygon():
    m=mesh([(quad(-8,0,-5,5,4.),False),(quad(0,10,-5,5,0.),False)])
    f=bx.Floor(conformal.build(m),m,SPEC); start=np.array([0.,6.5,0.])
    q,j,leg=f.fall(start,np.array([3.,0.,0.]),4.)
    assert q[0]>0. and q[1]==pytest.approx(2.5)
    assert j==1 and leg.ticks>0.


def test_box_state_is_consumed_by_forward_backward_and_region_search():
    p=np.array([[0.,2.,0.],[5.,2.,0.]])
    g=coarse.Graph(p,np.zeros(2,bool),np.full(2,-1),[[(1,5.,'roll')],[]],[],dict(rest_places=2,edge_node=np.zeros(2,bool)))
    state=coarse.StateGraph(coarse.Relaxed(g),g,np.ones(2,bool),np.full(2,-1),0.,[],
        [dict(name='box',type='loose_box',edges=[(coarse.BOX_PUSH,0,0,1.,1<<coarse.WOOD,'push_loose_box')])],0.,guards={(0,1,'roll'):(1,0)})
    assert state.step(0,0)==[(1,0,1.,'push_loose_box')]
    assert state.step(0,0,lock=1)==[]
    assert state.togo(1)[0,0]==pytest.approx(6.)
    R=regions.Regions(regions.roll_labels(state),p)
    broke,way,_=regions.connect(regions.links_of(state,R),(R.of(coarse.WOOD,0),coarse.WOOD,0),{R.of(coarse.WOOD,1)})
    assert broke==0 and any(x[3]=='push_loose_box' for x in way)


def test_flat_block_push_checks_player_braking_and_box_rest():
    m=mesh([(quad(-20,20,-20,20,0.),False)]); s=conformal.build(m)
    g=blockroute.geometry_graph(s,[],{}); g.info['standing_ranges']=blockroute.stand_ranges(s,m,None)
    b=dict(name='P_Box_01',rest=[0.,2.5,0.]); p=dict(name=b['name'],type='P_Box',sector=1)
    (row,)=bx.state_config(g,m,[b],[p],1)
    w=row['push']; assert row['status']=='verified_push' and w['end_reason']=='floor_rest'
    assert np.linalg.norm(w['box_end']-w['box_start'])>0.
    assert all(end==int(end) for end,_ in w['keys'])
