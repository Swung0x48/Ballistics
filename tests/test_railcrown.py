import math

import numpy as np
import pytest

from ballistics import railcrown as rc, rails


def test_crown_is_unstable_not_a_lateral_snap_band():
    assert not rc.swing(0.,0.,20.,0.,2.35,.2,-.94)
    for x in (1.e-4,.2):
        out=rc.swing(x,0.,20.,0.,2.35,.2,-.94)
        assert out[-1].event=='pressure'
        assert out[-1].end>x
        assert out[-1].end_speed>0.
        inn=rc.swing(-x,0.,20.,0.,2.35,.2,-.94)
        assert inn[-1].end< -x
        assert inn[-1].end_speed<0.


def test_crown_energy_turn_and_inward_contact_are_analytic():
    t=-.25;r=2.35;k=.2;g=20.
    a=rc.swing(t,.1,g,0.,r,k,-math.asin(1.75/r))
    assert a[0].event=='turn'
    assert a[0].barrier==pytest.approx(g*r*(1-math.cos(t)))
    assert a[-1].event=='twin'
    t1=a[-1].end
    assert r*math.sin(t1)==pytest.approx(-1.75)
    assert rc.swing(.1,20.,g,0.,r,k,-1.)[0].end==pytest.approx(.1)
    assert rc.swing(0.,0.,g,2.,r,k,-1.)[-1].end>0.


def test_paired_tube_metadata_keeps_axes_not_sampled_nodes():
    seg=np.array([[0.,0.,0.,45.,0.,0.,.35],[0.,0.,3.8,45.,0.,3.8,.35]])
    pair=next(l for l in rails.lanes(seg) if l['double'])
    assert len(pair['points'])>40
    assert len(pair['segments'])==2
    assert all(len(a)==2 for a in pair['axes'])
    np.testing.assert_allclose(np.linalg.norm(pair['axes'][0]-pair['axes'][1],axis=1),3.8)



def test_normal_collision_preserves_tangent_and_spin_with_paper_restitution():
    from ballistics import closedform as cf, physics as ph
    n=np.array([0.,1.,0.])
    s=ph.State(np.zeros(3),np.array([14.,-4.,2.]),np.array([.1,0.,-.3]))
    for ball,scale in ((ph.PAPER,.9),(ph.WOOD,1.),(ph.STONE,1.)):
        out,kind=cf.collision_impulse(ball,s,n,.3)
        assert kind=='normal_only'
        np.testing.assert_array_equal(out.omega,s.omega)
        np.testing.assert_array_equal(out.vel[[0,2]],s.vel[[0,2]])
        assert out.vel@n==pytest.approx(4*scale*math.sqrt(ball.elasticity*.3))


def test_paper_direction_cone_encloses_explicit_normal_and_oblique_impulses():
    from ballistics import closedform as cf, paperimpact as pi, physics as ph
    v=np.array([14.,-10.,-.2]); n=np.array([0.,1.,0.]); p=np.array([5.,0.,2.]); cone=pi.Cone(p,v,n,.3,'rail_end')
    _,_,vlo,vhi=cone.ranges()
    for d in (n, np.array([.6,.8,0.]), np.array([-.6,.8,0.]), np.array([0.,.8,-.6])):
        out,_=cf.collision_impulse(ph.PAPER,ph.State(p,v),d,.3)
        assert np.all(vlo<=out.vel) and np.all(out.vel<=vhi)
        impulse=cf.air_impulse(ph.PAPER,push=np.array([1.,0.]))
        plo,phi,_,_=cone.at_tick(10.,impulse)
        dx,_=cf._motion(10.,out.vel,impulse,ph.PAPER.lin_damp*ph.DT)
        assert np.all(plo<=p+dx) and np.all(p+dx<=phi)


def test_arrival_frontier_requires_witnessed_coverage_and_same_history():
    from ballistics.railride import ArrivalFrontier
    f=ArrivalFrontier(); visits=frozenset({(1,1)}); objects=frozenset({1}); k=(1,2,3)
    assert f.accept(k,2.,np.array([0.,0.,0.]),np.array([1.,1.,1.]),visits,objects)
    assert not f.accept(k,3.,np.array([.2,.3,.4]),np.array([.5,.6,.7]),visits,objects)
    assert f.accept(k,3.,np.array([.2,.3,.4]),np.array([2.,.6,.7]),visits,objects)
    assert f.accept(k,3.,np.zeros(3),np.zeros(3),frozenset(),objects)
    f=ArrivalFrontier()
    for j in (0.,1.,.5):  # two disjoint witnesses do not cover the interior
        v=np.full(3,j)
        assert f.accept(k,2.,v,v,visits,objects)


def test_confirmed_road_goal_skips_only_road_aim_not_rail_relay():
    from types import SimpleNamespace as NS
    from ballistics import railride as rr, physics as ph
    s=ph.State(np.zeros(3),np.array([1.,0.,0.])); c=rr.Catch(0,0,1,s,0.,0)
    base=rr.Departure(s,0.,'chain_end',c,(),(),np.zeros(2),aimed=1)
    aims=(base, rr.replace(base,joint=(1,0)), rr.replace(base,aimed=2))
    called=[]; stats=rr.defaultdict(int)
    r=NS(obs=NS(mesh=NS(owner=np.array([0]))),g=NS(info={'rest_places':3}),labels=np.array([1,2,3]),
         walk=lambda *a: iter(aims),fixed_aims=lambda *a: (),air=lambda tf,ball,d,*a:called.append(d))
    assert list(rr.RailPaths.search(r,None,ph.PAPER,c,confirmed={2},stats=stats))==[]
    assert called==list(aims[1:])
    assert stats['rail_confirmed_aim_skipped']==1
