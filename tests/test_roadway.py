import numpy as np
import pytest

from ballistics import conformal, physics as ph, roadway, takeoff as tf
from test_coarse import mesh, quad


def test_roadway_partial_finite_seam_and_analytic_runway():
    lower=quad(0,10,-5,5,0)
    upper=[tuple(np.array([x,.15*(x-10),z]) for x,_,z in row) for row in quad(10,18,-2,2,0)]
    m=mesh([(lower+upper,False)]); s=conformal.build(m); top=roadway.Topology(s)
    k=int(s.face_patch[2]); r=roadway.Runway.trace(top,m,k,np.array([17.,1.05,0.]),np.array([1.,0.]),2.)
    assert r.patches==(k,int(s.face_patch[0]))
    np.testing.assert_allclose(r.arc,[0.,8.,18.])
    key=max(tf.KEYS,key=lambda q:ph.key_push(q,0)[0]-.5*abs(ph.key_push(q,0)[2]))
    end,ticks,parts=r.roll(tf,ph.WOOD,key,18.)
    assert ticks>0 and end>0
    assert len(parts)==2 and parts[0]['end_velocity'][0]>0.
    depth,time,trail=r.for_speed(tf,ph.WOOD,key,end,r.profiles(tf,ph.WOOD,key),None)
    assert depth==pytest.approx(18.) and time==pytest.approx(ticks)
    assert len(trail)==2
    a,b=r.lip[:2]
    assert not top.neighbours(k,a,b,r.points[0])


def test_parallel_boundaries_at_different_height_are_not_a_seam():
    m=mesh([(quad(0,10,-5,5,0)+quad(10,18,-2,2,.5),False)]);t=roadway.Topology(conformal.build(m))
    k=int(t.surface.face_patch[0]);_,_,a,b=t.ray_exit(k,np.array([9.,0.]),np.array([1.,0.]))
    assert not t.neighbours(k,a,b,np.array([10.,0.,0.]))


def test_arbitrary_runup_joint_support_and_conditional_controls():
    from itertools import product
    from types import SimpleNamespace
    from ballistics import runreach, closedform as cf, supportcarry as sc
    n=np.array([-.12,1.,.05]);n/=np.linalg.norm(n); patch=SimpleNamespace(normal=n,offset=0.)
    r=runreach.PlaneReach.make(ph.WOOD,patch,np.zeros(3),np.zeros(3)); ticks=4
    def explicit(schedule):
        p=r.origin.copy(); v=r.velocity.copy(); at=0
        for end,push in schedule:
            law=sc.law(ph.WOOD,n,np.array(push)); dx,v=law.at(v,end-at); p+=dx; at=end
        return p,v
    states=[explicit(enumerate(zip(x,y),1)) for x in product((-1.,1.),repeat=ticks) for y in product((-1.,1.),repeat=ticks)]
    for a,b in ((np.array([1.,2.,-4.]),np.array([-.15,0.,.2])), (np.array([3.,-2.,4.]),np.array([-.03,0.,-.06]))):
        assert r.support(ticks,a,b)==pytest.approx(max(p@a+v@b for p,v in states),abs=1.e-10)
    center,v=r.center(ticks); point=center+r.axes@np.array([.003,-.001]); v,lo,hi,_=r.slice(ticks,point)
    for choice in product((0,1),repeat=2):
        p,w=explicit(r.schedule(ticks,point,choice)); np.testing.assert_allclose(p,point,atol=1.e-10)
        np.testing.assert_allclose(r.inverse@(w-v),np.where(choice,hi,lo),atol=1.e-10)


def test_arbitrary_runup_edge_times_agree_with_exact_integer_sections():
    from types import SimpleNamespace
    from ballistics import runreach
    patch=SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.)
    r=runreach.PlaneReach.make(ph.WOOD,patch,np.zeros(3),np.array([-.5,0.,.1])); y=r.origin[1]
    a=np.array([1.7,y,-.4]);b=np.array([2.0,y,1.2]); windows=r.time_windows(a,b)
    assert windows and any(np.isinf(v) for _,v in windows)
    for t in range(1,80):  # reference queries, not a solver trajectory
        assert (r.edge(a,b,t) is not None)==any(q-1.e-7<=t<=w+1.e-7 for q,w in windows)
