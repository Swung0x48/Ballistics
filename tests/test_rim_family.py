import math
from types import SimpleNamespace

import numpy as np
import pytest

from ballistics import closedform as cf, motion, physics as ph, rimclear, runfamily, runreach, supportcarry as sc


def make_arc(along=0.):
    return motion.arc(np.zeros(3),np.array([0.,2.,0.]),np.array([2.,0.,0.]),
                      0.,math.pi/3,np.array([0.,0.,along]),(25.,0.,0.))


def test_carried_edge_is_not_an_incident_face_or_endpoint():
    triangle = np.array([[0.,0.,-3.],[0.,0.,3.],[0.,-4.,0.]])
    path = make_arc()
    hits = [(f,rimclear.feature_minimum(path,triangle,f)) for f in rimclear.features(triangle)]
    possible = [(f,q) for f,q in hits if q]
    assert len(possible)==1
    f,q=possible[0]
    assert f[0]=='edge' and rimclear.support_edge(triangle[list(f[1])],triangle[:2])
    assert q['distance']==pytest.approx(2.)
    assert not rimclear.support_edge(triangle[:2],triangle[:2]/2)
    shifted=triangle.copy(); shifted[:,2] += 4.
    j=next(f for f in rimclear.features(shifted) if f[0]=='vertex' and f[1]==(0,))
    q=rimclear.feature_minimum(path,shifted,j)
    assert q['distance']==pytest.approx(math.sqrt(5.))


def test_rim_face_checks_its_interior_and_detects_between_endpoints():
    angle=.43
    n = np.array([math.sin(angle),math.cos(angle),0.]); m=np.array([n[1],-n[0],0.])
    a=3*n; z=np.array([0.,0.,1.])
    tri=np.array([a-4*m-5*z,a+4*m-5*z,a+7*z])
    f=next(x for x in rimclear.features(tri) if x[0]=='face')
    q=rimclear.feature_minimum(make_arc(),tri,f)
    assert q['distance']==pytest.approx(1.)
    assert q['parameter']==pytest.approx(angle/(math.pi/3))


def test_axial_distance_envelope_certifies_only_a_lower_bound():
    path=make_arc(3.)
    tri=np.array([[2.3,1.1,0.5],[2.4,1.1,0.5],[2.3,1.2,0.5]])
    f=next(x for x in rimclear.features(tri) if x[0]=='vertex' and x[1]==(0,))
    q=rimclear.feature_minimum(path,tri,f)
    assert q['actual_distance'] is not None
    assert q['distance']<q['actual_distance']


def test_joint_family_reuses_same_tick_at_fractional_concave_seam():
    flat=SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.)
    normal=sc.unit(np.array([-.1,1.,.02])); uphill=SimpleNamespace(normal=normal,offset=0.)
    wood=ph.WOOD; r=wood.radius+sc.SKIN
    R=runreach.PlaneReach.make(wood,flat,np.zeros(3),np.array([4.,0.,2.]))
    S=runreach.PlaneReach.make(wood,uphill,np.zeros(3),np.zeros(3))
    key=np.array([[1.,0.],[-1.,1.],[0.,-1.]])
    F=runfamily.Family.start(R,3).advance(R,1.7)
    base=F.center+F.image@key.ravel()
    a=base[:3]-[0.,0.,1.]; b=base[:3]+[0.,0.,1.]
    G=F.on_edge(a,b).transfer(R.normal,uphill,r).advance(S,3.)
    p,v=R.origin,R.velocity
    for duration,u in ((1.,key[0]),(.7,key[1])):
        dx,v=cf._motion(duration,v,R.gravity+R.axes@u,R.loss); p=p+dx
    np.testing.assert_allclose(base,np.r_[p,v],atol=1.e-11)
    p=p-r*R.normal; p=p+normal*(r-normal@p); v=v-normal*(v@normal)
    for duration,u in ((.3,key[1]),(1.,key[2])):
        dx,v=cf._motion(duration,v,S.gravity+S.axes@u,S.loss); p=p+dx
    np.testing.assert_allclose(G.center+G.image@key.ravel(),np.r_[p,v],atol=1.e-11)
    solved=G.solve(velocity=v)
    assert solved is not None
    np.testing.assert_allclose(solved[0][[3,5]],v[[0,2]],atol=1.e-9)
    assert G.solve(velocity=[100.,0.,100.]) is None


def test_literal_keys_round_prefix_and_leave_integer_switches():
    fractional=((4.52,np.array([.206,.988])),(6.,np.array([-.5,1.])),(10.2,np.array([-1.,-.4])))
    runs=runfamily.literal_schedule(fractional)
    assert runs[-1][0]==11 and all(isinstance(t,int) for t,_ in runs)
    times,inp=zip(*runs)
    commands=np.repeat(inp,np.diff([0,*times]),axis=0)
    assert np.max(abs(commands))<=1
    for t,value in enumerate(np.cumsum(commands,axis=0),1):
        ends=np.array([4.52,6.,11.]); starts=np.r_[0.,ends[:-1]]
        area=np.clip(t-starts,0,ends-starts)@np.array([k for _,k in fractional]); assert np.max(abs(value-area))<=.5+1.e-10

