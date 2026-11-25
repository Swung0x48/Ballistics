from types import SimpleNamespace
import numpy as np
from ballistics import physics as ph, railjoint, sharedrail, takeoff as tf
from test_coarse import mesh, quad


def test_departure_span_cannot_waive_nearby_receiver_or_tip_successor():
    tubes=np.array([[0,0,0,2,0,0,.35],[2,0,0,4,0,0,.35],
                    [0,0,1.5,2,0,1.5,.35],[1,0,0,2,0,0,.35]])
    found=railjoint.tube_span(tubes,np.array([0.,0.,0.]),np.array([2.,0.,0.]))
    np.testing.assert_array_equal(found,tubes[[0,3]])


def test_unfitted_upward_impact_bounces_before_verified_road():
    m=mesh([(quad(-1,3,-4,4,0.),False),(quad(-10,40,-5,5,-4.),False)])
    m.rail[:2]=True;obs=tf.Obstacles(m,[])
    st=ph.State(np.array([0.,3.,0.]),np.array([12.,-9.,0.]))
    schedule=((200.,np.zeros(2)),)
    def rail_chart_accept(e):return not m.rail[e['triangle']]
    old=tf.trace_air(obs,ph.WOOD,st,schedule)
    assert old.result[0]=='land' and m.rail[old.result[1]]
    new=tf.trace_air(obs,ph.WOOD,st,schedule,accept_support=rail_chart_accept)
    assert new.result[0]=='land' and not m.rail[new.result[1]]
    assert new.contacts[0]['kind']=='rail_uncaptured_impact' and new.contacts[0]['outgoing'][1]>0
    budget=tf.trace_air(obs,ph.WOOD,st,schedule,accept_support=rail_chart_accept,max_bounces=0)
    assert budget.result[0]=='bounce_limit' and budget.bounces==0


def test_whole_tick_steering_preserves_prefix():
    a=np.array([1.,0.]);b=np.array([0.,1.]);z=np.zeros(2)
    prev=((3,a),(16,b))
    assert [t for t,_ in sharedrail._after_key(prev,8,16,z)]==[3,8,16]
    np.testing.assert_array_equal(sharedrail._after_key(prev,8,16,z)[-1][1],z)
    assert [t for t,_ in sharedrail._after_key(prev,3,16,z)]==[3,16]


def test_in_air_switch_used_at_whole_tick_before_receiving_chart(monkeypatch):
    calls=[]
    monkeypatch.setattr(sharedrail.cf,'landing_step',lambda *_:40.)
    dep=SimpleNamespace(body=ph.State(np.array([0.,2.,0.]),np.zeros(3)),kind='crown_pressure',tick=1.25,pieces=(dict(motion=()),))
    def air(_tf,_ball,_dep,*,controls,clock,diagnostics,allow_zero_rail):
        calls.append(controls(clock+_dep.tick))
        return dict(reason='land',ticks=7.,face=0,contacts=(),state=dep.body,catch=object())
    paths=SimpleNamespace(air=air,obs=SimpleNamespace(bottom=-20.,mesh=SimpleNamespace(rail=[True])))
    first=np.array([1.,0.]); second=np.array([0.,-1.]); audit={}
    q,why=sharedrail._trace(tf,paths,ph.WOOD,dep,first,2.,rail_arrival=True,flight_key=second,diagnostic=audit)
    assert why=='rail_next_support_not_road' and q['flight_switch_tick']==4
    assert calls[0][0][0]==.75 and calls[0][-1][0]==40.
    np.testing.assert_array_equal(q['arrival_key'],second)


def test_airborne_budget_only_resets_after_supported_motion():
    def dep(kind, *ticks):
        return SimpleNamespace(kind=kind, pieces=tuple(dict(ticks=t) for t in ticks))

    assert sharedrail._bounce_prefix(dep('rail_rebound'), 3) == 4
    assert sharedrail._bounce_prefix(dep('crown_twin', 0.), 3) == 3
    assert sharedrail._bounce_prefix(dep('crown_pressure', 0., 12.), 3) == 0
