import math
from types import SimpleNamespace

import numpy as np
import pytest

from ballistics import closedform as cf, crowncontrol, crowninlet, physics as ph, sweep


def test_corrects_an_unstable_crown_at_a_literal_integer():
    t,up,side=np.eye(3)
    tick,push,result=crowncontrol.correction(ph.WOOD,18.4,.2,.8,14.,t,up,side,2.36,-.9,12.,np.zeros(2))
    assert tick==19 and push[1]==-1.
    assert result['lateral_load_margin']>0.
    assert result['angle']<result['turn_angle']<result['saddle']
    assert crowncontrol.correction(ph.WOOD,18.4,0.,0.,14.,t,up,side,2.36,-.9,12.,np.zeros(2)) is None


@pytest.mark.parametrize('direction', [-1.,1.])
def test_correction_preserves_available_forward_input(direction):
    t,up,side=np.eye(3)
    tick,push,_=crowncontrol.correction(ph.WOOD,18.4,.2,.8,.3,direction*t,up,side,2.36,-.9,12.,np.zeros(2))
    assert tick==19
    assert tuple(push)==(direction,-1.)


def test_inverse_recovers_finite_lip_station_runway_speed_and_flight_time():
    ball=ph.WOOD;d,n,z=np.eye(3);a=np.array([3.,4.,-1.]);b=a+2*z; push=np.array([1.,0.]);t=29.3
    loss=ball.lin_damp*ph.DT; imp=cf.air_impulse(ball,push=push)
    target=sweep.curve(a+.25*(b-a),11*d,imp,t,loss,math.log1p(-loss))
    assert crowninlet.invert(ball,n,a,b,d,target,push)==pytest.approx((.25,11.,29.3))
    assert crowninlet.invert(ball,n,a,b,d,target+z*3,push) is None


def test_execution_table_checks_all_integer_transitions_not_just_the_last():
    a=np.array([1.,0.]);b=np.array([1.,-1.]);seed=dict(ticks=10.4,key_table=((11,a),))
    sections=[dict(ticks=.6,push=a),dict(ticks=6.2,push=b)]
    table=crowncontrol.execution_table(seed,SimpleNamespace(pieces=sections),b,19.8)
    assert [p[0] for p in table]==[11,20]
    sections[0]['ticks']=.5
    assert crowncontrol.execution_table(seed,SimpleNamespace(pieces=sections),b,19.8) is None
