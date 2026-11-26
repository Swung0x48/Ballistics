import math
from types import SimpleNamespace

import numpy as np

from ballistics import physics as ph, railride, supportcarry as sc, takeoff as tf
from test_coarse import mesh, quad


def pair():
    paths = railride.RailPaths.__new__(railride.RailPaths)
    paths.obs = SimpleNamespace(mesh=SimpleNamespace(owner=[0, 0], entities=[{'elasticity': .3}]))
    t = np.array([[1., 0., 0.]])
    left = np.array([[0., 0., 1.], [10., 0., 1.]])
    right = np.array([[0., 0., -1.], [10., 0., -1.]])
    paths.lanes = {0: SimpleNamespace(p=(left + right) / 2, tangent=t),
                   1: SimpleNamespace(axis=left, twin=right, tangent=t, length=[10.], tube=.35, line='left_crown'),
                   2: SimpleNamespace(axis=right, twin=left, tangent=t, length=[10.], tube=.35, line='right_crown')}
    paths.pairs = {0: (1, 2)}
    return paths


def feature(axis_z, normal, triangle):
    n = np.asarray(normal, dtype=float)
    return dict(triangle=triangle, normal=n, point=np.array([5., 0., axis_z]) + .35 * n)


def test_native_first_tube_wins_over_nearer_up_normal_on_other_tube():
    r = ph.WOOD.radius + .35 + sc.SKIN
    n = np.array([0., .8, .6])
    first = feature(-1., n, 1)
    proxy = feature(1., [0., 1., 0.], 0)
    event = dict(proxy, position=np.array([5., 0., -1.]) + r * n,
                 incoming=np.array([9., -.3, 1.]), first_rail_feature=first,
                 coactive_rail_features=(first,))
    c = pair().paired_capture(ph.WOOD, event, np.zeros(3), 0, (first, proxy))
    assert c.lane == 2 and c.triangle == 1
    np.testing.assert_allclose(c.body.pos, event['position'])
    assert abs(c.lateral) > 0.


def test_simultaneous_distinct_tubes_can_make_groove():
    r = ph.WOOD.radius + .35 + sc.SKIN
    y = math.sqrt(r*r - 1.)
    left = feature(1., [0., y/r, -1./r], 0)
    right = feature(-1., [0., y/r, 1./r], 1)
    event = dict(right, position=np.array([5., y, 0.]), incoming=np.array([9., -.3, 1.]),
                 first_rail_feature=right, coactive_rail_features=(right, left))
    paths = pair()
    # The test uses a groove centre at the exact paired resting height.
    paths.lanes[0].p[:, 1] = y
    c = paths.paired_capture(ph.WOOD, event, np.zeros(3), 0, (right, left))
    assert c.lane == 0
    c = paths.paired_capture(ph.WOOD, dict(event, coactive_rail_features=(right, right)),
                             np.zeros(3), 0, (right, left))
    assert c.lane == 2


def test_closed_form_sweep_provides_first_feature_before_capture():
    m = mesh([(quad(-4, 4, -4, 4, 0.), False)])
    m.rail[:] = True
    start = ph.State(np.array([0., 3., 0.]), np.array([1., -9., 0.]))
    q = tf.trace_air(tf.Obstacles(m, []), ph.WOOD, start, ((100., np.zeros(2)),), accept_support=lambda _: True)
    contact, = q.contacts
    feature = contact['first_rail_feature']
    assert contact['tick'] > 0. and feature['triangle'] == q.result[1]
    np.testing.assert_allclose(feature['normal'], [0., 1., 0.], atol=1.e-12)
    np.testing.assert_allclose(feature['point'][1], 0., atol=1.e-12)



def test_supported_crown_can_release_at_start_but_not_create_zero_time_reentry(monkeypatch):
    from ballistics import sharedrail

    monkeypatch.setattr(sharedrail.cf, 'landing_step', lambda *_: 20.)
    body = ph.State(np.array([0., 2., 0.]), np.array([5., 0., 0.]))
    source = SimpleNamespace(crown={'supported': True})
    dep = SimpleNamespace(body=body, tick=0., kind='crown_pressure', source=source, pieces=())
    hit = dict(reason='land', ticks=4., face=0, contacts=(), state=body)
    paths = SimpleNamespace(obs=SimpleNamespace(bottom=-5., mesh=SimpleNamespace(rail=[True])),
                            air=lambda *a, **k: dict(hit))
    q, reason = sharedrail._trace(tf, paths, ph.WOOD, dep, np.ones(2), 0., rail_arrival=True)
    assert reason == 'rail_next_support_not_road' and q['ticks'] == 4.
    hit['ticks'] = 0.
    assert sharedrail._trace(tf, paths, ph.WOOD, dep, np.ones(2), 0., rail_arrival=True) == (None, 'rail_empty_ride')
    hit['ticks'] = 4.
    source.crown['supported'] = False
    assert sharedrail._trace(tf, paths, ph.WOOD, dep, np.ones(2), 0., rail_arrival=True) == (None, 'rail_empty_ride')
