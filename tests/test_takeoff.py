import numpy as np
import pytest

from ballistics import closedform as cf, coarse, takeoff as tf
from ballistics.physics import BALLS, PAPER, WOOD, KEY_RIGHT, KEY_UP
from test_coarse import mesh, quad


def platforms(drop=8, x=13, z=0, wall=False, boxes=()):
    parts = [(quad(0, 9, 0, 5, 0), False), (quad(x, x + 4, z, z + 5, -drop), False)]
    if wall:
        a, b, c, d = [(11, y, z) for y, z in ((-30, -30), (30, -30), (30, 30), (-30, 30))]
        parts.append(([(a, b, c), (a, c, d)], False))
    m = mesh(parts)
    g = coarse.build(m)
    tf.add_launches(g, m, dict(death_boxes=boxes))
    return g, [e for (u, v) in g.info['launches'] if g.pos[u, 1] > 0 and g.pos[v, 0] >= x
               for e in tf.variants(g, u, v)]


@pytest.mark.parametrize('ball', list(BALLS.values()), ids=lambda ball: ball.name)
def test_all_balls_can_leave_a_ledge(ball):
    _, edges = platforms(drop=19, x=12)
    assert any(e['ball'] == ball.name for e in edges)


def test_landing_without_a_time_cap_and_with_a_lateral_input():
    _, edges = platforms(drop=355, x=260, z=185)
    paper = [e for e in edges if e['ball'] == 'paper']
    assert paper and max(e['flight_ticks'] for e in paper) > 1700
    for e in paper:
        assert np.max(np.abs(e['keys_in_air'])) <= 1
        assert abs(e['keys_in_air'][1]) > .1
        assert e['time'] == pytest.approx(e['ticks'] / 132)


def test_in_air_control_is_not_limited_to_holding_the_runup_keys():
    d = np.array([1., 0.])
    q = np.array([[8., -20., 8.]])
    lo, hi = tf.reachable(WOOD, q, d, 0., 12.)
    assert 0 <= lo[0] <= hi[0]
    assert np.all(np.abs(q[:, 2]) > 0)  # holding or releasing X alone has zero Z displacement
    q[0, 2] = 30.
    lo, hi = tf.reachable(WOOD, q, d, 0., 12.)
    assert lo[0] > hi[0]


@pytest.mark.parametrize('obstacle', ['wall', 'death'])
def test_intermediate_obstacle_removes_the_flight(obstacle):
    assert platforms(drop=19, x=14)[1]
    boxes = [dict(lo=[10.99, -30, -30], hi=[11.01, 30, 30])] if obstacle == 'death' else []
    assert not platforms(drop=19, x=14, wall=obstacle == 'wall', boxes=boxes)[1]


def test_runway_length_and_diagonal_push_change_launch_speed():
    def speed(L, key):
        direction = tf.ph.key_push(key, 0)
        axis = cf.rolling(WOOD, (0, 1, 0), key).along(direction)
        return tf.runway_speed(axis, L)
    assert 0 < speed(2, KEY_RIGHT) < speed(10, KEY_RIGHT) < speed(10, KEY_RIGHT | KEY_UP)
    assert speed(10, KEY_RIGHT) == pytest.approx(cf.rolling(WOOD, (0, 1, 0), KEY_RIGHT).along((1, 0, 0)).over(0, 10)[0])


def test_state_graph_uses_only_the_current_ball_variants():
    g, _ = platforms(drop=19, x=12)
    # Select a pair with a paper proposal and narrow it to that variant.
    u, v = next(k for k in g.info['launches']
                if g.pos[k[0], 1] - g.pos[k[1], 1] > 10
                and any(e['ball'] == PAPER.name for e in tf.variants(g, *k)))
    e = tf.way_for(g, u, v, 'paper')
    g.info['launches'][u, v] = dict(e, variants=(e,))
    for w, adj in enumerate(g.adj):
        g.adj[w] = [(x, c, k) for x, c, k in adj if k != 'launch' or (w, x) == (u, v)]
    rel = type('Relaxed', (), {'fwd': g.adj})()
    sg = coarse.StateGraph(rel, g, np.ones(len(g.pos), bool), np.full(len(g.pos), -1), 0., (), [], 0.)
    assert any(w == v and k == 'launch' for _, w, _, k in sg.step(coarse.PAPER, u))
    assert not any(k == 'launch' for _, _, _, k in sg.step(coarse.WOOD, u))
    assert np.isfinite(sg.togo(v)[coarse.PAPER, u])
    assert not np.isfinite(sg.togo(v)[coarse.WOOD, u])


def test_gate_condition_does_not_make_a_launch_available_to_other_balls():
    g, _ = platforms(drop=19, x=12)
    u, v = next(k for k in g.info['launches'] if any(e['ball'] == 'paper' for e in tf.variants(g, *k)))
    e = tf.way_for(g, u, v, 'paper')
    g.info['launches'][u, v] = dict(e, variants=(e,))
    empty = type('Relaxed', (), {'fwd': [[] for _ in g.pos]})()
    mechs = [dict(name='gate', type='gate', edges=[(coarse.GATE_THROUGH, u, v, tf.cost(e), 0, 'launch')])]
    tk = np.full(len(g.pos), -1)
    tk[v] = coarse.STONE
    sg = coarse.StateGraph(empty, g, np.ones(len(g.pos), bool), tk, 5., (), mechs, 0.)
    closed, opened = 2 * coarse.PAPER, 2 * coarse.PAPER + 1
    assert not sg.step(closed, u) and not sg.step(2 * coarse.WOOD + 1, u)
    assert sg.step(opened, u) == [(2 * coarse.STONE + 1, v, tf.cost(e) + 5., 'launch')]
    assert np.isfinite(sg.togo(v)[opened, u]) and not np.isfinite(sg.togo(v)[closed, u])


def test_way_chords_are_at_most_one_unit_apart():
    _, edges = platforms(drop=355, x=260, z=185)
    assert edges
    for e in edges:
        assert np.linalg.norm(np.diff(e['way'], axis=0), axis=1).max() <= 1.0001


@pytest.mark.parametrize('speed', [0., 2.])
def test_rim_release_skips_its_own_vertical_side(speed):
    upper = quad(0, 9, 0, 5, 0)
    lower = quad(9, 22, 0, 5, -6)
    a, b, c, d = [(9, y, z) for y, z in ((0, 0), (-6, 0), (-6, 5), (0, 5))]
    m = mesh([(upper + [(a,b,c), (a,c,d)] + lower, False)])
    rim = tf.rim_release(WOOD, np.array([9.,0.,2.5]), np.array([0.,1.,0.]),
                         np.array([1.,0.,0.]), np.array([speed,0.,0.]), KEY_RIGHT)
    assert rim.seconds > 0
    assert rim.start.pos[0] > 9 and rim.start.vel[1] < 0
    obs = tf.Obstacles(m, [])
    why, face, _, hit = obs.first_contact(WOOD, rim.start, np.zeros(2), 100, rim)
    assert why == 'land' and face >= 4
    assert hit[1] == pytest.approx(-6 + WOOD.radius + tf.FLOOR)


def test_first_support_face_changes_landing_node_and_wins_over_later_death():
    upper = quad(0, 18, 0, 8, 0)
    lower = quad(0, 18, 0, 8, -8)
    m = mesh([(lower + upper, False)])  # the later collision has smaller triangle IDs
    boxes = [dict(lo=[-100., -7., -100.], hi=[100., -6., 100.])]
    obs = tf.Obstacles(m, boxes)
    g = coarse.build(m)
    idx = tf.LandingPlaces(m, g.pos, [tf._support(m, p, WOOD.radius) for p in g.pos])
    start = tf.ph.State(np.array([5., 8., 3.]), np.array([3., -2., 0.]))
    why, face, elapsed, hit = obs.first_contact(WOOD, start, np.zeros(2), 200.)
    assert why == 'land' and face >= 2 and elapsed < 200.
    own = idx.near(face, hit)
    target = g.nearest((10.,2.,3.))
    assert own != target and own >= 0
    labels = tf.roll_labels(g)
    assert labels[own] == labels[target]
    assert labels[own] != labels[g.nearest((10.,-6.,3.))]


def test_wall_before_a_support_face_still_blocks():
    a, b, c, d = [(7, y, z) for y,z in ((-3.,-8.),(20.,-8.),(20.,12.),(-3.,12.))]
    m = mesh([(quad(0, 18, 0, 8, 0) + [(a,b,c),(a,c,d)], False)])
    obs = tf.Obstacles(m, [])
    start = tf.ph.State(np.array([2.5, 8., 3.]), np.array([12., 0., 0.]))
    why, _, elapsed, _ = obs.first_contact(WOOD, start, np.zeros(2), 200.)
    assert why == 'face' and elapsed < 20.


def test_rail_ridge_landing_uses_radial_normal_and_lane_place():
    # Both tessellated faces are steep, but the first point of contact is
    # the top ridge and its radial contact normal points upwards.
    a, b = (-4., 0., 0.), (4., 0., 0.)
    m = mesh([([(a, b, (4., -2., 0.4)), (a, (-4., -2., -0.4), b)], False)])
    m.rail[:] = True
    obs = tf.Obstacles(m, [])
    start = tf.ph.State(np.array([0., 5., 0.]), np.array([0., -4., 0.]))
    why, tri, _, hit = obs.first_contact(WOOD, start, np.zeros(2), 150.)
    assert why == 'land' and abs(m.N[tri, 1]) < coarse.WALK_NY
    pts = np.array([[0., WOOD.radius, 0.]])
    places = tf.LandingPlaces(m, pts, [m.contacts(pts[0], WOOD.radius)])
    assert places.near(tri, hit) == 0


def test_rail_departure_ignores_connected_support_but_not_later_same_owner():
    a, b, c, d = [(5., y, z) for y, z in ((-8., -5.), (8., -5.), (8., 5.), (-8., 5.))]
    m = mesh([(quad(-5, 0, -2, 2, 0) + [(a, b, c), (a, c, d)], False)])
    m.rail[:] = True  # one mesh object with a second, separated feature
    obs = tf.Obstacles(m, [])
    p = np.array([0., WOOD.radius + tf.FLOOR - .025, 0.])
    start = tf.ph.State(p, np.array([12., 1., 0.]))
    push = np.zeros(2)
    support = tf.rail_support(obs, p, WOOD.radius)
    clear = obs.rail_clearance(WOOD, start, push, 90., support)
    assert 0 < clear < 12
    assert obs.first_contact(WOOD, start, push, 90.)[0] == 'land'
    why, tri, elapsed, _ = obs.first_contact(WOOD, start, push, 90., departure=support, clearance=clear)
    assert why == 'face' and tri >= 2 and elapsed > clear


def test_witness_component_does_not_merge_via_roll_outside_sector():
    positions = np.array([[0., 0., 0.], [0., 0., 2.], [0., 0., 1.]])
    adj = [[(2, 1., 'roll')], [(2, 1., 'roll')], [(0, 1., 'roll'), (1, 1., 'roll')]]
    g = type('Graph', (), {'pos':positions,'adj':adj,'info':{'launch_sector_masks':[np.array([True,True,False])]}})()
    assert tf.roll_labels(g)[0] == tf.roll_labels(g)[1]
    lab = tf.witness_labels(g, {}, {}, 'wood')
    assert lab[0] != lab[1]


def test_zero_flight_and_midcourse_transform_are_not_single_kind_witnesses():
    way = np.array([[0.,2.,0.], [10.,2.,0.]])
    moving = dict(flight_ticks=20., way=way, ball='paper')
    transformer = (np.array([5.,2.,0.]), 'stone')
    assert not tf.valid_in_sector(dict(moving, flight_ticks=0.), way[-1], [], 1.)
    assert not tf.valid_in_sector(moving, way[-1], [transformer], 1.)
    assert tf.valid_in_sector(moving, way[-1], [], 1.)
    assert tf.valid_in_sector(moving, way[-1], [(way[-1], 'stone')], 1.)


def _ridge(z):
    return [((-15., 0., z), (15., 0., z), (0., -1., z))]


def _flight_trace(m, pos, vel, *, boxes=(), ticks=300, **kw):
    obs = tf.Obstacles(m, boxes)
    return tf.trace_air(obs, WOOD, tf.ph.State(np.array(pos), np.array(vel)),
                        [(ticks, np.zeros(2))], **kw)


def test_skimming_rail_bounces_to_road():
    m = mesh([(_ridge(0.), False), (quad(-20, 20, -20, 20, -6), False)])
    m.rail[0] = True
    trace = _flight_trace(m, (1., 1., 3.), (0., -1., -12.))
    assert trace.result[0] == 'land' and trace.result[1] >= 1
    assert trace.bounces == 1 and trace.contacts[0]['normal'][1] < coarse.WALK_NY
    assert trace.contacts[0]['outgoing'][2] > 0.
    assert trace.contacts[1]['tick'] > trace.contacts[0]['tick']


def test_opposing_rail_reactions_can_land_without_one_supporting_normal():
    # Start slightly closer to the first of two ridges. The side hit alters
    # lateral velocity, then the sphere enters the other ridge's contact skin
    # while still within the first's, so neither contact alone is a landing.
    m = mesh([(_ridge(-1.86), False), (_ridge(1.86), False)])
    m.rail[:] = True
    trace = _flight_trace(m, (1., 1.6, -.012), (0., -8., 0.))
    assert trace.result[0] == 'land'
    assert trace.bounces >= 1 and trace.contacts[-1]['kind'] == 'dual_rail'
    assert trace.contacts[-1]['normal'][1] < coarse.WALK_NY


def test_fourth_non_support_contact_exhausts_bounce_budget():
    def wall(x):
        a, b, c, d = [(x, y, z) for y, z in ((-500., -50.), (20., -50.), (20., 50.), (-500., 50.))]
        return [(a,b,c),(a,c,d)]
    # Separate bounces by more than one PSI for all four encounters.
    m = mesh([(wall(0.) + wall(4.25), False)])
    trace = _flight_trace(m, (2.125, 10., 1.), (30., 0., 0.), ticks=500)
    assert trace.result[0] == 'bounce_limit'
    assert trace.bounces == tf.MAX_BOUNCES and len(trace.contacts) == tf.MAX_BOUNCES + 1
    assert 'outgoing' not in trace.contacts[-1]


def test_death_after_rail_bounce_precedes_landing():
    m = mesh([(_ridge(0.), False), (quad(-20, 20, -20, 20, -6), False)])
    m.rail[0] = True
    boxes = [dict(lo=[-20., -2., -20.], hi=[20., -1., 20.])]
    trace = _flight_trace(m, (1., 1., 3.), (0., -1., -12.), boxes=boxes)
    assert trace.bounces == 1 and trace.result[0] == 'death_box'


def test_support_capture_preserves_tangent_and_measured_paper_ratio():
    from ballistics import supportcarry as sc
    n = np.array([0.,1.,0.]); v = np.array([12.,-3.,4.])
    assert sc.rolling_entry(PAPER,v,n) == pytest.approx([12/1.277,0,4/1.277])
    assert sc.rolling_entry(WOOD,v,n) == pytest.approx([12/1.4,0,4/1.4])
    rolling = tf.ph.rolling_spin(np.array([12.,0.,4.]),n,PAPER.radius)
    assert sc.rolling_entry(PAPER,v,n,rolling) == pytest.approx([12,0,4])


def test_supported_law_reaches_end_in_one_closed_form_piece():
    from ballistics import supportcarry as sc
    tangent=sc.unit(np.array([1.,-.5,0.]))
    n=sc.unit(np.array([.5,1.,0.]))
    axis=sc.law(PAPER,n,np.array([1.,0.])).along(tangent)
    v,seconds=axis.over(10.,20.)
    x,s=axis.at(10.,seconds/tf.ph.DT)
    assert x==pytest.approx(20.) and v==pytest.approx(s) and v>10.


def test_double_rail_energy_and_convex_pressure_are_speed_dependent():
    from ballistics import supportcarry as sc
    n=np.array([0.,1.,0.]); t=np.array([1.,0.,0.]); q=np.zeros(3); push=np.zeros(2)
    assert sc.lateral_exit(PAPER,3.8,.4,1.,n) is None
    assert sc.lateral_exit(PAPER,3.8,.4,10.,n)[2] < 10.
    nn=sc.unit(np.array([.4,1.,0.]))
    r=tf.convex_release(PAPER,q,n,nn,t,14*t,push,.277)
    assert r is not None and r.seconds==0


def test_carry_rail_end_then_air_then_coast_to_coarse_floor():
    from ballistics import supportcarry as sc
    # A simple rail fitting above a longer receiving road.
    tris = [(a,b,c) for a,b,c in _ridge(0.)]
    m=mesh([(tris,False),(quad(4,40,-8,8,-6),False)])
    m.rail[0]=True
    obs=tf.Obstacles(m,[])
    points=np.array([[0.,2.,0.],[2.,2.,0.],[4.,2.,0.],[19.,-4.,0.]])
    g=type('Graph',(),dict(pos=points,lane=np.array([0,0,0,-1]),
        lanes=[dict(double=False,gauge=0.,radius=.3,segments=points[[0,2]])]))()
    class Landing:
        def near(self,face,hit):
            return 3 if face>0 else 2
    # Keep actual mesh discontinuity at x=4, not one infinite ridge.
    m.A[0]=[-6.,0.,0.]; m.B[0]=[4.,0.,0.]
    obs=tf.Obstacles(m,[])
    paths=sc.SurfacePaths(g,obs,Landing())
    last=dict(kind='land',triangle=0,position=points[0],normal=np.array([0.,1.,0.]),
              incoming=np.array([14.,-.1,0.]))
    out=tf.carry_contact(paths,PAPER,last,0,coast=True)
    assert out['reason']=='handoff' and out['node']==3
    assert {'rail','flight','floor'} <= {s['kind'] for s in out['pieces']}
    assert out['pieces'][0]['start_velocity'][0] > 10.

    # The concatenated witness is a real new graph edge, not just a trace.
    initial = dict(ball='paper', ticks=30., flight_ticks=30., time=30./132., keys_in_air=np.zeros(2),
                   contacts=(dict(last,object='rail',tick=30.),), bounce_count=0,
                   launch_position=np.array([-2.,5.,0.]), launch_velocity=np.array([14.,-3.,0.]),
                   landing_position=points[0],landing_face=0,way=np.vstack([[-2.,5.,0.], points[0]]))
    g.adj=[[], [], [(0,tf.cost(initial),'launch')], []]
    g.info={'launches':{(2,0):initial}}
    tf.add_continuations(g,m,{},obs=obs,landing=Landing())
    joined=tf.way_for(g,2,3,'paper')
    assert joined['segments'][0]['kind']=='flight' and joined['first_landing_node']==0
    assert joined['time']==pytest.approx(sum(s['time'] for s in joined['segments']))
    assert any(v==3 and k=='launch' for v,_,k in g.adj[2])


def test_replay_impulse_has_fitted_normal_rule_and_distinct_wall_fallback():
    from ballistics import contactimpulse as ci
    n=np.array([0.,1.,0.]); props={'friction':.7,'elasticity':.3}; p=np.zeros(3)
    weak=tf.ph.State(p,np.array([0.,-.3,0.]))
    strong=tf.ph.State(p,np.array([0.,-4.,0.]))
    assert ci.resolve(WOOD,weak,n,props).vel[1]==pytest.approx(0.)
    assert ci.resolve(WOOD,strong,n,props).vel[1]==pytest.approx(4*np.sqrt(.06))
    assert ci.resolve(PAPER,weak,n,props).vel[1]==pytest.approx(.3*.9*np.sqrt(.12))
    sliding=tf.ph.State(p,np.array([12.,-4.,0.]))
    assert ci.resolve(PAPER,sliding,n,props,support=False).vel[0]==pytest.approx(12.)


def test_replay_impulse_coulomb_filter_and_spin_conserve_contact_momentum():
    from ballistics import contactimpulse as ci
    n=np.array([0.,1.,0.]); p=np.zeros(3); material={'friction':.7,'elasticity':.3}
    v=np.array([18.,-6.,0.]); s=tf.ph.State(p,v)
    out=ci.resolve(PAPER,s,n,material,rail=True)
    cap=ci.resolve(PAPER,s,n,material,rail=True,cap_sample=True)
    dn=(out.vel-v)@n; dv=out.vel-v-dn*n
    k=ci.inertia_ratio(PAPER)
    assert np.linalg.norm(dv)==pytest.approx(.25*.35*dn)
    assert out.omega == pytest.approx(np.cross(-n,dv)/(k*PAPER.radius))
    assert abs(cap.vel[0]-v[0]) <= .35*dn + 1e-9
    assert cap.vel[0] < out.vel[0]
    with_spin = tf.ph.State(p,v,tf.ph.rolling_spin(np.array([18.,0.,0.]),n,PAPER.radius))
    assert ci.resolve(PAPER,with_spin,n,material,rail=True).vel[0] == pytest.approx(18.)


def test_rail_envelope_inverts_a_continuous_release_interval():
    from ballistics import rail_inverse as ri
    # Descending vertical reach is identical at every release station. The
    # closed-form interval thus has an independently computable exact width.
    a=np.array([0.,12.,0.]); v=np.array([9.,0.,0.]); zero=np.zeros(3)
    law=ri.ReleaseLaw(a,v,np.array([.1,0.,0.]),zero,zero,-.02)
    A,B,_=ri.contact_time(tf,PAPER,law.at(0.),np.zeros(3))
    q=np.array([25.+A*v[0],0.,0.]); width=B*PAPER.push
    x0, x1=(25.-width)/.1,(25.+width)/.1
    assert x0>0 and x1<500
    xs=ri.invert(tf,PAPER,law,0.,500.,q)
    assert len(xs)==1 and xs[0]==pytest.approx((x0,x1),abs=.002)
    assert not ri.invert(tf,PAPER,law,0.,500.,q+np.array([0.,0.,width+1.]))


def test_paired_rails_on_a_steep_slope_use_the_local_cross_section():
    from ballistics import railride as rr
    # World-horizontal projections are not opposed enough at this slope;
    # the two true lateral reactions cancel in the rail's cross-section.
    t=np.array([.8,-.6,0.]); n=rr.support_normal(t)
    a=.45*n+np.sqrt(1.-.45**2)*np.array([0.,0.,1.]); b=a.copy(); b[2]*=-1
    assert rr.sc.unit(a[rr.XZ]) @ rr.sc.unit(b[rr.XZ]) > -.9
    from types import SimpleNamespace
    cs=[SimpleNamespace(triangle=i,point=np.array([0.,0.,2.*i]),normal=v) for i,v in enumerate((a,b))]
    fake=SimpleNamespace(obs=SimpleNamespace(mesh=SimpleNamespace(rail=[True,True],contacts=lambda *_:cs)))
    assert rr.RailPaths.geometry_support(fake,np.zeros(3),PAPER,True,t)==0


def test_transverse_rail_escape_uses_energy_and_normal_load():
    from ballistics import railride as rr
    zeros=np.zeros((0,3)); lane=rr.Lane(np.array([],int),zeros,np.array([]),zeros,True,.35,3.8)
    n=np.array([0.,1.,0.]); side=np.array([0.,0.,1.])
    assert rr.crest(PAPER,lane,n,side,0.,np.zeros(2)) is None
    arc=rr.crest(PAPER,lane,n,side,1.,np.array([1.,1.]))
    assert arc is not None and arc[-1]>0 and arc[2][2]>0
