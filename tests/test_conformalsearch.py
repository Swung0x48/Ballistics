from types import SimpleNamespace

import numpy as np

from ballistics import conformal, conformalsearch as cs, physics as ph, runfamily, runreach, takeoff as tf
from test_coarse import mesh, quad


def planes(grade=.1, wall=False):
    left=quad(0,10,-6,6,0)
    right=[tuple((x,grade*(x-10),z) for x,_,z in tri) for tri in quad(10,20,-6,6,0)]
    parts=[(left+right,False)]
    if wall:
        a,b,c,d=[(10,y,z) for y,z in ((0,-6),(0,6),(5,6),(5,-6))]
        parts.append(([(a,b,c),(a,c,d)],False))
    m=mesh(parts); surface=conformal.build(m)
    return m,surface


def test_micro_seam_merges_per_ball_without_coarse_nodes():
    m,s=planes(); layout=cs.build(tf,tf.Obstacles(m,()),s)
    assert len(s.patches)==2 and len(layout.roads)==3
    assert all(len(q.patches)==2 for q in layout.roads.values())
    assert all(len(q.rest_patches)==2 for q in layout.roads.values())


def test_wall_and_mechanism_never_merge_patches():
    m,s=planes(wall=True); layout=cs.build(tf,tf.Obstacles(m,()),s)
    assert len(layout.roads)==6 and all('mesh_blocker' in r for r in layout.verdicts.values())
    m,s=planes(); part=dict(placed=[10.,2.,0.],axes=np.eye(3),half=[.1,2.,2.])
    a=cs.build(tf,tf.Obstacles(m,()),s,parts=[part],mechanism=('closed',))
    assert all(q.mechanism==('closed',) for q in a.roads)
    assert all(q.guarded_patches and not q.rest_patches for q in a.roads.values())


def test_static_slope_is_ball_specific():
    m,s=planes(.35); layout=cs.build(tf,tf.Obstacles(m,()),s)
    assert 'static_slope' in layout.verdicts['stone']
    assert 'static_slope' not in layout.verdicts['wood']


def test_finite_worklist_only_prunes_coupled_family_coverage_and_equal_suffixes():
    a=cs.Road('wood',1); b=cs.Rail('wood',0,'groove',2,1)
    node=SimpleNamespace(normal=np.array([0.,1.,0.]),offset=0.)
    r=runreach.PlaneReach.make(ph.WOOD,node,np.zeros(3),np.zeros(3)); family=runfamily.Family.start(r,8).advance(r,8)
    broad=cs.Entry(a,'broad'); narrow=cs.Entry(a,'narrow'); unrelated=cs.Entry(a,'different_inputs'); rail=cs.Entry(b,'entry')
    domain=cs.Domain.from_family(family,('fixed_clock',8))
    small=cs.Domain.from_family(family.on_edge([0,0,0],[1,0,0]),('fixed_clock',8))
    other=cs.Domain.from_family(family,('changed_clock',8))
    work=cs.Worklist([broad,narrow,unrelated,rail],{broad:[rail],narrow:[rail],rail:[narrow,broad,unrelated]},
                     {broad:domain,narrow:small,unrelated:other})
    r=work.propagate([broad]); assert len(r['expanded'])==3 and len(r['covered'])==2
    assert r['covered'][0][2]==broad and unrelated.family in r['seen'][a]
    assert not small.covers(domain)


def test_rail_keys_have_two_directions_and_three_lines_without_point_seeds():
    m,s=planes(); axes=([[-1.,0.,-1.3],[4.,0.,-1.3],[10.,1.,-1.3]],
                        [[-1.,0.,1.3],[4.,0.,1.3],[10.,1.,1.3]])
    tracks=[dict(radius=.35,segments=np.array([[-1.,2.,0.],[4.,2.,0.],[10.,3.,0.]]),double=True,axes=axes)]
    g=cs.build(tf,tf.Obstacles(m,()),s,metadata=tracks)
    assert len(g.rail_segments)==36
    assert len([e for e in g.portals if e.kind=='rail_ring'])==18
    assert {k.line for k in g.rail_segments}=={'groove','left_crown','right_crown'}
