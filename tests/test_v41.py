import math

import numpy as np

from ballistics import motion, railtransition as tr


def pt(*v):
    return np.array(v,float)


def test_road_lip_meets_finite_rail_without_plane_coincidence():
    # Rail chart is lower than the supporting road plane; a convex lip
    # supplies the common position even though the planes do not coincide.
    a,b=pt(0,1.8,0),pt(5,1.8,0); lip=(pt(0,0,-2),pt(0,0,2))
    (s,c,q),=tr.station_roots(a,b,*lip,2.)
    np.testing.assert_allclose(c,[math.sqrt(4.-1.8**2),1.8,0.])
    np.testing.assert_allclose(q,[0.,0.,0.])
    assert 0.<s<1.


def test_lip_cylinder_cannot_extend_beyond_raw_edge():
    assert not tr.station_roots(pt(0,1.8,0),pt(5,1.8,0),pt(0,0,2),pt(0,0,3),2.)
    assert not tr.station_roots(pt(1.,1.8,0),pt(5,1.8,0),pt(0,0,-2),pt(0,0,2),2.)


def test_reverse_certificate_is_the_same_geometric_arc():
    arc=tr.rotation(pt(0,0,0),2.,pt(0,1,0),pt(.6,.8,0)); a,b=pt(1.2,1.6,0),pt(2.,1.6,0)
    tail,reverse=tr.reverse_parts((arc,(a,b)))
    assert np.array_equal(tail[0],b)
    assert isinstance(reverse,motion.Piece)
    np.testing.assert_allclose(reverse.point(0.),arc.point(1.))
    np.testing.assert_allclose(reverse.point(.35),arc.point(.65))
