"""Rails: tube axes recovered from tessellated tubes, and the lanes a ball rides."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ballistics import rails
from ballistics.geometry import Mesh

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1_MESH = DATA / "Level_01-mesh.npz"
L1_GAME = DATA / "engine" / "L01-mesh.npz"


def tube(path, radius, sides=8, arc=2 * math.pi, up=(0.0, 1.0, 0.0)):
    """Triangles of a tube along a polyline: a ring of `sides` vertices at every point, covering `arc` of the circle
    (less than a full turn = an open profile, as rails are often modelled)."""
    path = np.asarray(path, float)
    rings = []
    for k, p in enumerate(path):
        t = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
        t = t / np.linalg.norm(t)
        u = np.cross(t, up)
        u = u / np.linalg.norm(u)
        w = np.cross(u, t)
        closed = arc >= 2 * math.pi - 1e-9
        angles = np.linspace(0, arc, sides, endpoint=not closed) + (math.pi - arc) / 2     # an open profile is centred on the top
        rings.append([p + radius * (math.cos(a) * u + math.sin(a) * w) for a in angles])
    A, B, C = [], [], []
    for a, b in zip(rings, rings[1:]):
        m = len(a)
        for i in range(m if arc >= 2 * math.pi - 1e-9 else m - 1):
            j = (i + 1) % m
            A += [a[i], a[i]]; B += [b[j], b[i]]; C += [a[j], b[j]]
    return np.array(A), np.array(B), np.array(C)


def axis_distance(points, seg):
    a, b = seg[:, :3], seg[:, 3:6]
    d = b - a
    out = []
    for p in points:
        t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        out.append(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())
    return np.array(out)


def test_a_straight_tube_gives_back_its_axis_and_radius():
    path = [(x, 5.0, 2.0) for x in np.arange(0.0, 20.1, 2.0)]
    seg, info = rails.fit_tubes(*tube(path, 0.35))
    assert info["pieces"] == 1 and len(seg) == len(path) - 1
    assert np.allclose(seg[:, 6], 0.35, atol=2e-3)
    assert axis_distance(np.array(path), seg).max() < 2e-3


def test_the_cross_section_is_a_polygon_on_a_circle_complete_or_not():
    # in the game a rail is an octagon, or its upper half (five corners) where the underside is left open; the
    # corners lie on the tube's circle either way, so the centre and radius come out exactly, while the average
    # position of a half ring's corners would be off the axis
    path = [(x, 5.0, 2.0) for x in np.arange(0.0, 20.1, 2.0)]
    for sides, arc in ((8, 2 * math.pi), (5, math.pi), (7, 1.5 * math.pi)):
        seg, info = rails.fit_tubes(*tube(path, 0.3513, sides=sides, arc=arc))
        assert info["corners_per_ring"] == {sides: len(path)}
        assert np.allclose(seg[:, 6], 0.3513, atol=1e-6)
        assert axis_distance(np.array(path), seg).max() < 1e-6


def test_a_long_stretch_between_two_rings_is_one_segment():
    seg, _ = rails.fit_tubes(*tube([(0.0, 3.0, 0.0), (20.7, 3.0, 0.0)], 0.35))
    assert len(seg) == 1 and np.linalg.norm(seg[0, 3:6] - seg[0, :3]) == pytest.approx(20.7)


def test_a_curved_tube_is_followed():
    path = [(20 * math.cos(a), 3.0 + 0.2 * k, 20 * math.sin(a)) for k, a in enumerate(np.linspace(0, math.pi / 2, 25))]
    seg, _ = rails.fit_tubes(*tube(path, 0.35))
    assert len(seg) == len(path) - 1 and np.allclose(seg[:, 6], 0.35, atol=0.01)
    assert axis_distance(np.array(path), seg).max() < 0.01


def test_two_parallel_tubes_are_a_double_rail_and_a_lone_thin_one_is_not():
    xs = np.arange(0.0, 30.1, 1.5)
    a, b = [(x, 4.0, 0.0) for x in xs], [(x, 4.0, 3.8) for x in xs]
    guard = [(x, 6.0, 12.0) for x in xs]
    tris = [tube(a, 0.35), tube(b, 0.35), tube(guard, 0.25)]
    seg, _ = rails.fit_tubes(*(np.concatenate([t[k] for t in tris]) for k in range(3)))
    ln = rails.lanes(seg)
    double = [l for l in ln if l["double"]]
    single = [l for l in ln if not l["double"]]
    assert len(double) == 1 and not double[0]["thin"] and double[0]["gauge"] == pytest.approx(3.8, abs=0.02)
    P = double[0]["points"]
    assert np.allclose(P[:, 2], 1.9, atol=0.02)                                          # between the two tubes
    assert np.allclose(P[:, 1], 4.0 + math.sqrt(2.35 ** 2 - 1.9 ** 2), atol=0.02)         # resting on both
    assert len(single) == 1 and single[0]["thin"]
    assert np.allclose(single[0]["points"][:, 1], 6.0 + 2.25, atol=0.02)                  # on top of the thin tube


def test_posts_are_not_lanes():
    post = [(5.0, y, 5.0) for y in np.arange(0.0, 8.1, 1.0)]
    seg, _ = rails.fit_tubes(*tube(post, 0.25, up=(1.0, 0.0, 0.0)))
    assert len(seg) == len(post) - 1 and rails.lanes(seg) == []


@pytest.mark.skipif(not L1_MESH.is_file(), reason="level 1 mesh not in the data folder")
def test_level_1_rails():
    m = Mesh.load(L1_MESH)
    assert len(m.tubes) > 900
    r = m.tubes[:, 6]
    assert ((np.abs(r - 0.247) < 0.01) | (np.abs(r - 0.357) < 0.012)).all()               # the two tube sizes of the game
    # every rail vertex lies on a fitted tube's surface
    V = np.unique(np.round(np.concatenate([m.A[m.rail], m.B[m.rail], m.C[m.rail]]), 4), axis=0)[::5]
    a, b = m.tubes[:, :3], m.tubes[:, 3:6]
    d = b - a
    off = []
    for p in V:
        t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        off.append(np.abs(np.linalg.norm(a + t[:, None] * d - p, axis=1) - m.tubes[:, 6]).min())
    assert np.median(off) < 0.005 and max(off) < 0.05
    ln = rails.lanes(m.tubes)
    double = sorted((l for l in ln if l["double"] and not l["thin"]), key=lambda l: -rails.length(l["points"]))
    assert len(double) >= 8
    spiral = double[0]["points"]                                                           # the spiral of sector 1: 141 long, 10 up
    assert rails.length(spiral) == pytest.approx(141.5, abs=3.0) and abs(spiral[0, 1] - spiral[-1, 1]) == pytest.approx(10.4, abs=0.5)
    if L1_GAME.is_file():                                                                  # another project's fit of the same rails
        ref = Mesh.load(L1_GAME).tubes
        mid = (m.tubes[:, :3] + m.tubes[:, 3:6]) / 2
        assert axis_distance(mid, ref).max() < 0.02                                        # nothing of mine off its axes
        assert axis_distance((ref[:, :3] + ref[:, 3:6]) / 2, m.tubes).max() < 0.02          # and nothing of its missing here
        total = lambda t: float(np.linalg.norm(t[:, 3:6] - t[:, :3], axis=1).sum())
        assert total(m.tubes) == pytest.approx(total(ref), rel=0.005)
