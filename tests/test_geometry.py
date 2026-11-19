"""Sphere queries against the collision triangles themselves."""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from ballistics.geometry import Mesh
from ballistics.trace import read_trace

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1_MESH = DATA / "Level_01-mesh.npz"
L1_TRACE = DATA / "traces" / "SR01_1_14_457_264.txt"


def quad(x0, x1, z0, z1, y):
    """Two triangles covering a horizontal rectangle."""
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[1], p[2]), (p[0], p[2], p[3])]


def mesh(tris) -> Mesh:
    A, B, C = (np.array([t[k] for t in tris]) for k in range(3))
    return Mesh(A, B, C, np.zeros(len(A), np.int64), np.zeros(len(A), bool), np.zeros((0, 7)), [dict(name="test")])


def test_a_ball_rests_on_a_face_at_exactly_its_radius():
    m = mesh(quad(0, 20, 0, 20, 0.0))
    assert m.distance((5, 2.5, 5))[0] == pytest.approx(2.5)
    assert m.rest(5, 5, 30.0) == pytest.approx(2.0, abs=1e-6)
    c = m.contacts((5, 2.0, 5))[0]          # on the seam between the two triangles: still a face contact
    assert c.feature == "face" and c.gap == pytest.approx(0.0, abs=1e-9) and np.allclose(c.normal, (0, 1, 0))
    assert math.isnan(m.rest(50, 50, 30.0))                    # nothing under it


def test_a_kerb_holds_the_ball_up_by_its_edge():
    # a road at 0 and a kerb 0.7 high starting at x = 10: one unit before the kerb the ball cannot reach the road
    m = mesh(quad(0, 10, 0, 20, 0.0) + quad(10, 20, 0, 20, 0.7))
    y = m.rest(9.0, 10.0, 30.0)
    assert y == pytest.approx(0.7 + math.sqrt(4.0 - 1.0), abs=1e-6)       # pivoting on the kerb's edge
    c = m.contacts((9.0, y, 10.0), margin=1e-3)
    assert c and c[0].feature == "edge" and c[0].point[0] == pytest.approx(10.0) and c[0].point[1] == pytest.approx(0.7)
    # far enough from the kerb the ball is on the road; on top of it, on the kerb
    assert m.rest(7.0, 10.0, 30.0) == pytest.approx(2.0, abs=1e-6)
    assert m.rest(12.0, 10.0, 30.0) == pytest.approx(2.7, abs=1e-6)
    # the exact distance where the road takes over: the edge is no longer within the radius of a ball on the road
    reach = math.sqrt(4.0 - (2.0 - 0.7) ** 2)
    assert m.rest(10.0 - reach - 0.01, 10.0, 30.0) == pytest.approx(2.0, abs=1e-6)
    assert m.rest(10.0 - reach + 0.01, 10.0, 30.0) > 2.0


def test_a_corner_is_a_vertex_contact():
    m = mesh(quad(10, 20, 10, 20, 0.7))
    y = m.rest(9.0, 9.0, 30.0)
    assert y == pytest.approx(0.7 + math.sqrt(4.0 - 2.0), abs=1e-6)
    assert m.contacts((9.0, y, 9.0), margin=1e-3)[0].feature == "vertex"


def test_save_and_load_keep_the_triangles(tmp_path):
    m = mesh(quad(0, 10, 0, 10, 1.25))
    m.save(tmp_path / "m.npz")
    again = Mesh.load(tmp_path / "m.npz")
    assert np.array_equal(again.A, m.A) and again.entities == m.entities
    assert again.rest(5, 5, 9.0) == pytest.approx(3.25, abs=1e-6)


@pytest.mark.skipif(not (L1_MESH.is_file() and L1_TRACE.is_file()), reason="level 1 mesh export or the recorded engine trace not in the data folder")
def test_the_engine_ball_never_enters_the_exported_triangles():
    """Over a whole recorded level 1 run (20,483 frames) the ball centre is never closer than its radius to the
    exported triangles, and while it rolls it stays about a hundredth above them: the engine's contact margin."""
    m = Mesh.load(L1_MESH)
    m = m.subset(~m.placed)              # the run pushed the loose objects about; the fixed geometry is what it never entered
    _, names, P = read_trace(L1_TRACE)
    assert set(names) == {"Ball_Wood"}
    gap = m.distances(P)[0] - 2.0
    assert gap.min() > 0.005
    touching = gap[gap < 0.05]
    assert len(touching) > 0.6 * len(gap)
    assert 0.010 <= np.median(touching) <= 0.016 and np.percentile(touching, 95) < 0.03
