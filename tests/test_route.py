"""The reference line file: its arrays, and that it stops at a break that was left."""
import json

import numpy as np

from ballistics import route


def stretch(x0, x1, speed=9.0):
    """A trace of a ball rolling along +x at a steady speed: rows as the kernel's traces (position first)."""
    n = int(abs(x1 - x0) / speed * 66)
    tr = np.zeros((n, 9))
    tr[:, 0] = np.linspace(x0, x1, n)
    tr[:, 1] = 2.0
    return tr


INFO = dict(start=[0.0, 3.6, 0.0], checkpoints=[[100.0, 0.0, 0.0]], airship=[200.0, 0.0, 0.0])


def result(sector, parts, solved=True, breaks=()):
    return dict(sector=sector, solved=solved, segments=parts, breaks=list(breaks))


def test_the_line_has_evenly_spaced_points_sectors_that_never_go_back_and_a_clock():
    res = [result(1, [dict(type="follow", trace=stretch(0, 100))]), result(2, [dict(type="follow", trace=stretch(100, 200, 18.0))])]
    P, sector, t, meta = route.reference_line(res, INFO, 7)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    assert P.dtype == np.float32 and sector.dtype == np.uint8 and t.dtype == np.float64
    assert np.allclose(d[:-1], 0.5, atol=1e-3) and abs(P[-1, 0] - 200.0) < 0.2
    assert (np.diff(sector.astype(int)) >= 0).all() and set(sector) == {1, 2} and (np.diff(t) > 0).all()
    assert meta["complete"] and meta["level"] == 7 and meta["version"] == 1 and meta["start"] == INFO["start"] and len(meta["checkpoints"]) == 1
    # the clock carries the speed: twice as fast in sector 2
    v1 = 0.5 / np.diff(t)[sector[1:] == 1].mean() * 1000 / 2
    v2 = 0.5 / np.diff(t)[sector[1:] == 2].mean() * 1000 / 2
    assert abs(v1 - 9.0) < 0.5 and abs(v2 - 18.0) < 1.0             # units per physics second (two per game second)


def test_the_line_ends_at_a_break_that_was_left_and_says_so():
    left = dict(status="unresolved", no="B1", near=[60.0, 2.0, 0.0], far=[70.0, 2.0, 0.0], between=["P_Ball_Paper"],
                attempts=[dict(mode="strict", tries=300000), dict(mode="free", tries=500)])
    res = [result(1, [dict(type="follow", trace=stretch(0, 60)), dict(type="assumed", no="B1", to=np.array([70.0, 2.0, 0.0])),
                      dict(type="follow", trace=stretch(70, 100))], solved=False, breaks=[left]),
           result(2, [dict(type="follow", trace=stretch(100, 200))])]
    P, sector, t, meta = route.reference_line(res, INFO, 1)
    assert P[:, 0].max() < 60.5 and set(sector) == {1}               # nothing past the break, nothing of the next sector
    assert not meta["complete"] and meta["left_breaks"][0]["no"] == "B1" and meta["left_breaks"][0]["tries"] == 300000


def test_the_file_round_trips(tmp_path):
    res = [result(1, [dict(type="follow", trace=stretch(0, 100))])]
    P, sector, t, meta = route.reference_line(res, INFO, 3)
    path = route.write_line(tmp_path, 3, P, sector, t, meta)
    assert path.name == "L03.npz"
    z = np.load(path, allow_pickle=False)
    assert sorted(z.files) == ["meta", "points", "sector", "t_ms"] and json.loads(str(z["meta"]))["points"] == len(z["points"])
