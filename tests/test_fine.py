"""The fine level: the search finds key tapes in the kernel, and what it returns is what the kernel does."""
import numpy as np

from ballistics import fine
from ballistics import physics as ph
from ballistics.geometry import Mesh
from ballistics.kernel import OK, World


def quad(x0, x1, z0, z1, y):
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]           # wound to face up


def world(parts, **kw) -> World:
    A, B, C, owner, ents = [], [], [], [], []
    for tris in parts:
        ents.append(dict(name="part%d" % len(ents), placed=False, type=""))
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    m = Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents)
    return World.of(m, **kw)


def places(x0, x1, y):
    return np.array([(x, y, z) for x in np.arange(x0, x1 + 0.5) for z in np.arange(2.0, 6.5)])


def test_a_gap_is_crossed_by_a_flight_and_the_tape_reproduces_it():
    w = world([quad(0, 30, 0, 8, 0.0), quad(36, 60, 0, 8, -1.0)])
    s = fine.Search(w, places(44, 56, 1.0), seed=0)
    s.add_start(fine.rest_state(w, 4.0, 4.0, 5.0))
    c = s.run(max_expansions=40_000, max_seconds=120)
    assert c.found and c.stats["air_ticks"] > 10 and fine.describe(c) == "with flight or impact"
    again = fine.replay(w, ph.WOOD, ph.Params(), ph.State(c.start, np.zeros(3), np.zeros(3)), c.tape)
    assert again["status"] == OK and np.allclose(again["pos"], c.end) and len(c.trace) == c.ticks == sum(n for _, n in c.tape)
    assert c.end[0] > 42 and abs(c.end[1] - 1.0) < 0.2


def test_rolling_is_reported_as_rolling():
    w = world([quad(0, 40, 0, 8, 0.0)])
    s = fine.Search(w, places(28, 34, 2.0), seed=0)
    s.add_start(fine.rest_state(w, 4.0, 4.0, 5.0))
    c = s.run(max_expansions=20_000, max_seconds=60)
    assert c.found and fine.describe(c) == "rolling only" and c.stats["impact_ticks"] == 0


def test_a_wall_is_not_crossed_and_that_is_reported_as_not_found():
    riser = [(np.array((30.0, 0.0, 0.0)), np.array((30.0, 6.0, 0.0)), np.array((30.0, 6.0, 8.0))),
             (np.array((30.0, 0.0, 0.0)), np.array((30.0, 6.0, 8.0)), np.array((30.0, 0.0, 8.0)))]
    w = world([quad(0, 30, 0, 8, 0.0) + riser, quad(30, 60, 0, 8, 6.0)])
    s = fine.Search(w, places(40, 50, 8.0), seed=0)
    s.add_start(fine.rest_state(w, 4.0, 4.0, 5.0))
    c = s.run(max_expansions=4_000, max_seconds=60)
    assert not c.found and c.best_distance > 5.0 and c.expansions == 4_000
