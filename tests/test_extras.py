"""Score balls and life balls along a route (ballistics.extras): the rules on made-up paths. The comparison with
the original engine's records needs game data and is not in the repository."""
import copy

import numpy as np

from ballistics import extras, things

INFO = dict(checkpoints=[[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
            placements=[dict(name="P1", type="P_Extra_Point", sector=1, position=[20.0, 2.0, 0.0]),
                        dict(name="L1", type="P_Extra_Life", sector=1, position=[50.0, 2.0, 0.0]),
                        dict(name="P2", type="P_Extra_Point", sector=2, position=[60.0, 2.0, 0.0]),
                        dict(name="P3", type="P_Extra_Point", sector=3, position=[230.0, 2.0, 0.0]),
                        dict(name="B", type="P_Box", sector=1, position=[30.0, 2.0, 0.0])])


def line(x0, x1, z=0.0, y=2.0, step=0.2):
    x = np.arange(x0, x1, step)
    return np.stack([x, np.full(len(x), y), np.full(len(x), z)], 1)


def trace(x0, x1, **k):
    p = line(x0, x1, **k)
    return np.c_[p, np.zeros((len(p), 6))]              # as a rollout's trace: position first


def test_numbers_are_the_catalogues():
    R = extras.Rules.load()
    ex = things.load()["P_Extra"].numbers
    assert R.point_radius == ex["point_pickup_radius"]["v"] == 3 and R.life_radius == ex["life_pickup_radius"]["v"] == 4.5
    assert (R.core, R.follower, R.followers) == (100, 20, 6)
    assert (R.off_after, R.on_after, R.armed_after) == (2, 8, 12)
    assert R.basis["point_radius"] == "MEASURED" and R.basis["checkpoint_radius"] == "READ"


def test_strictly_inside_the_distance():
    for z, got in ((2.99, True), (3.0, False), (3.01, False)):
        w = extras.Walk(INFO)
        w.feed(line(10.0, 30.0, z=z, step=0.01))
        assert ("P1" in w.taken) is got, z
    w = extras.Walk(INFO)
    w.feed(line(40.0, 60.0, z=4.49))
    assert w.taken["L1"]["kind"] == "life"
    w = extras.Walk(INFO)
    w.feed(line(40.0, 60.0, z=4.51))
    assert "L1" not in w.taken


def test_collected_at_the_first_frame_inside_and_only_once():
    w = extras.Walk(INFO)
    p = line(10.0, 30.0)
    w.feed(p)
    d = np.linalg.norm(p - np.array([20.0, 2.0, 0.0]), axis=1)
    assert w.taken["P1"]["frame"] == int(np.flatnonzero(d < 3.0)[0])
    w.feed(p[::-1])                                     # back through it: it is gone
    assert sum(e["kind"] == "point" for e in w.events) == 1


def test_only_the_current_sectors_balls_are_there():
    w = extras.Walk(INFO)
    w.feed(line(10.0, 70.0))                            # sector 1 is current: P2 (sector 2) is passed through
    assert set(w.taken) == {"P1", "L1"} and "P2" in w.inactive
    w = extras.Walk(INFO, sector=2)
    w.feed(line(10.0, 70.0))
    assert set(w.taken) == {"P2"}


def test_the_checkpoint_changes_the_sector():
    R = extras.Rules.load()
    w = extras.Walk(INFO)
    p = line(80.0, 99.0)
    w.feed(p)
    flame = np.array([100.0, R.flame_height, 0.0])
    first = int(np.flatnonzero(np.linalg.norm(p - flame, axis=1) < R.checkpoint_radius)[0])
    cp = [e for e in w.events if e["kind"] == "checkpoint"]
    assert len(cp) == 1 and cp[0]["frame"] == first and w.sector_off[1] == first + R.off_after
    assert w.sector == 2                                # the frames after it have passed
    w.feed(line(99.0, 15.0, step=-0.2))                 # back: P2 is there now; L1 and P1 are not
    assert set(w.taken) == {"P2"}


def test_a_score_ball_counts_with_its_small_balls():
    R = extras.Rules.load()
    flame_x = 100.0 - float(np.sqrt(R.checkpoint_radius ** 2 - (2.0 - R.flame_height) ** 2))     # where the path gets inside the checkpoint's distance
    for gap in (0.5, 4.5):                              # however soon the checkpoint comes after it: Esc collects them
        info = copy.deepcopy(INFO)
        info["placements"] = [dict(name="P1", type="P_Extra_Point", sector=1, position=[flame_x - 0.2 * 132 * gap + 3.0, 2.0, 0.0])]
        res = extras.Walk(info).result(dict(segments=[dict(type="follow", trace=trace(-60.0, 99.0))]))
        assert res["extras"][0]["collected"] and res["extras"][0]["points"] == 220 and res["points"] == 220
    res = extras.Walk(INFO, sector=3).result(dict(segments=[dict(type="follow", trace=trace(220.0, 240.0))]))
    assert res["points"] == R.core + R.follower * R.followers == 220


def test_result_counts_and_marks():
    w = extras.Walk(INFO)
    res = w.result(dict(segments=[dict(type="follow", trace=trace(10.0, 40.0, z=2.95)),             # grazes P1
                                  dict(type="assumed", no="B1", to=np.zeros(3)),
                                  dict(type="follow", trace=trace(45.0, 55.0))]))
    rows = {r["name"]: r for r in res["extras"]}
    assert rows["P1"]["collected"] and rows["P1"]["marginal"] and not rows["P1"]["after_left_break"]
    assert rows["L1"]["collected"] and rows["L1"]["after_left_break"]
    assert res["lives"] == 0 and res["points"] == 220 and res["game_sector"] == 1
    w = extras.Walk(INFO)
    res = w.result(dict(segments=[dict(type="follow", trace=trace(10.0, 40.0, z=3.05))]))            # just outside
    assert not res["extras"][0]["collected"] and res["extras"][0]["marginal"]
    assert extras.lines(res)


def test_annotate_walks_on_from_the_sector_before():
    one = dict(sector=1, rolling_start=False, segments=[dict(type="follow", trace=trace(0.0, 92.0))])     # ends short of the flame
    two = dict(sector=2, rolling_start=True, segments=[dict(type="follow", trace=trace(92.0, 99.0))],
               alternatives=[dict(found=True, route=dict(segments=[dict(type="follow", trace=trace(92.0, 99.0))]))])
    extras.annotate([two, one], INFO)
    assert one["extras"]["checkpoints"] == [] and one["extras"]["game_sector_at_end"] == 1
    assert [c["number"] for c in two["extras"]["checkpoints"]] == [1] and two["extras"]["game_sector_at_end"] == 2
    assert two["alternatives"][0]["route"]["extras"]["checkpoints"][0]["number"] == 1
