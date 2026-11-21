"""The dome (P_Dome): a fixed body of its sector, with its own friction and elasticity. Made-up meshes only."""
import numpy as np

from ballistics import physics as ph, things
from ballistics.coarse import in_sector
from ballistics.geometry import Mesh
from ballistics.kernel import OK, PLACED, World
from ballistics.solve import world_for


def quad(x0, x1, z0, z1, y):
    p = [np.array(v, float) for v in ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))]
    return [(p[0], p[2], p[1]), (p[0], p[3], p[2])]           # wound to face up


def level():
    """A floor at 0, and three slabs at height 3 standing for placed parts: a dome of sector 1, a dome of sector 2, a push board."""
    parts = [(quad(-40, 40, -40, 40, 0.0), dict(name="floor", groups=["Phys_Floors"], friction=0.7, elasticity=0.3)),
             (quad(-6, 6, -6, 6, 3.0), dict(name="P_Dome_MF", groups=[], placed=True, type="P_Dome", placement="P_Dome_01", sector=1)),
             (quad(14, 26, -6, 6, 3.0), dict(name="P_Dome_MF", groups=[], placed=True, type="P_Dome", placement="P_Dome_02", sector=2)),
             (quad(-26, -14, -6, 6, 3.0), dict(name="P_Modul_25_Bridge", groups=[], placed=True, type="P_Modul_25", placement="P_Modul_25_01", sector=1))]
    A, B, C, owner, ents = [], [], [], [], []
    for tris, e in parts:
        ents.append(e)
        for t in tris:
            A.append(t[0]); B.append(t[1]); C.append(t[2]); owner.append(len(ents) - 1)
    return Mesh(np.array(A), np.array(B), np.array(C), np.array(owner), np.zeros(len(A), bool), np.zeros((0, 7)), ents, {})


def test_the_sectors_dome_is_fixed_geometry_and_the_others_are_not_there():
    m = level()
    body = things.load()["P_Dome"].bodies[0]
    one = in_sector(m, 1)
    kinds = {e.get("placement"): (bool(e.get("placed")), e.get("friction"), e.get("elasticity")) for e in one.entities if e.get("placement")}
    assert kinds["P_Dome_01"] == (False, body["friction"]["v"], body["elasticity"]["v"]) == (False, 0.2, 0.8)
    assert kinds["P_Modul_25_01"][0] is True                             # anything else placed stays as it is
    assert len(one.A) == len(m.A) - 2 and not (one.owner == 2).any()      # sector 2's dome is taken out
    assert int(one.placed.sum()) == 2
    every = in_sector(m)
    assert len(every.A) == len(m.A) and int(every.placed.sum()) == 2      # no sector given: every dome is there
    assert m.entities[1].get("placed") is True                           # the mesh it was made from is untouched
    plain = Mesh(m.A[:2], m.B[:2], m.C[:2], m.owner[:2], m.rail[:2], m.tubes, m.entities[:1], {})
    assert in_sector(plain, 1) is plain


def test_a_ball_bounces_on_the_dome_by_its_own_elasticity():
    m, P = level(), ph.Params()
    info = dict(placements=[], death_boxes=[])
    was = World.of(m)                                                    # as a placed thing: touching it ends the rollout
    assert was.rollout(ph.WOOD, P, [0.0, 9.0, 0.0], [0, 0, 0], [0, 0, 0], np.zeros((1, 3)), [200])[0] == PLACED
    w = world_for(m, info, "strict", 1)
    for ball, want in ((ph.WOOD, np.sqrt(0.2 * 0.8)), (ph.STONE, np.sqrt(0.1 * 0.8))):
        st, done, pos, vel, om, _s, tr = w.rollout(ball, P, [0.0, 9.0, 0.0], [0, 0, 0], [0, 0, 0], np.zeros((1, 3)), [60], trace=True)
        assert st == OK
        y = tr[:, 1]
        hit = int(np.argmin(y))
        down = (y[hit - 2] - y[hit - 1]) * ph.PSI_HZ                      # speed just before and just after the bounce
        up = (y[hit + 2] - y[hit + 1]) * ph.PSI_HZ
        assert abs(up / down - want) < 0.03, (ball.name, up / down, want)
    st, done, pos, *_ = w.rollout(ph.WOOD, P, [20.0, 9.0, 0.0], [0, 0, 0], [0, 0, 0], np.zeros((1, 3)), [60])
    assert st == OK and pos[1] < 4.5                                     # where sector 2's dome is, there is nothing: it comes down on the floor
    st, *_ = w.rollout(ph.WOOD, P, [-20.0, 9.0, 0.0], [0, 0, 0], [0, 0, 0], np.zeros((1, 3)), [200])
    assert st == PLACED                                                  # the push board is still a placed thing


def test_it_is_slippery():
    m, P = level(), ph.Params()
    w = world_for(m, dict(placements=[], death_boxes=[]), "strict", 1)
    on_dome = w.rollout(ph.WOOD, P, [-4.0, 5.011, 0.0], [0, 0, 0], [0, 0, 0], np.array([[1.0, 0.0, 0.0]]), [20])
    on_floor = w.rollout(ph.WOOD, P, [-4.0, 2.011, 20.0], [0, 0, 0], [0, 0, 0], np.array([[1.0, 0.0, 0.0]]), [20])
    assert on_dome[0] == OK and on_floor[0] == OK
    # the same push: on the dome (friction 0.8 x 0.2) the ball slips and spins up less than on the floor (0.8 x 0.7)
    assert abs(on_dome[4][2]) < abs(on_floor[4][2]) and on_dome[3][0] > on_floor[3][0]
