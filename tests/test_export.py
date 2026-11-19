"""From a map's objects and groups to the collision mesh and the anchors."""
import os
from pathlib import Path

import numpy as np
import pytest

from ballistics.export import FLOOR_GROUP, RAIL_GROUP, STOPPER_GROUP, build_map, build_mesh, export_map
from ballistics.geometry import Mesh
from ballistics.nmo import NmoLevel, Object3D, find_bmap

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1_NMO = DATA / "maps" / "Level_01.NMO"
L1_GAME = DATA / "engine" / "L01-mesh.npz"     # the fixed bodies' triangles as read out of the running game


def at(x, y, z):
    m = np.eye(4)
    m[3, :3] = (x, y, z)
    return m


def square(name, x, groups):
    """A 4 x 4 horizontal square whose object sits at (x, 0, 0)."""
    return Object3D(name, at(x, 0, 0), np.array([(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)], float), np.array([(0, 1, 2), (0, 2, 3)]), list(groups))


def marker(name, pos, groups):
    return Object3D(name, at(*pos), None, None, list(groups))


def level() -> NmoLevel:
    objs = [square("Floor", 10, [FLOOR_GROUP]), square("Rail", 30, [RAIL_GROUP]), square("Stopper", 50, [STOPPER_GROUP]), square("Decoration", 70, []),
            marker("PS_FourFlames_01", (0, 0, 0), ["PS_Levelstart"]), marker("PE_Balloon_01", (99, 1, 2), ["PE_Levelende"]),
            marker("PR_Resetpoint_02", (40, 3, 0), ["PR_Resetpoints"]), marker("PR_Resetpoint_01", (1, 3, 0), ["PR_Resetpoints", "Sector_01"]),
            marker("PC_TwoFlames_01", (40, 0, 0), ["PC_Checkpoints", "Sector_01"]),
            marker("P_Trafo_Stone_01", (20, 0, 2), ["P_Trafo_Stone", "Sector_02"])]
    groups: dict = {}
    for o in objs:
        for g in o.groups:
            groups.setdefault(g, []).append(o.name)
    return NmoLevel("Test.nmo", {o.name: o for o in objs}, groups)


def test_only_floors_and_rails_collide_and_the_cube_is_added():
    m = build_mesh(level())
    assert [e["name"] for e in m.entities] == ["Floor", "Rail", "FixCube"]
    assert len(m) == 2 + 2 + 12 and int(m.rail.sum()) == 2 and m.meta["stoppers_left_out"] == 1
    assert m.A[m.owner == 0][:, 0].min() == 10.0                     # moved by the object's world matrix
    assert abs(m.rest(12.0, 2.0, 20.0) - 2.0) < 1e-6                  # a ball rests on the floor
    assert abs(m.rest(0.0, 0.0, 20.0) - 4.0) < 1e-6                   # and on top of the game's cube at the origin
    assert np.isnan(m.rest(52.0, 2.0, 20.0)) and np.isnan(m.rest(72.0, 2.0, 20.0))    # nothing where the stopper and the decoration are
    assert len(build_mesh(level(), fixture=False)) == 4


def test_anchors_and_placements_come_from_the_groups():
    info = build_map(level())
    assert info["start"] == [1.0, 3.0, 0.0]                           # the first reset point, by number, not file order
    assert info["reset_points"] == [[1.0, 3.0, 0.0], [40.0, 3.0, 0.0]]
    assert info["checkpoints"] == [[40.0, 0.0, 0.0]] and info["airship"] == [99.0, 1.0, 2.0] and info["sectors"] == 2
    assert [(p["name"], p["type"], p["sector"]) for p in info["placements"]] == [("P_Trafo_Stone_01", "P_Trafo_Stone", 2)]
    assert info["start_ball"] is None


def _bmap_found() -> bool:
    try:
        find_bmap()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not (L1_NMO.is_file() and _bmap_found()), reason="Level_01.NMO not in the data folder, or the BMap library not found")
def test_level_1_from_its_map_file(tmp_path):
    r = export_map(L1_NMO, tmp_path)
    assert (r["triangles"], r["rail_triangles"], r["entities"]) == (21775, 14358, 37)
    assert (r["sectors"], r["reset_points"], r["checkpoints"], r["airship"]) == (4, 4, 3, True)
    m = Mesh.load(tmp_path / "Level_01-mesh.npz")
    if not L1_GAME.is_file():
        pytest.skip("no triangles read out of the running game to compare with")
    g = Mesh.load(L1_GAME)

    def key(mesh):
        keep = ~np.array([e["name"] == "FixCube" for e in mesh.entities])[mesh.owner]
        T = np.stack([mesh.A[keep], mesh.B[keep], mesh.C[keep]], axis=1)
        o = np.lexsort((T[:, :, 2], T[:, :, 1], T[:, :, 0]), axis=1)
        T = np.take_along_axis(T, o[:, :, None], axis=1).reshape(len(T), 9)
        return T[np.lexsort(np.round(T, 3).T[::-1])]

    a, b = key(m), key(g)
    assert a.shape == b.shape and np.abs(a - b).max() < 1e-6          # the same triangles the game collides with
