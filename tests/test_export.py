"""The exporter's own part: turning a loaded level's data into a mesh. Talking to the engine is not tested here."""
from types import SimpleNamespace

import numpy as np

from ballistics.export import RAIL_GROUP, build_mesh, to_world

IDENTITY = np.vstack([np.eye(3), np.zeros(3)])


class FakeLevel:
    """Two entities: a floor square moved 10 along x by its world matrix, and a one-triangle rail."""

    def __init__(self):
        shift = np.vstack([np.eye(3), [10.0, 0.0, 0.0]])
        self.meshes = {
            7: [SimpleNamespace(vertices=[(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)], faces=[0, 1, 2, 0, 2, 3], matrix=shift)],
            9: [SimpleNamespace(vertices=[(0, 1, 0), (1, 1, 0), (0, 1, 1)], faces=[0, 1, 2], matrix=IDENTITY)],
        }
        self.entities = {7: SimpleNamespace(name="Floor"), 9: SimpleNamespace(name="Rail")}
        self.phys_by_ent = {7: SimpleNamespace(friction=0.7, elasticity=0.3, surface="stone"), 9: SimpleNamespace(friction=0.3, elasticity=0.1, surface="metal")}
        self.ent_groups = {7: {"Phys_Floors"}, 9: {RAIL_GROUP}}

    def static_entities(self):
        return [7, 9]

    def meshes_of(self, ck):
        return self.meshes[ck]

    def in_group(self, ck, group):
        return group in self.ent_groups[ck]

    def static_key(self):
        return "abc"


def test_vertices_are_moved_by_the_world_matrix():
    m = np.vstack([np.eye(3), [1.0, 2.0, 3.0]])
    assert np.allclose(to_world([(1, 1, 1)], m), [(2, 3, 4)])


def test_the_mesh_keeps_every_triangle_with_its_entity():
    mesh = build_mesh(FakeLevel())
    assert len(mesh) == 3 and mesh.rail.tolist() == [False, False, True]
    assert mesh.A[:, 0].min() == 0.0 and mesh.A[:2, 0].min() == 10.0          # the floor sits at x = 10..14
    assert [e["name"] for e in mesh.entities] == ["Floor", "Rail"]
    assert mesh.entities[0]["friction"] == 0.7 and mesh.entities[1]["groups"] == [RAIL_GROUP]
    assert mesh.meta["rail_triangles"] == 1 and mesh.meta["static_key"] == "abc" and len(mesh.tubes) == 0
    # the result is a working mesh: a ball lowered over the floor rests at its radius
    assert abs(mesh.rest(12.0, 2.0, 20.0) - 2.0) < 1e-6


def test_rail_triangles_go_through_the_tube_fitter():
    seen = {}

    def fit(A, B, C, owner):
        seen["n"] = len(A)
        return np.array([[0, 1, 0, 1, 1, 0, 0.35]]), np.zeros(len(A), bool), {}

    mesh = build_mesh(FakeLevel(), fit)
    assert seen["n"] == 1 and mesh.tubes.shape == (1, 7) and mesh.meta["tubes"] == 1
