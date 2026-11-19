"""Reading a Ballance map file (.nmo) directly: objects, their meshes and world matrices, and the groups.

A Ballance map is a Virtools composition. It holds only geometry and group membership; what an object is to the
game follows from the groups it is in (Phys_Floors, Phys_FloorRails, Sector_01, PC_Checkpoints, ...). This module
reads those and nothing else: no game, no engine.

The file format is parsed by BMap, the native library of the libcmo21 project, through its Python binding
`pybmap`. Neither is part of this repository. `find_bmap` looks for a folder that contains the `pybmap` package:
the one passed in, then the folder named by the BALLISTICS_BMAP environment variable, then a Ballance Blender
Plugin (bbp_ng) installed as a Blender extension, which ships it."""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ENCODINGS = ("cp1252", "gbk")  # names in maps are Western or Chinese


@dataclass
class Object3D:
    name: str
    matrix: np.ndarray                      # (4, 4) world matrix, row vectors: translation in the last row
    vertices: np.ndarray | None = None      # (n, 3) in the object's own frame; None without a mesh
    faces: np.ndarray | None = None         # (m, 3) vertex indices
    groups: list = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.matrix, np.float64)[3, :3]

    def world_vertices(self) -> np.ndarray:
        m = np.asarray(self.matrix, np.float64)
        return np.asarray(self.vertices, np.float64) @ m[:3, :3] + m[3, :3]


@dataclass
class NmoLevel:
    path: str
    objects: dict                           # name -> Object3D
    groups: dict                            # group name -> [object names], in file order

    def members(self, group: str) -> list:
        return [self.objects[n] for n in self.groups.get(group, []) if n in self.objects]


def find_bmap(explicit=None) -> Path:
    """The folder that contains the `pybmap` package."""
    tried = []
    for c in (explicit, os.environ.get("BALLISTICS_BMAP")):
        if c:
            tried.append(str(c))
            if (Path(c) / "pybmap").is_dir():
                return Path(c)
    appdata = os.environ.get("APPDATA")
    if appdata:
        root = Path(appdata) / "Blender Foundation" / "Blender"
        tried.append(str(root / "<version>" / "extensions" / "<repo>" / "bbp_ng"))
        for c in sorted(root.glob("*/extensions/*/bbp_ng"), reverse=True):
            if (c / "pybmap").is_dir():
                return c
    raise FileNotFoundError("the BMap library (pybmap) was not found; pass its folder, or set BALLISTICS_BMAP. Looked in: " + "; ".join(tried))


def read_nmo(path, bmap=None, encodings=ENCODINGS) -> NmoLevel:
    """Every 3D object of the map with its mesh and world matrix, and every group with its members."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    folder = str(find_bmap(bmap))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import pybmap.bmap_wrapper as bw

    if not bw.is_bmap_available():
        raise RuntimeError("the BMap native library in %s could not be loaded" % folder)
    objects, groups = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        with bw.BMFileReader(str(path), tmp, tmp, tuple(encodings)) as r:
            for o in r.get_3dobjects():
                name = o.get_name()
                matrix = np.array(o.get_world_matrix().to_const(), np.float64).reshape(4, 4)
                mesh = o.get_current_mesh()
                V = F = None
                if mesh is not None and mesh.get_vertex_count() and mesh.get_face_count():
                    V = np.array([v.to_const() for v in mesh.get_vertex_positions()], np.float64).reshape(-1, 3)
                    F = np.array([f.to_const() for f in mesh.get_face_indices()], np.int64).reshape(-1, 3)
                objects[name] = Object3D(name, matrix, V, F)
            for g in r.get_groups():
                names = [o.get_name() for o in g.get_objects()]
                groups[g.get_name()] = names
                for n in names:
                    if n in objects:
                        objects[n].groups.append(g.get_name())
    return NmoLevel(str(path), objects, groups)
