"""Command line: python -m ballistics <command>."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ballistics")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="read a map file (.nmo) and write its collision mesh and anchors")
    e.add_argument("map", type=Path, help="the map file, e.g. Level_01.NMO or a custom map")
    e.add_argument("--out", type=Path, default=Path("data"), help="folder for <name>-mesh.npz and <name>-map.json")
    e.add_argument("--name", help="name for the output files (default: the map file's name without extension)")
    e.add_argument("--bmap", type=Path, help="folder that contains the pybmap package (default: BALLISTICS_BMAP, then an installed bbp_ng)")
    a = ap.parse_args(argv)
    if a.command == "export":
        from .export import export_map

        r = export_map(a.map, a.out, a.name, a.bmap)
        print("%s: %d objects, %d collision triangles (%d on rails) in %d bodies; %d sectors, %d reset points, %d checkpoints, airship %s, %d placements -> %s" % (
            r["name"], r["objects"], r["triangles"], r["rail_triangles"], r["entities"], r["sectors"], r["reset_points"], r["checkpoints"],
            "found" if r["airship"] else "not found", r["placements"], a.out))


if __name__ == "__main__":
    main()
