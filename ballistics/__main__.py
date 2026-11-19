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
    e.add_argument("--templates", type=Path, help="the game's folder of placed-object templates (3D Entities/PH): adds the mechanisms, boxes and loose balls at their rest poses")
    c = sub.add_parser("calibrate", help="measure how far the hand-written physics is from a recorded engine run")
    c.add_argument("--mesh", type=Path, required=True, help="<name>-mesh.npz of the map the run was played on")
    c.add_argument("--run", type=Path, required=True, help="a recorded run file (.npz)")
    c.add_argument("--every", type=int, default=30, help="start an open-loop roll every this many ticks")
    v = sub.add_parser("solve", help="search key tapes in the physics kernel, sector by sector, and say what each trajectory leans on")
    v.add_argument("--mesh", type=Path, required=True, help="<name>-mesh.npz (exported with --templates, so that placed objects are in it)")
    v.add_argument("--map", type=Path, required=True, help="<name>-map.json")
    v.add_argument("--sector", default="all", help="sector number, or all")
    v.add_argument("--mode", default="strict", choices=("strict", "solid", "nohang", "free"), help="what a rollout may not do (see ballistics/solve.py)")
    v.add_argument("--expansions", type=int, default=3_000_000)
    v.add_argument("--seconds", type=float, default=600.0, help="time limit per sector")
    v.add_argument("--out", type=Path, default=Path("data"), help="folder for the results (<name>-<mode>-s<N>.pkl) and the coarse graph")
    a = ap.parse_args(argv)
    if a.command == "solve":
        import json
        import pickle

        from . import solve
        from .geometry import Mesh

        mesh = Mesh.load(a.mesh)
        info = json.load(open(a.map, encoding="utf8"))
        name = a.mesh.name.replace("-mesh.npz", "")
        g = solve.graph_for(mesh, info, a.out / (name + "-coarse.pkl"))
        sectors = range(1, len(info["reset_points"]) + 1) if a.sector == "all" else [int(a.sector)]
        for sec in sectors:
            world, c = solve.solve_sector(mesh, info, g, sec, a.mode, a.expansions, a.seconds)
            res = dict(sector=sec, mode=a.mode, crossing=c, analysis=solve.analyse(world, info, c) if c.found else None)
            pickle.dump(res, open(a.out / ("%s-%s-s%d.pkl" % (name, a.mode, sec)), "wb"))
        return
    if a.command == "calibrate":
        from .calibrate import report
        from .geometry import Mesh
        from .trace import Run

        mesh = Mesh.load(a.mesh)
        report(mesh.subset(~mesh.placed), Run.load(a.run), every=a.every)      # a recorded run moves the placed objects; compare on the fixed geometry
        return
    if a.command == "export":
        from .export import export_map

        r = export_map(a.map, a.out, a.name, a.bmap, a.templates)
        if a.templates:
            print("%s: %d placed parts at rest (%d triangles)" % (r["name"], r["placed_parts"], r["placed_triangles"]))
        print("%s: %d objects, %d collision triangles (%d on rails, %d rail tube segments) in %d bodies; %d sectors, %d reset points, %d checkpoints, airship %s, %d placements -> %s" % (
            r["name"], r["objects"], r["triangles"], r["rail_triangles"], r["tubes"], r["entities"], r["sectors"], r["reset_points"], r["checkpoints"],
            "found" if r["airship"] else "not found", r["placements"], a.out))


if __name__ == "__main__":
    main()
