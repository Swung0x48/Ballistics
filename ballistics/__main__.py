"""Command line: python -m ballistics <command>."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m ballistics")
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export", help="load a level once in the headless engine and write its collision mesh and anchors")
    e.add_argument("--level", type=int, required=True)
    e.add_argument("--out", type=Path, default=Path("data"), help="folder for L<nn>-mesh.npz and L<nn>-map.json")
    e.add_argument("--engine-root", type=Path, required=True, help="game folder the headless engine runs in")
    e.add_argument("--engine-build", type=Path, required=True, help="the headless engine executable")
    e.add_argument("--instance", required=True, help="label of an existing headless instance that is free to use")
    e.add_argument("--ballrain", type=Path, help="a Ballrain checkout (its engine client is used to talk to the engine)")
    a = ap.parse_args(argv)
    if a.command == "export":
        from . import export

        if a.ballrain:
            export.add_ballrain(a.ballrain)
        r = export.export_level(a.level, a.out, a.engine_root, a.engine_build, a.instance)
        print("L%02d: %d collision triangles (%d on rails, %d rail tube segments), %d entities; %d checkpoints, airship %s, start ball %s -> %s" % (
            r["level"], r["triangles"], r["rail_triangles"], r["tubes"], r["entities"], r["checkpoints"], "found" if r["airship"] else "not found", r["start_ball"], a.out))


if __name__ == "__main__":
    main()
