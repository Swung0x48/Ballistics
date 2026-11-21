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
    v = sub.add_parser("solve", help="coarse plan, plain roll along it, fine search only at the breaks; says what each break needed")
    v.add_argument("--mesh", type=Path, required=True, help="<name>-mesh.npz (exported with --templates, so that placed objects are in it)")
    v.add_argument("--map", type=Path, required=True, help="<name>-map.json")
    v.add_argument("--sector", default="all", help="sector number, or all")
    v.add_argument("--tries", type=int, default=300_000, help="how many times the fine search may try a key hold at one break in one mode")
    v.add_argument("--shortcuts", type=int, default=4, help="how many islands (places no edge of the coarse level leads onto) to try per sector for another route; 0: only the standard route")
    v.add_argument("--whole", choices=("strict", "free"), help="instead: one search over the whole sector in this mode (the first version)")
    v.add_argument("--seconds", type=float, default=600.0, help="with --whole: time limit per sector")
    v.add_argument("--out", type=Path, default=Path("data"), help="folder for the results and the coarse graph")
    v.add_argument("--png", type=Path, help="folder for a top view of each sector")
    k = sub.add_parser("coarse", help="the coarse level's plan alone: per sector the cheapest way over the coarse graph and the links it needs; nothing is rolled or searched")
    k.add_argument("--mesh", type=Path, required=True, help="<name>-mesh.npz (exported with --templates, so that placed objects are in it)")
    k.add_argument("--map", type=Path, required=True, help="<name>-map.json")
    k.add_argument("--out", type=Path, default=Path("data"), help="folder for the coarse graph and <name>-coarse-plan.json")
    k.add_argument("--png", type=Path, help="folder for a top view of the whole map and of each sector")
    r = sub.add_parser("route", help="write the solved level out as a reference line file (L<nn>.npz) for other tools to read")
    r.add_argument("--map", type=Path, required=True, help="<name>-map.json")
    r.add_argument("--results", type=Path, default=Path("data"), help="folder with the solver's results (<name>-staged-s<N>.pkl)")
    r.add_argument("--level", type=int, required=True, help="the level number the file is named after")
    r.add_argument("--out", type=Path, required=True, help="folder to write L<nn>.npz into")
    t = sub.add_parser("things", help="the catalogue of balls, loose objects, mechanisms, triggers, pickups and level rules: list it, print one entry, or check it")
    t.add_argument("name", nargs="?", help="an entry id (P_Modul_25, ball_paper), a placed type name (P_Trafo_Stone) or a rule id (death_timing); without it: the list")
    t.add_argument("--check", action="store_true", help="check every file of the catalogue and say what is wrong")
    a = ap.parse_args(argv)
    if a.command == "things":
        import sys

        from . import things

        out = sys.stdout
        if hasattr(out, "reconfigure"):
            out.reconfigure(encoding="utf-8")                # the entries are Chinese text; a console code page would garble them
        if a.check:
            bad = things.check()
            for f, problems in bad.items():
                for p in problems:
                    print("%s: %s" % (f, p))
            print("%d files with problems" % len(bad) if bad else "the catalogue is well formed")
            return 1 if bad else 0
        cat = things.load()
        if not a.name:
            print("| id | 名称 | 类别 | 摆放类型 | 状态 | 木 / 石 / 纸 | 读 / 测 / 推 / 假定 / 未知 |")
            print("|---|---|---|---|---|---|---|")
            for th in cat.values():
                c = th.count()
                print("| %s | %s | %s | %s | %d | %s | %d / %d / %d / %d / %d |" % (
                    th.id, th.name_zh, th.category, "、".join(th.types) or "—", len(th.states),
                    " / ".join(th.by_ball[k]["verdict"] for k in things.BALLS), c["READ"] + c["WIKI"], c["MEASURED"], c["DERIVED"], c["ASSUMED"], c["UNKNOWN"]))
            print()
            print("共用规则：" + "、".join("%s（%s）" % (r["id"], r["name_zh"]) for r in cat.rules.values()))
            return
        th = cat.get(a.name) or cat.of_type(a.name)
        if th is not None:
            print(things.render(th, cat.rules))
        elif a.name in cat.rules:
            print(things.render_rule(cat.rules[a.name]))
        else:
            print("no entry, placed type or rule named %r; entries: %s; rules: %s" % (a.name, ", ".join(cat), ", ".join(cat.rules)))
            return 1
        return
    if a.command == "coarse":
        import json

        from . import solve
        from .geometry import Mesh

        mesh = Mesh.load(a.mesh)
        info = json.load(open(a.map, encoding="utf8"))
        name = a.mesh.name.replace("-mesh.npz", "")
        g = solve.graph_for(mesh, info, a.out / (name + "-coarse-solid.pkl"))
        plans = solve.coarse_plans(mesh, info, g)
        for p in plans:
            if not p["ok"]:
                print("sector %d: no plan: %s" % (p["sector"], p["reason"]))
                continue
            c = p["counts"]
            print("sector %d: %d places, %.0f long; %d launch edges, %d steps, %d links%s" % (
                p["sector"], len(p["nodes"]), p["length"], c.get("launch", 0), c.get("step", 0), len(p["links"]),
                "; it ends %.0f short of the goal marker (nothing nearer to be on)" % p["short_of_goal"] if "short_of_goal" in p else ""))
            if "starts_off" in p:
                print("   NO PLAN for the first %.0f from the reset point: nothing nearer reaches the goal, not even with links; the plan begins at %s"
                      % (p["starts_off"], [round(float(x), 1) for x in g.pos[p["nodes"][0]]]))
            for n, e in enumerate(p["links"], 1):
                print("   link %d at %s: %.1f across, rise %+.1f%s" % (n, [round(float(x), 1) for x in e["near"]], e["across"], e["rise"],
                                                                 "; placed within 9: " + ", ".join(e["placed_near"]) if e["placed_near"] else ""))

        def plain(v):
            return v.tolist() if hasattr(v, "tolist") else v

        json.dump(dict(map=name, what="the coarse level's plan alone: positions only, no speeds; nothing was rolled or searched",
                       sectors=[dict(sector=p["sector"], ok=p["ok"], reason=p.get("reason"), short_of_goal=p.get("short_of_goal"),
                                     starts_off_reset_point=p.get("starts_off"), length=p["length"], edges=p["counts"],
                                     points=[plain(g.pos[n]) for n in p["nodes"]], kinds=p["kinds"],
                                     launches=[{k2: plain(v) for k2, v in e.items()} for e in p["launches"]],
                                     steps=[{k2: plain(v) for k2, v in e.items()} for e in p["steps"]],
                                     links=[{k2: plain(v) for k2, v in e.items()} for e in p["links"]]) for p in plans]),
                  open(a.out / (name + "-coarse-plan.json"), "w", encoding="utf8"), ensure_ascii=False)
        if a.png:
            import numpy as np

            from .render import draw_coarse
            n = len(info["checkpoints"])
            names, anchors = ["出生点"] + ["盘点 %d" % (i + 1) for i in range(n)] + ["飞船"], [info["start"]] + info["checkpoints"] + [info.get("airship")]
            good = [p for p in plans if p["ok"]]
            draw_coarse(mesh, g, good, a.png / ("%s-coarse-level.png" % name), "%s：只有粗层的规划线（没有跟随器，没有细搜）" % name, anchors, names, labels=False, width=2400)
            for p in good:
                A = g.pos[p["nodes"]] if "reset" not in p else np.vstack([g.pos[p["nodes"]], p["reset"]])
                box = (A[:, 0].min() - 10, A[:, 0].max() + 10, A[:, 2].min() - 10, A[:, 2].max() + 10)
                c = p["counts"]
                draw_coarse(mesh, g, [p], a.png / ("%s-coarse-s%d.png" % (name, p["sector"])),
                            "第 %d 小节的粗层规划线：%d 个落脚点，长 %.0f；起飞边 %d 条，台阶 %d 个，链接 %d 处%s" % (
                                p["sector"], len(p["nodes"]), p["length"], c.get("launch", 0), c.get("step", 0), len(p["links"]),
                                "；终点离飞船标记还有 %.0f" % p["short_of_goal"] if "short_of_goal" in p else "")
                            + ("；开头 %.0f 没有规划" % p["starts_off"] if "starts_off" in p else ""), anchors, names, box=box)
        return
    if a.command == "route":
        import json
        import pickle

        from . import route

        info = json.load(open(a.map, encoding="utf8"))
        name = a.map.name.replace("-map.json", "")
        results = []
        for sec in range(1, len(info["reset_points"]) + 1):
            f = a.results / ("%s-staged-s%d.pkl" % (name, sec))
            if not f.exists():
                break
            results.append(pickle.load(open(f, "rb")))
        from . import extras
        extras.annotate(results, info)
        P, sector, t_ms, meta = route.reference_line(results, info, a.level)
        path = route.write_line(a.out, a.level, P, sector, t_ms, meta)
        print("%s: %d points, %.0f long, %.1f game s, sectors %s on the line; %s" % (
            path, len(P), meta["length"], meta["game_ms"] / 1000, meta["sectors_on_line"],
            "complete" if meta["complete"] else "ends at the first break that was left: " + ", ".join("sector %d %s" % (b["sector"], b["no"]) for b in meta["left_breaks"])))
        return
    if a.command == "solve":
        import json
        import pickle

        from . import extras, solve, staged
        from .geometry import Mesh

        mesh = Mesh.load(a.mesh)
        info = json.load(open(a.map, encoding="utf8"))
        name = a.mesh.name.replace("-mesh.npz", "")
        g = solve.graph_for(mesh, info, a.out / (name + "-coarse-solid.pkl"))
        sectors = range(1, len(info["reset_points"]) + 1) if a.sector == "all" else [int(a.sector)]
        carry = None                                          # the state the ball reached the last checkpoint in
        results = {}
        for sec in sectors:
            if a.whole:
                world, c = solve.whole_sector(mesh, info, g, sec, a.whole, 6_000_000, a.seconds)
                res = dict(sector=sec, mode=a.whole, crossing=c, analysis=solve.analyse(world, info, c) if c.found else None)
                pickle.dump(res, open(a.out / ("%s-%s-s%d.pkl" % (name, a.whole, sec)), "wb"))
                continue
            before = a.out / ("%s-staged-s%d.pkl" % (name, sec - 1))
            if carry is None and sec > 1 and before.exists():     # one sector asked for: go on from the result of the one before
                carry = pickle.load(open(before, "rb")).get("end_state")
            res = staged.solve_sector(mesh, info, g, sec, a.tries, start=carry)
            carry = res["end_state"]                              # None when the sector was not got through: the next one starts at rest
            results[sec] = res
            extras.annotate([results[k] for k in sorted(results)], info)     # what the routes collect on the way: changes nothing about them
            staged.summary(res)
            pickle.dump(res, open(a.out / ("%s-staged-s%d.pkl" % (name, sec)), "wb"))
            if a.png:
                from .render import draw_staged
                draw_staged(mesh, res, a.png / ("%s-staged-s%d.png" % (name, sec)))
        done = [a.out / ("%s-staged-s%d.pkl" % (name, k)) for k in range(1, len(info["reset_points"]) + 1)]
        if not a.whole:
            # other routes, after every standard route is done (so that they change nothing about those): each
            # sector's islands are tried from its standard route. The next sector still goes on from the standard one
            from . import shortcut
            for sec in sectors if a.shortcuts > 0 else ():
                res = results[sec]
                res["alternatives"] = shortcut.alternatives(mesh, info, g, sec, res, a.tries, a.shortcuts)
                extras.annotate([results[k] for k in sorted(results)], info)
                shortcut.summary(res)
                pickle.dump(res, open(a.out / ("%s-staged-s%d.pkl" % (name, sec)), "wb"))
                if a.png:
                    from .render import draw_staged
                    for rid, route, _alt in shortcut.routes(res)[1:]:
                        sc = route["shortcut"]
                        state = "通" if route["solved"] else "不通，留有断点 " + "、".join(route["unresolved"])
                        saved = "" if sc["saved"] is None else "，比标准路线%s %.1f" % ("少" if sc["saved"] >= 0 else "多", abs(sc["saved"]) / 132)
                        draw_staged(mesh, route, a.png / ("%s-staged-s%d-%s.png" % (name, sec, rid.lower())),
                                    title="第 %d 小节 路线 %s：经孤岛 %s（%s，轨迹 %.1f 游戏秒%s）" % (sec, rid, sc["no"], state, route["ticks"] / 132, saved))
            have = [pickle.load(open(p, "rb")) for p in done if p.exists()]
            extras.annotate(have, info)
            if have:                                              # every route of every sector solved so far, for other tools
                json.dump(shortcut.listing(name, have), open(a.out / (name + "-routes.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
        if a.png and not a.whole and all(p.exists() for p in done):       # every sector has a result: the whole level on one view
            from .render import draw_level
            n = len(info["checkpoints"])
            draw_level(mesh, [pickle.load(open(p, "rb")) for p in done], a.png / ("%s-staged-level.png" % name),
                       ["出生点"] + ["盘点 %d" % (k + 1) for k in range(n)] + ["飞船"], [info["start"]] + info["checkpoints"] + [info["airship"]])
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
    raise SystemExit(main())
