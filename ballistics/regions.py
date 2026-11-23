"""The coarse level as connectivity between regions: where the ball can get, and over what.

A region is a set of the coarse level's places that reach each other by rolling (roll, rail and rail-end edges, as
the coarse level has them with every mechanism as it is at the start): its strongly connected components. What
leads from one region to another is a link: a launch, a drop or a step the coarse level has, a roll that only goes
one way, a paper ball's lift, or a mechanism's edge (through an open gate, pushing a gate open, pushing a push board
over, over a board that is down, over a seesaw from its low end), each with what it needs (the ball's kind, the
mechanisms' states) and what it changes. The coarse level's links where it does not connect ('gap', 'bridge') are
breaks: ways nothing has shown to be there.

The connectivity is worked out over (region, ball kind, mechanisms' states): from the sector's reset point, with
as few breaks as there can be, then as few links. Within a region the ball can get anywhere by rolling, so it can
also get into any transformer's range that lies in it.

Nothing is rolled here: the places and edges are the coarse level's (`solve.graph_for`, its launches and lifts rolled
in the kernel when the graph was made, and cached), the mechanisms' rules are `coarse.StateGraph`'s.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import coarse

ROLL = ("roll", "rail", "rail end")
BREAK = ("gap", "bridge")
KINDS = ("wood", "stone", "paper")
MECH = {coarse.GATE_THROUGH: "through_gate", coarse.GATE_OPERATE: "push_gate_open", coarse.BOARD_PUSH: "push_board_over",
        coarse.BOARD_OVER: "over_board", coarse.SEESAW: "over_seesaw"}
BIG = 30                  # a region of fewer places is drawn and listed only where a way goes through it


def components(n: int, adj) -> np.ndarray:
    """Strongly connected components of a directed graph (Kosaraju, without recursion): a label per node."""
    order, seen = [], np.zeros(n, bool)
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, 0)]
        while stack:
            u, i = stack[-1]
            if i < len(adj[u]):
                stack[-1] = (u, i + 1)
                v = adj[u][i]
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
                stack.pop()
    radj = [[] for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            radj[v].append(u)
    lab = np.full(n, -1, np.int64)
    c = 0
    for s in reversed(order):
        if lab[s] >= 0:
            continue
        lab[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for v in radj[u]:
                if lab[v] < 0:
                    lab[v] = c
                    stack.append(v)
        c += 1
    return lab


def links_of(sg, lab) -> dict:
    """The links between regions of a sector's state graph, one per (from, to, what, condition): key -> dict(cost (the
    least), count, at (an example: (from place, to place), local)). Keys: ("plain", a, b, kind, mask, ek, tk of the
    place it lands on), ("lift", a, b, lk, tk), ("mech", a, b, type, m, side)."""
    out: dict = {}

    def add(key, c, u, v):
        e = out.get(key)
        if e is None:
            out[key] = dict(cost=float(c), count=1, at=(int(u), int(v)))
        else:
            e["count"] += 1
            if c < e["cost"]:
                e["cost"], e["at"] = float(c), (int(u), int(v))

    for u in range(sg.n):
        for v, c, kind, ek, mask in sg.fwd[u]:
            if lab[u] == lab[v]:
                continue                                      # within the region: nothing it does not have (a transformer's range in it counts by itself)
            add(("plain", int(lab[u]), int(lab[v]), kind, int(mask), int(ek), int(sg.tk[v])), c, u, v)
        for v, c, lk in sg.lfwd[u]:
            add(("lift", int(lab[u]), int(lab[v]), int(lk), int(sg.tk[v])), c, u, v)
        for v, c, _kind, m, typ, side in sg.cfwd[u]:
            add(("mech", int(lab[u]), int(lab[v]), int(typ), int(m), int(side)), c, u, v)
    return out


def _moves(key, k: int, bits: int, M: int):
    """What a link does from a ball of kind k with mechanism bits `bits`: [(kind, bits, what)] (empty: not there)."""
    if key[0] == "plain":
        _p, _a, _b, kind, mask, ek, tk = key
        if (bits & mask) != mask:
            return []
        t = tk if tk >= 0 else ek
        return [(t if t >= 0 else k, bits, kind)]
    if key[0] == "lift":
        _l, _a, _b, lk, tk = key
        if k != coarse.PAPER:
            return []
        t = tk if tk >= 0 else lk
        return [(t if t >= 0 else k, bits, "lift")]
    _m, _a, _b, typ, m, side = key
    bit = (bits >> m) & 1
    if typ == coarse.GATE_THROUGH and bit == 1:
        return [(k, bits, MECH[typ])]
    if typ == coarse.GATE_OPERATE and bit == 0 and k == coarse.STONE:
        return [(k, bits | (1 << m), MECH[typ])]
    if typ == coarse.BOARD_PUSH and k != coarse.PAPER:
        return [(k, bits | (1 << m), MECH[typ])]
    if typ == coarse.BOARD_OVER and bit == 1:
        return [(k, bits, MECH[typ])]
    if typ == coarse.SEESAW and bit == side:
        return [(k, (bits & ~(1 << m)) | ((1 - side) << m) if k == coarse.STONE else bits, MECH[typ])]
    return []


def connect(links: dict, trafo_kinds: dict, M: int, start, goal_region: int):
    """The way through the regions from start = (region, kind, bits) to the goal's region with the fewest breaks,
    then the fewest links: (breaks, [(region, kind, bits, how it got there, the link's key or None)]) or (None, [])
    if there is none even with breaks. Also returns every (region, kind, bits) reached without a break."""
    by_from: dict = {}
    for key in links:
        by_from.setdefault(key[1], []).append(key)
    best = {start: (0, 0)}
    prev = {start: None}
    heap = [(0, 0, start)]
    end = None
    while heap:
        nb, ns, node = heapq.heappop(heap)
        if best.get(node) != (nb, ns):
            continue
        r, k, bits = node
        if r == goal_region:
            end = node
            break
        steps = [((r, t, bits), "transform", None) for t in trafo_kinds.get(r, ()) if t != k]
        for key in by_from.get(r, ()):
            for k2, b2, what in _moves(key, k, bits, M):
                steps.append(((key[2], k2, b2), what, key))
        for nxt, what, key in steps:
            cost = (nb + (1 if what in BREAK else 0), ns + 1)
            if cost < best.get(nxt, (1 << 30, 0)):
                best[nxt] = cost
                prev[nxt] = (node, what, key)
                heapq.heappush(heap, (cost[0], cost[1], nxt))
    free = {n for n, c in best.items() if c[0] == 0}
    if end is None:
        return None, [], free
    path = []
    node = end
    while node is not None:
        p = prev[node]
        path.append((node[0], node[1], node[2], p[1] if p else "start", p[2] if p else None))
        node = p[0] if p else None
    return best[end][0], path[::-1], free


def scan(g, info, sector: int, world, entry_kind: str = "wood", operate_cost: float = 20.0) -> dict:
    """Step 1: a sector's regions (the places that reach each other by rolling), where its reset point and its goal
    are, and which transformers' ranges each region has."""
    from .solve import state_graph_for
    from .staged import nearest_node
    sg, state_of = state_graph_for(g, info, sector, world, operate_cost)
    n, M = sg.n, sg.M
    adj = [[v for v, _c, kind, _e, _m in sg.fwd[u] if kind in ROLL] for u in range(n)]
    lab = components(n, adj)
    P = g.pos[sg.glob]
    sizes = np.bincount(lab)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    goal = g.nearest(anchors[sector], 25.0)
    togo = sg.togo(goal)
    s0 = state_of(entry_kind, world.scene())
    reset = np.array(info["reset_points"][sector - 1], float)
    u0 = nearest_node(g, togo[s0], reset)
    if sg.loc[u0] < 0:
        u0 = int(sg.glob[np.argmin(np.linalg.norm(P - reset, axis=1))])
    trafo_kinds: dict = {}
    for u in np.flatnonzero(sg.tk >= 0):
        trafo_kinds.setdefault(int(lab[u]), set()).add(int(sg.tk[u]))
    regions = {}
    for r in range(len(sizes)):
        idx = np.flatnonzero(lab == r)
        regions[r] = dict(places=int(sizes[r]), centre=P[idx].mean(0).tolist(), lo=P[idx].min(0).tolist(), hi=P[idx].max(0).tolist(),
                          transformers=sorted(KINDS[t] for t in trafo_kinds.get(r, ())))
    return dict(sector=sector, sg=sg, lab=lab, places=P, regions=regions, trafo_kinds=trafo_kinds,
                start=(int(lab[sg.loc[u0]]), s0 >> M, s0 & ((1 << M) - 1)), start_place=int(u0), s0=int(s0), togo=togo,
                goal=int(goal), goal_region=int(lab[sg.loc[goal]]) if sg.loc[goal] >= 0 else -1,
                mechs=[dict(name=m["name"], type=m["type"]) for m in sg.mechs], entry_kind=entry_kind)


def link(res: dict) -> dict:
    """Step 2: the links between a sector's regions (`scan`), the way through them with the fewest breaks, and the
    coarse level's own cheapest plan as regions. Adds them to res and returns it."""
    sg, lab, M = res["sg"], res["lab"], len(res["mechs"])
    g = sg.g
    res["links"] = links_of(sg, lab)
    res["breaks"], res["path"], res["free"] = connect(res["links"], res["trafo_kinds"], M, res["start"], res["goal_region"])
    nodes, kinds, states = sg.plan(res["s0"], res["start_place"], res["togo"])
    plan = []
    for i, gu in enumerate(nodes):
        r = int(lab[sg.loc[gu]])
        st = states[i]
        if plan and plan[-1]["region"] == r and plan[-1]["state"] == st:
            continue
        plan.append(dict(region=r, state=st, kind=KINDS[st >> M], bits=st & ((1 << M) - 1), how=kinds[i - 1] if i else "start", at=g.pos[gu].tolist()))
    res["plan"] = plan
    return res


def analyse(g, info, sector: int, world, entry_kind: str = "wood", operate_cost: float = 20.0) -> dict:
    """Both steps for one sector."""
    return link(scan(g, info, sector, world, entry_kind, operate_cost))


def describe(key) -> str:
    """A link's key in words."""
    if key is None:
        return ""
    if key[0] == "plain":
        _p, _a, _b, kind, mask, ek, tk = key
        s = {"launch": "起飞", "drop": "下落", "step": "台阶", "roll": "单向滚下", "rail": "钢轨", "rail end": "钢轨尽头", "gap": "断点(gap)",
             "bridge": "断点(bridge)"}.get(kind, kind)
        if mask:
            s += "，要机关位 %d" % mask
        if tk >= 0 or ek >= 0:
            s += "，进入%s的变球范围" % {0: "木", 1: "石", 2: "纸"}[tk if tk >= 0 else ek]
        return s
    if key[0] == "lift":
        return "纸球乘风扇上升"
    _m, _a, _b, typ, m, side = key
    return {coarse.GATE_THROUGH: "过开着的栅栏", coarse.GATE_OPERATE: "石球推开栅栏", coarse.BOARD_PUSH: "木球或石球推倒推板",
            coarse.BOARD_OVER: "从倒下的推板上过", coarse.SEESAW: "从低端过跷跷板（石球会把它翻过来）"}[typ] + "（机关 %d）" % m


def _r1(v):
    return [round(float(x), 1) for x in v]


def summary_regions(res: dict) -> dict:
    """Step 1 as plain data: every region of the sector."""
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], mechanisms=res["mechs"], start_region=res["start"][0], goal_region=res["goal_region"],
                regions={str(r): dict(d, centre=_r1(d["centre"]), lo=_r1(d["lo"]), hi=_r1(d["hi"])) for r, d in res["regions"].items()})


def summary_links(res: dict) -> dict:
    """Step 2 as plain data: the links between the sector's regions, and the way with the fewest breaks."""
    P = res["places"]
    links = []
    for key, e in sorted(res["links"].items(), key=lambda kv: (kv[0][1], kv[0][2], str(kv[0]))):
        links.append(dict(**{"from": key[1], "to": key[2]}, what=describe(key), cost=round(e["cost"], 1), count=e["count"],
                          at=[[round(float(x), 2) for x in P[e["at"][0]]], [round(float(x), 2) for x in P[e["at"][1]]]],
                          is_break=bool(key[0] == "plain" and key[3] in BREAK)))
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], start_region=res["start"][0], goal_region=res["goal_region"],
                fewest_breaks=res["breaks"],
                way=[dict(region=r, kind=KINDS[k], mechanism_bits=b, by=what, link=describe(key)) for r, k, b, what, key in res["path"]],
                coarse_plan=[dict(region=p["region"], kind=p["kind"], mechanism_bits=p["bits"], by=p["how"], at=_r1(p["at"])) for p in res["plan"]],
                links=[l for l in links if not l["is_break"]],
                breaks=[l for l in links if l["is_break"] and res["regions"][l["from"]]["places"] >= BIG and res["regions"][l["to"]]["places"] >= BIG],
                breaks_all=sum(1 for l in links if l["is_break"]),
                note="mechanism_bits: bit m is mechanism m's state (gate open, push board down, seesaw low at its other end)" if res["mechs"] else "")


def run(mesh, info, g, name: str, out, png=None, log=print) -> list:
    """The two steps for every sector of a map: step 1 the regions, step 2 the links between them and the way with
    the fewest breaks. Writes <name>-regions.json and <name>-links.json into `out` (and pictures of each step for
    each sector into `png`). A sector is entered as the kind of ball the way through the one before ended as (the
    first as a wood ball: ASSUMED, the map file does not say). Returns the sectors' analyses."""
    import json
    from pathlib import Path
    from .staged import world_for
    kind, done = "wood", []
    for sec in range(1, len(info["reset_points"]) + 1):
        res = link(scan(g, info, sec, world_for(mesh, info, "strict", sec), entry_kind=kind))
        done.append(res)
        if res["path"]:
            kind = KINDS[res["path"][-1][1]]
    log("step 1, regions (the places that reach each other by rolling):")
    for res in done:
        big = sum(1 for d in res["regions"].values() if d["places"] >= BIG)
        log("   sector %d: %d regions, %d of them of %d places or more; the reset point in R%d, the goal in R%d" % (
            res["sector"], len(res["regions"]), big, BIG, res["start"][0], res["goal_region"]))
    log("step 2, the links between them, and the way with the fewest breaks:")
    for res in done:
        n_links = sum(1 for k in res["links"] if not (k[0] == "plain" and k[3] in BREAK))
        log("   sector %d (enters as %s): %d links, %d breaks between regions; %s" % (
            res["sector"], res["entry_kind"], n_links, len(res["links"]) - n_links,
            "no way to the goal even with breaks" if res["breaks"] is None else
            "the goal is reached with %d break%s, over %d links" % (res["breaks"], "" if res["breaks"] == 1 else "s", len(res["path"]) - 1)))
        for r, k, b, what, key in res["path"]:
            log("      R%-4d %-5s bits %d  %s" % (r, KINDS[k], b, describe(key) or what))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = dict(map=name, generator="ballistics", what="", note="the coarse level only: nothing is rolled or searched")
    json.dump(dict(head, what="step 1: the regions, the places that reach each other by rolling", sectors=[summary_regions(r) for r in done]),
              open(out / (name + "-regions.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    json.dump(dict(head, what="step 2: the links between the regions, and the way with the fewest breaks", sectors=[summary_links(r) for r in done]),
              open(out / (name + "-links.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    if png is not None:
        png = Path(png)
        png.mkdir(parents=True, exist_ok=True)
        for res in done:
            s = res["sector"]
            draw_regions(mesh, res, png / ("%s-regions-s%d.png" % (name, s)),
                         "第 %d 小节，第一步：只靠滚动互相到得了的区域（%d 个，%d 个点以上的 %d 个）" % (
                             s, len(res["regions"]), BIG, sum(1 for d in res["regions"].values() if d["places"] >= BIG)))
            draw(mesh, res, png / ("%s-links-s%d.png" % (name, s)),
                 "第 %d 小节，第二步：区域之间的连接；%s" % (s, "到不了目标" if res["breaks"] is None else "到目标最少要跨 %d 个断点" % res["breaks"]))
    return done


PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (57, 59, 121), (99, 121, 57), (140, 109, 49), (132, 60, 57), (123, 65, 115)]
LINK = {"launch": (212, 122, 28), "drop": (16, 132, 120), "step": (82, 81, 78), "roll": (150, 149, 144), "lift": (62, 100, 150), "mech": (196, 60, 48)}


def _arrow(v, a, b, colour, width: int = 5):
    """An arrow from world point a to world point b; a ring where they are the same place (a step done in place)."""
    from .render import SURFACE
    (x0, y0), (x1, y1) = v.P(a[0], a[2]), v.P(b[0], b[2])
    L = float(np.hypot(x1 - x0, y1 - y0))
    if L < 4.0:
        v.pen.ellipse((x1 - 13, y1 - 13, x1 + 13, y1 + 13), outline=SURFACE + (255,), width=width + 4)
        v.pen.ellipse((x1 - 13, y1 - 13, x1 + 13, y1 + 13), outline=colour + (255,), width=width)
        return
    v.halo_line([a, b], colour, width, ring=2)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    h = 18.0
    tip = [(x1, y1), (x1 - h * ux + 0.55 * h * uy, y1 - h * uy - 0.55 * h * ux), (x1 - h * ux - 0.55 * h * uy, y1 - h * uy + 0.55 * h * ux)]
    v.pen.polygon(tip, fill=colour + (255,), outline=SURFACE + (255,))


def _rim(P):
    """Of a region's places, the ones at its edge: a neighbour 1 away (x or z) is not one of its places."""
    keys = {(int(round(p[0])), int(round(p[2]))) for p in P}
    return np.array([p for p in P if any((int(round(p[0])) + dx, int(round(p[2])) + dz) not in keys
                                          for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))]).reshape(-1, 3)


def draw_regions(mesh, res: dict, path, title: str = ""):
    """Step 1: the sector's regions in colours (those of BIG places or more, and the ones the reset point and the goal
    are in), each of 300 places or more named, with the transformers in it."""
    from .render import INK, SURFACE, TopView, _font
    P, lab = res["places"], res["lab"]
    lo, hi = P.min(0), P.max(0)
    v = TopView(mesh, box=(lo[0], hi[0], lo[2], hi[2]), width=2000, title=title, height=True)
    show = {r for r, d in res["regions"].items() if d["places"] >= BIG} | {res["start"][0], res["goal_region"]}
    order = sorted(show, key=lambda r: -res["regions"][r]["places"])
    colour = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(order)}
    for r in order:
        for p in P[lab == r]:
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 2, y - 2, x + 2, y + 2), fill=colour[r] + (220,))
    f = _font(22)
    for r in order:
        d = res["regions"][r]
        if d["places"] >= 300 or r in (res["start"][0], res["goal_region"]):
            x, y = v.P(d["centre"][0], d["centre"][2])
            t = "R%d" % r + ("（%s变）" % "".join({"wood": "木", "stone": "石", "paper": "纸"}[k] for k in d["transformers"]) if d["transformers"] else "")
            v.pen.rectangle((x - 2, y - 2, x + v.pen.textlength(t, font=f) + 4, y + 26), fill=SURFACE + (220,))
            v.pen.text((x, y), t, fill=INK, font=f)
    sg = res["sg"]
    v.dot(sg.g.pos[res["start_place"]], "起点", colour=(208, 59, 59))
    v.dot(sg.g.pos[res["goal"]], "目标", colour=(20, 28, 40))
    v.save(path)


def draw(mesh, res: dict, path, title: str = ""):
    """The sector's regions in colours (the big ones and the ones on the way), the links between them as thin lines,
    and the way with the fewest breaks: the regions it goes through outlined, and each of its links as a numbered
    arrow from where it leaves one region to where it lands in the next. Within a region nothing is drawn: rolling
    gets the ball anywhere in it."""
    from .render import INK, SURFACE, TopView, _font
    P = res["places"]
    lab = res["lab"]
    lo, hi = P.min(0), P.max(0)
    v = TopView(mesh, box=(lo[0], hi[0], lo[2], hi[2]), width=2000, title=title, height=True)
    on_way = [p[0] for p in res["path"]]
    show = {r for r, d in res["regions"].items() if d["places"] >= BIG} | set(on_way)
    order = sorted(show, key=lambda r: -res["regions"][r]["places"])
    colour = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(order)}
    for r in order:
        alpha = 220 if r in on_way else 90
        for p in P[lab == r]:
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 2, y - 2, x + 2, y + 2), fill=colour[r] + (alpha,))
    for r in set(on_way):                                  # the regions of the way, outlined
        for p in _rim(P[lab == r]):
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 3, y - 3, x + 3, y + 3), fill=INK + (255,))
    for key, e in res["links"].items():
        if key[1] not in show or key[2] not in show or (key[0] == "plain" and key[3] in BREAK):
            continue
        a, b = P[e["at"][0]], P[e["at"][1]]
        c = LINK["mech"] if key[0] == "mech" else LINK["lift"] if key[0] == "lift" else LINK.get(key[3], (82, 81, 78))
        v.halo_line([a, b], c, 2, ring=1, alpha=110)
    f = _font(22)
    for r in order:
        d = res["regions"][r]
        if d["places"] >= 300 or r in on_way:
            x, y = v.P(d["centre"][0], d["centre"][2])
            t = "R%d" % r + ("（%s变）" % "".join({"wood": "木", "stone": "石", "paper": "纸"}[k] for k in d["transformers"]) if d["transformers"] else "")
            v.pen.rectangle((x - 2, y - 2, x + v.pen.textlength(t, font=f) + 4, y + 26), fill=SURFACE + (220,))
            v.pen.text((x, y), t, fill=INK, font=f)
    # the way's links, in order: an arrow each with its number in a disc by its head (moved down until it lands on no
    # other); a transformation's number goes in its region. What each number is, is listed under the map
    BALL = {"wood": "木", "stone": "石", "paper": "纸"}
    fb = _font(22)
    n, rows, discs = 0, [], []
    for r, k, b, what, key in res["path"]:
        if what == "start":
            continue
        n += 1
        if key is None:                                    # transformed within the region
            d = res["regions"][r]
            x, y = v.P(d["centre"][0], d["centre"][2])
            rows.append("%d　在 R%d 变%s球" % (n, r, BALL[KINDS[k]]))
        else:
            e = res["links"][key]
            a, bp = P[e["at"][0]], P[e["at"][1]]
            c = (20, 28, 40) if key[0] == "plain" and key[3] in BREAK else LINK["mech"] if key[0] == "mech" else LINK["lift"] if key[0] == "lift" \
                else LINK.get(key[3], (82, 81, 78))
            _arrow(v, a, bp, c, 6)
            x, y = v.P(bp[0], bp[2])
            x, y = x + 18, y - 18
            rows.append("%d　R%d → R%d　%s（%s球）" % (n, key[1], key[2], describe(key), BALL[KINDS[k]]))
        while any(abs(x - qx) < 30 and abs(y - qy) < 30 for qx, qy in discs):
            y += 32
        discs.append((x, y))
        v.pen.ellipse((x - 15, y - 15, x + 15, y + 15), fill=INK + (255,), outline=SURFACE + (255,), width=2)
        t = str(n)
        v.pen.text((x - v.pen.textlength(t, font=fb) / 2, y - 14), t, fill=SURFACE, font=fb)
    sg = res["sg"]
    v.dot(sg.g.pos[res["start_place"]], "起点", colour=(208, 59, 59))
    v.dot(sg.g.pos[res["goal"]], "目标", colour=(20, 28, 40))
    v.legend([(LINK["launch"], "起飞"), (LINK["drop"], "下落"), (LINK["step"], "台阶"), (LINK["lift"], "风扇上升"), (LINK["mech"], "机关"),
              ((20, 28, 40), "断点"), (INK, "黑框：走法经过的区域；号码：它的连接，按顺序（列在图下）")])
    v.save(path)
    _list_under(path, rows)


def _list_under(path, rows, per_column: int = 12):
    """The way's steps written under the picture, in columns."""
    from PIL import Image, ImageDraw
    from .render import INK, SURFACE, _font
    if not rows:
        return
    img = Image.open(path)
    f = _font(22)
    cols = [rows[i:i + per_column] for i in range(0, len(rows), per_column)]
    h = 24 + 32 * min(len(rows), per_column) + 16
    out = Image.new("RGB", (img.size[0], img.size[1] + h), SURFACE)
    out.paste(img, (0, 0))
    pen = ImageDraw.Draw(out)
    x = 16
    for col in cols:
        w = max(pen.textlength(t, font=f) for t in col)
        for i, t in enumerate(col):
            pen.text((x, img.size[1] + 12 + 32 * i), t, fill=INK, font=f)
        x += int(w) + 48
    out.save(path)
