"""The coarse level as connectivity between regions: where the ball can get, and over what.

A region is a set of the coarse level's places that a ball of one kind reaches from each other by rolling (roll,
rail and rail-end edges, as the coarse level has them with every mechanism as it is at the start): the strongly
connected components, for each kind of ball apart. A transformer's range of another kind is not part of a ball's
regions: a ball that rolls into it is that kind from there on (`trafo`: the game transforms it, there is no choice),
so for a wood ball a paper transformer across a road cuts the road in two. Every place is thus in up to three
regions, one per kind.

What leads from one region to another is a link: a launch, a drop or a step the coarse level has, a roll that only
goes one way, a roll or a flight into a transformer's range of another kind (the ball goes on as that kind, in its
regions), a paper ball's lift, or a mechanism's edge (through an open gate, pushing a gate open, pushing a push board
over, over a board that is down, over a seesaw from its low end), each with what it needs (the ball's kind, the
mechanisms' states) and what it changes. The coarse level's links where it does not connect ('gap', 'bridge') are
breaks: ways nothing has shown to be there.

The way through is worked out over (region, mechanisms' states) from the sector's reset point, with as few breaks as
there can be, then as few links.

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
BALL = ("木", "石", "纸")
MECH = {coarse.GATE_THROUGH: "through_gate", coarse.GATE_OPERATE: "push_gate_open", coarse.BOARD_PUSH: "push_board_over",
        coarse.BOARD_OVER: "over_board", coarse.SEESAW: "over_seesaw"}
BIG = 30                  # a region of fewer places is drawn only where a way goes through it (and its breaks are not listed)


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


def kind_labels(n: int, roll, tk) -> np.ndarray:
    """(3, n): each place's region for each kind of ball (0, 1, 2 ...), -1 where a ball of that kind cannot be (in a
    transformer's range of another kind). roll[u]: the places u rolls to. tk[u]: the kind of transformer whose range
    u is in, or -1."""
    tk = np.asarray(tk)
    out = np.full((3, n), -1, np.int64)
    for k in range(3):
        keep = (tk < 0) | (tk == k)
        lab = components(n, [[v for v in roll[u] if keep[v]] if keep[u] else [] for u in range(n)])
        lab[~keep] = -1
        _u, out[k][keep] = np.unique(lab[keep], return_inverse=True)
    return out


class Regions:
    """The regions of every kind with one numbering: region r is (kind, label); r = base[kind] + label."""

    def __init__(self, labs, places):
        self.labs = labs
        self.base = np.r_[0, np.cumsum([int(labs[k].max()) + 1 for k in range(3)])]
        self.kind = np.repeat(np.arange(3), np.diff(self.base))
        self.places = places

    def __len__(self):
        return int(self.base[-1])

    def of(self, k: int, u: int) -> int:
        return int(self.base[k] + self.labs[k][u]) if self.labs[k][u] >= 0 else -1

    def name(self, r: int) -> str:
        k = int(self.kind[r])
        return "%s%d" % (BALL[k], r - int(self.base[k]))

    def members(self, r: int) -> np.ndarray:
        k = int(self.kind[r])
        return np.flatnonzero(self.labs[k] == r - self.base[k])


def links_of(sg, R: Regions) -> dict:
    """The links between regions of a sector's state graph, one per (from, to, what, condition): key -> dict(cost (the
    least), count, at (an example: (from place, to place), local)). Keys: ("plain", a, b, edge kind, mask, kind
    from, kind to), ("lift", a, b, kind to), ("mech", a, b, type, m, side, kind from, kind to). The kind it goes on
    as is that of the transformer's range it lands in, or that a link passes through (ek, lk), or its own."""
    out: dict = {}

    def add(key, c, u, v):
        e = out.get(key)
        if e is None:
            out[key] = dict(cost=float(c), count=1, at=(int(u), int(v)))
        else:
            e["count"] += 1
            if c < e["cost"]:
                e["cost"], e["at"] = float(c), (int(u), int(v))

    for k in range(3):
        for u in np.flatnonzero(R.labs[k] >= 0):
            a = R.of(k, u)
            for v, c, kind, ek, mask in sg.fwd[u]:
                t = int(sg.tk[v]) if sg.tk[v] >= 0 else (int(ek) if ek >= 0 else k)
                b = R.of(t, v)
                if b >= 0 and b != a:
                    add(("plain", a, b, kind, int(mask), k, t), c, u, v)
            if k == coarse.PAPER:
                for v, c, lk in sg.lfwd[u]:
                    t = int(sg.tk[v]) if sg.tk[v] >= 0 else (int(lk) if lk >= 0 else k)
                    b = R.of(t, v)
                    if b >= 0:
                        add(("lift", a, b, t), c, u, v)
            for v, c, _kind, m, typ, side in sg.cfwd[u]:
                if (typ == coarse.GATE_OPERATE and k != coarse.STONE) or (typ == coarse.BOARD_PUSH and k == coarse.PAPER):
                    continue                                  # a wood or paper ball cannot open a gate, a paper ball cannot push a board over
                t = int(sg.tk[v]) if sg.tk[v] >= 0 else k
                b = R.of(t, v)
                if b >= 0:
                    add(("mech", a, b, int(typ), int(m), int(side), k, t), c, u, v)
    return out


def _moves(key, bits: int):
    """What a link does with the mechanism bits `bits`: [(kind it goes on as, bits after, what)] (empty: not there)."""
    if key[0] == "plain":
        _p, _a, _b, kind, mask, _k, t = key
        return [(t, bits, kind)] if (bits & mask) == mask else []
    if key[0] == "lift":
        return [(key[3], bits, "lift")]
    _m, _a, _b, typ, m, side, k, t = key
    bit = (bits >> m) & 1
    if typ == coarse.GATE_THROUGH and bit == 1:
        return [(t, bits, MECH[typ])]
    if typ == coarse.GATE_OPERATE and bit == 0 and k == coarse.STONE:
        return [(t, bits | (1 << m), MECH[typ])]
    if typ == coarse.BOARD_PUSH and k != coarse.PAPER:
        return [(t, bits | (1 << m), MECH[typ])]
    if typ == coarse.BOARD_OVER and bit == 1:
        return [(t, bits, MECH[typ])]
    if typ == coarse.SEESAW and bit == side:
        return [(t, (bits & ~(1 << m)) | ((1 - side) << m) if k == coarse.STONE else bits, MECH[typ])]
    return []


def connect(links: dict, start, goals):
    """The way through the regions from start = (region, kind, bits) to any of the regions `goals` with the fewest
    breaks, then the fewest links: (breaks, [(region, kind, bits, how it got there, the link's key or None)], every
    (region, kind, bits) reached without a break); (None, [], ...) if there is none even with breaks."""
    goals = set(goals)
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
        r, _k, bits = node
        if r in goals:
            end = node
            break
        for key in by_from.get(r, ()):
            for k2, b2, what in _moves(key, bits):
                nxt = (key[2], k2, b2)
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
    """Step 1: a sector's regions for each kind of ball, where its reset point and its goal are."""
    from . import trafo
    from .solve import state_graph_for
    from .staged import nearest_node
    sg, state_of = state_graph_for(g, info, sector, world, operate_cost)
    n, M = sg.n, sg.M
    P = g.pos[sg.glob]
    R = Regions(kind_labels(n, [[v for v, _c, kind, _e, _m in sg.fwd[u] if kind in ROLL] for u in range(n)], sg.tk), P)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    goal = g.nearest(anchors[sector], 25.0)
    togo = sg.togo(goal)
    s0 = state_of(entry_kind, world.scene())
    reset = np.array(info["reset_points"][sector - 1], float)
    u0 = nearest_node(g, togo[s0], reset)
    if sg.loc[u0] < 0:
        u0 = int(sg.glob[np.argmin(np.linalg.norm(P - reset, axis=1))])
    l0 = int(sg.loc[u0])
    k0 = s0 >> M
    if R.labs[k0][l0] < 0:                                    # the reset point is in a transformer's range of another kind
        k0 = int(sg.tk[l0])
    gl = int(sg.loc[goal])
    regions = {}
    for r in range(len(R)):
        idx = R.members(r)
        regions[r] = dict(name=R.name(r), kind=KINDS[int(R.kind[r])], places=int(len(idx)), centre=P[idx].mean(0).tolist(),
                          lo=P[idx].min(0).tolist(), hi=P[idx].max(0).tolist())
    return dict(sector=sector, sg=sg, R=R, places=P, regions=regions, transformers=trafo.of(info, sector),
                start=(R.of(k0, l0), k0, s0 & ((1 << M) - 1)), start_place=int(u0), s0=int(s0), togo=togo, goal=int(goal),
                goal_regions=sorted(R.of(k, gl) for k in range(3) if gl >= 0 and R.labs[k][gl] >= 0),
                mechs=[dict(name=m["name"], type=m["type"]) for m in sg.mechs], entry_kind=entry_kind)


def link(res: dict) -> dict:
    """Step 2: the links between a sector's regions (`scan`), the way through them with the fewest breaks, and the
    coarse level's own cheapest plan as regions. Adds them to res and returns it."""
    sg, R, M = res["sg"], res["R"], len(res["mechs"])
    res["links"] = links_of(sg, R)
    res["breaks"], res["path"], res["free"] = connect(res["links"], res["start"], res["goal_regions"])
    nodes, kinds, states = sg.plan(res["s0"], res["start_place"], res["togo"])
    plan = []
    for i, gu in enumerate(nodes):
        st = states[i]
        r = R.of(st >> M, int(sg.loc[gu]))
        if plan and plan[-1]["region"] == r and plan[-1]["state"] == st:
            continue
        plan.append(dict(region=r, state=st, kind=KINDS[st >> M], bits=st & ((1 << M) - 1), how=kinds[i - 1] if i else "start", at=sg.g.pos[gu].tolist()))
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
        _p, _a, _b, kind, mask, k, t = key
        s = {"launch": "起飞", "drop": "下落", "step": "台阶", "roll": "滚", "rail": "钢轨", "rail end": "钢轨尽头", "gap": "断点(gap)",
             "bridge": "断点(bridge)"}.get(kind, kind)
        if kind == "roll" and t == k:
            s = "单向滚下"
        if mask:
            s += "，要机关位 %d" % mask
        if t != k:
            s += ("进" if kind == "roll" else "，落进") + "%s变的范围，变%s球" % (BALL[t], BALL[t])
        return s
    if key[0] == "lift":
        return "纸球乘风扇上升" + ("，进%s变的范围，变%s球" % (BALL[key[3]], BALL[key[3]]) if key[3] != coarse.PAPER else "")
    _m, _a, _b, typ, m, side, k, t = key
    return {coarse.GATE_THROUGH: "过开着的栅栏", coarse.GATE_OPERATE: "石球推开栅栏", coarse.BOARD_PUSH: "推倒推板",
            coarse.BOARD_OVER: "从倒下的推板上过", coarse.SEESAW: "从低端过跷跷板" + ("（把它翻过来）" if k == coarse.STONE else "")}[typ] \
        + "（机关 %d）" % m + ("，变%s球" % BALL[t] if t != k else "")


def _r1(v):
    return [round(float(x), 1) for x in v]


def summary_regions(res: dict) -> dict:
    """Step 1 as plain data: every region of the sector, for each kind of ball."""
    R = res["R"]
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], mechanisms=res["mechs"],
                transformers=[dict(name=t["name"], kind=t["kind"], pos=_r1(t["pos"])) for t in res["transformers"]],
                start_region=R.name(res["start"][0]), goal_regions=[R.name(r) for r in res["goal_regions"]],
                regions={KINDS[k]: {d["name"]: dict(places=d["places"], centre=_r1(d["centre"]), lo=_r1(d["lo"]), hi=_r1(d["hi"]))
                                    for d in res["regions"].values() if d["kind"] == KINDS[k]} for k in range(3)})


def summary_links(res: dict) -> dict:
    """Step 2 as plain data: the links between the sector's regions, and the way with the fewest breaks."""
    P, R = res["places"], res["R"]
    links = []
    for key, e in sorted(res["links"].items(), key=lambda kv: (kv[0][1], kv[0][2], str(kv[0]))):
        links.append(dict(**{"from": R.name(key[1]), "to": R.name(key[2])}, what=describe(key), cost=round(e["cost"], 1), count=e["count"],
                          at=[[round(float(x), 2) for x in P[e["at"][0]]], [round(float(x), 2) for x in P[e["at"][1]]]],
                          is_break=bool(key[0] == "plain" and key[3] in BREAK), big=bool(min(res["regions"][key[1]]["places"], res["regions"][key[2]]["places"]) >= BIG)))
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], start_region=R.name(res["start"][0]), goal_regions=[R.name(r) for r in res["goal_regions"]],
                fewest_breaks=res["breaks"],
                way=[dict(region=R.name(r), kind=KINDS[k], mechanism_bits=b, by=what, link=describe(key)) for r, k, b, what, key in res["path"]],
                coarse_plan=[dict(region=R.name(p["region"]) if p["region"] >= 0 else None, kind=p["kind"], mechanism_bits=p["bits"], by=p["how"], at=_r1(p["at"]))
                             for p in res["plan"]],
                links=[{k: v for k, v in l.items() if k not in ("is_break", "big")} for l in links if not l["is_break"]],
                breaks=[{k: v for k, v in l.items() if k not in ("is_break", "big")} for l in links if l["is_break"] and l["big"]],
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
    log("step 1, regions (the places a ball of one kind reaches from each other by rolling):")
    for res in done:
        R = res["R"]
        per = ", ".join("%s %d (%d of %d places or more)" % (KINDS[k], int((R.kind == k).sum()),
                                                             sum(1 for d in res["regions"].values() if d["kind"] == KINDS[k] and d["places"] >= BIG), BIG) for k in range(3))
        log("   sector %d: %s; the reset point in %s, the goal in %s" % (res["sector"], per, R.name(res["start"][0]),
                                                                       "/".join(R.name(r) for r in res["goal_regions"])))
    log("step 2, the links between them, and the way with the fewest breaks:")
    for res in done:
        R = res["R"]
        n_links = sum(1 for k in res["links"] if not (k[0] == "plain" and k[3] in BREAK))
        log("   sector %d (enters as %s): %d links, %d breaks between regions; %s" % (
            res["sector"], res["entry_kind"], n_links, len(res["links"]) - n_links,
            "no way to the goal even with breaks" if res["breaks"] is None else
            "the goal is reached with %d break%s, over %d links" % (res["breaks"], "" if res["breaks"] == 1 else "s", len(res["path"]) - 1)))
        for r, k, b, what, key in res["path"]:
            log("      %-6s bits %d  %s" % (R.name(r), b, describe(key) or what))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = dict(map=name, generator="ballistics", what="", note="the coarse level only: nothing is rolled or searched")
    json.dump(dict(head, what="step 1: the regions, the places a ball of one kind reaches from each other by rolling", sectors=[summary_regions(r) for r in done]),
              open(out / (name + "-regions.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    json.dump(dict(head, what="step 2: the links between the regions, and the way with the fewest breaks", sectors=[summary_links(r) for r in done]),
              open(out / (name + "-links.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    if png is not None:
        png = Path(png)
        png.mkdir(parents=True, exist_ok=True)
        for res in done:
            s = res["sector"]
            k = KINDS.index(res["entry_kind"])
            mine = [d for d in res["regions"].values() if d["kind"] == res["entry_kind"]]
            draw_regions(mesh, res, png / ("%s-regions-s%d.png" % (name, s)),
                         "第 %d 小节，第一步：%s球只靠滚动互相到得了的区域（%d 个，%d 个点以上的 %d 个）；圈是变球器的范围" % (
                             s, BALL[k], len(mine), BIG, sum(1 for d in mine if d["places"] >= BIG)))
            draw(mesh, res, png / ("%s-links-s%d.png" % (name, s)),
                 "第 %d 小节，第二步：区域之间的连接；%s" % (s, "到不了目标" if res["breaks"] is None else "到目标最少要跨 %d 个断点" % res["breaks"]))
    return done


PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (57, 59, 121), (99, 121, 57), (140, 109, 49), (132, 60, 57), (123, 65, 115)]
LINK = {"launch": (212, 122, 28), "drop": (16, 132, 120), "step": (82, 81, 78), "roll": (150, 149, 144), "lift": (62, 100, 150), "mech": (196, 60, 48),
        "transform": (148, 52, 160)}
TRAFO = {"wood": (140, 86, 40), "stone": (90, 90, 96), "paper": (230, 230, 236)}


def _link_colour(key):
    if key[0] == "plain" and key[3] in BREAK:
        return (20, 28, 40)
    if key[-1] != (key[5] if key[0] == "plain" else key[6] if key[0] == "mech" else coarse.PAPER):
        return LINK["transform"]
    return LINK["mech"] if key[0] == "mech" else LINK["lift"] if key[0] == "lift" else LINK.get(key[3], (82, 81, 78))


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


def _view(mesh, res, title):
    from .render import TopView
    P = res["places"]
    lo, hi = P.min(0), P.max(0)
    return TopView(mesh, box=(lo[0], hi[0], lo[2], hi[2]), width=2000, title=title, height=True)


def _fill(v, res, regions, alpha):
    order = sorted(regions, key=lambda r: -res["regions"][r]["places"])
    for i, r in enumerate(order):
        c = PALETTE[i % len(PALETTE)]
        for p in res["places"][res["R"].members(r)]:
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 2, y - 2, x + 2, y + 2), fill=c + (alpha(r),))


def _names(v, res, regions, least: int):
    from .render import INK, SURFACE, _font
    f = _font(22)
    for r in regions:
        d = res["regions"][r]
        if d["places"] >= least:
            x, y = v.P(d["centre"][0], d["centre"][2])
            v.pen.rectangle((x - 2, y - 2, x + v.pen.textlength(d["name"], font=f) + 4, y + 26), fill=SURFACE + (220,))
            v.pen.text((x, y), d["name"], fill=INK, font=f)


def _trafos(v, res):
    """Each transformer's range as a ring, named by its kind."""
    from .render import INK, _font
    from .trafo import rules
    rad = rules().radius
    f = _font(22)
    for t in res["transformers"]:
        x, y = v.P(t["pos"][0], t["pos"][2])
        rr = rad * v.k
        v.pen.ellipse((x - rr, y - rr, x + rr, y + rr), outline=INK + (255,), width=4)
        v.pen.ellipse((x - rr + 3, y - rr + 3, x + rr - 3, y + rr - 3), outline=TRAFO[t["kind"]] + (255,), width=3)
        v.pen.text((x + rr + 4, y - 12), "%s变" % BALL[KINDS.index(t["kind"])], fill=INK, font=f)


def draw_regions(mesh, res: dict, path, title: str = ""):
    """Step 1: the regions of the kind of ball the sector is entered with (those of BIG places or more, and the one
    the reset point is in), each of 300 places or more named; the transformers' ranges as rings."""
    k = KINDS.index(res["entry_kind"])
    v = _view(mesh, res, title)
    mine = [r for r, d in res["regions"].items() if d["kind"] == res["entry_kind"] and (d["places"] >= BIG or r == res["start"][0])]
    _fill(v, res, mine, lambda r: 220)
    _names(v, res, mine, 300)
    _trafos(v, res)
    sg = res["sg"]
    v.dot(sg.g.pos[res["start_place"]], "起点", colour=(208, 59, 59))
    v.dot(sg.g.pos[res["goal"]], "目标", colour=(20, 28, 40))
    v.save(path)


def draw(mesh, res: dict, path, title: str = ""):
    """Step 2: the regions of the entry kind faint, the links between them as thin lines, and the way with the fewest
    breaks: the regions it goes through (of whichever kind) filled and outlined, each of its links a numbered arrow
    from where it leaves one region to where it lands in the next. Within a region nothing is drawn: rolling gets the
    ball anywhere in it."""
    from .render import INK, SURFACE, _font
    P, R = res["places"], res["R"]
    v = _view(mesh, res, title)
    on_way = list(dict.fromkeys(p[0] for p in res["path"]))
    back = [r for r, d in res["regions"].items() if d["kind"] == res["entry_kind"] and d["places"] >= BIG and r not in on_way]
    _fill(v, res, back + on_way, lambda r: 220 if r in on_way else 70)
    for r in on_way:                                       # the regions of the way, outlined
        for p in _rim(P[R.members(r)]):
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 3, y - 3, x + 3, y + 3), fill=INK + (255,))
    shown = set(back) | set(on_way)
    for key, e in res["links"].items():
        if key[1] not in shown or key[2] not in shown or (key[0] == "plain" and key[3] in BREAK):
            continue
        v.halo_line([P[e["at"][0]], P[e["at"][1]]], _link_colour(key), 2, ring=1, alpha=110)
    _trafos(v, res)
    _names(v, res, back, 300)
    _names(v, res, on_way, 0)
    fb = _font(22)
    n, rows, discs = 0, [], []
    for r, k, b, what, key in res["path"]:
        if what == "start":
            continue
        n += 1
        e = res["links"][key]
        a, bp = P[e["at"][0]], P[e["at"][1]]
        _arrow(v, a, bp, _link_colour(key), 6)
        x, y = v.P(bp[0], bp[2])
        x, y = x + 18, y - 18
        rows.append("%d　%s → %s　%s" % (n, R.name(key[1]), R.name(key[2]), describe(key)))
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
              (LINK["transform"], "变球"), ((20, 28, 40), "断点"), (INK, "黑框：走法经过的区域；号码：它的连接（列在图下）")])
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
