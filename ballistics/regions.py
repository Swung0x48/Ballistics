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


def analyse(g, info, sector: int, world, entry_kind: str = "wood", operate_cost: float = 20.0) -> dict:
    """A sector's regions, the links between them, and the way through them from its reset point to its goal."""
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
    start = (int(lab[sg.loc[u0]]), s0 >> M, s0 & ((1 << M) - 1))
    goal_region = int(lab[sg.loc[goal]]) if sg.loc[goal] >= 0 else -1
    links = links_of(sg, lab)
    trafo_kinds: dict = {}
    for u in np.flatnonzero(sg.tk >= 0):
        trafo_kinds.setdefault(int(lab[u]), set()).add(int(sg.tk[u]))
    breaks, path, free = connect(links, trafo_kinds, M, start, goal_region)
    # the coarse level's own cheapest plan (point by point, over the same states), as regions
    nodes, kinds, states = sg.plan(s0, u0, togo)
    plan = []
    for i, gu in enumerate(nodes):
        r = int(lab[sg.loc[gu]])
        st = states[i]
        how = kinds[i - 1] if i else "start"
        if plan and plan[-1]["region"] == r and plan[-1]["state"] == st:
            continue
        plan.append(dict(region=r, state=st, kind=KINDS[st >> M], bits=st & ((1 << M) - 1), how=how, at=g.pos[gu].tolist()))
    regions = {}
    for r in range(len(sizes)):
        idx = np.flatnonzero(lab == r)
        regions[r] = dict(places=int(sizes[r]), centre=P[idx].mean(0).tolist(), lo=P[idx].min(0).tolist(), hi=P[idx].max(0).tolist(),
                          transformers=sorted(KINDS[t] for t in trafo_kinds.get(r, ())))
    return dict(sector=sector, sg=sg, lab=lab, places=P, regions=regions, links=links, start=start, start_place=int(u0), goal_region=goal_region,
                goal=int(goal), breaks=breaks, path=path, free=free, plan=plan, togo_start=float(togo[s0, u0]),
                mechs=[dict(name=m["name"], type=m["type"]) for m in sg.mechs], entry_kind=entry_kind)


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


def summary(res: dict) -> dict:
    """The analysis as plain data, for a file."""
    M = len(res["mechs"])
    used = {p[0] for p in res["path"]} | {p["region"] for p in res["plan"]}
    regs = {r: d for r, d in res["regions"].items() if d["places"] >= BIG or r in used or r == res["start"][0] or r == res["goal_region"]}
    links = []
    for key, e in sorted(res["links"].items(), key=lambda kv: (kv[0][1], kv[0][2], str(kv[0]))):
        if key[1] not in regs or key[2] not in regs:
            continue
        P = res["sg"].g.pos[res["sg"].glob]
        links.append(dict(**{"from": key[1], "to": key[2]}, what=describe(key), cost=round(e["cost"], 1), count=e["count"],
                          at=[[round(float(x), 2) for x in P[e["at"][0]]], [round(float(x), 2) for x in P[e["at"][1]]]],
                          is_break=bool(key[0] == "plain" and key[3] in BREAK)))
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], mechanisms=res["mechs"],
                regions={str(r): dict(d, centre=[round(x, 1) for x in d["centre"]], lo=[round(x, 1) for x in d["lo"]], hi=[round(x, 1) for x in d["hi"]])
                         for r, d in regs.items()},
                regions_all=len(res["regions"]), start_region=res["start"][0], goal_region=res["goal_region"],
                fewest_breaks=res["breaks"],
                way=[dict(region=r, kind=KINDS[k], mechanism_bits=b, by=what, link=describe(key)) for r, k, b, what, key in res["path"]],
                coarse_plan=[dict(region=p["region"], kind=p["kind"], mechanism_bits=p["bits"], by=p["how"], at=[round(x, 1) for x in p["at"]]) for p in res["plan"]],
                links=[l for l in links if not l["is_break"]], breaks_between_regions=sum(1 for l in links if l["is_break"]),
                note="mechanism_bits: bit m is mechanism m's state (gate open, push board down, seesaw low at its other end)" if M else "")


PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (57, 59, 121), (99, 121, 57), (140, 109, 49), (132, 60, 57), (123, 65, 115)]
LINK = {"launch": (212, 122, 28), "drop": (16, 132, 120), "step": (82, 81, 78), "roll": (150, 149, 144), "lift": (62, 100, 150), "mech": (196, 60, 48)}


def draw(mesh, res: dict, path, title: str = ""):
    """The sector's regions in colours (the big ones and the ones on the way), the links between them as thin lines,
    and the way through the regions with the fewest breaks as a thick one."""
    from .render import INK, TopView, _font
    P = res["places"]
    lab = res["lab"]
    lo, hi = P.min(0), P.max(0)
    v = TopView(mesh, box=(lo[0], hi[0], lo[2], hi[2]), width=2000, title=title, height=True)
    on_way = [p[0] for p in res["path"]]
    show = {r for r, d in res["regions"].items() if d["places"] >= BIG} | set(on_way)
    order = sorted(show, key=lambda r: -res["regions"][r]["places"])
    colour = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(order)}
    for r in order:
        for p in P[lab == r]:
            x, y = v.P(p[0], p[2])
            v.pen.rectangle((x - 2, y - 2, x + 2, y + 2), fill=colour[r] + (200,))
    for key, e in res["links"].items():
        if key[1] not in show or key[2] not in show or (key[0] == "plain" and key[3] in BREAK):
            continue
        a, b = P[e["at"][0]], P[e["at"][1]]
        c = LINK["mech"] if key[0] == "mech" else LINK["lift"] if key[0] == "lift" else LINK.get(key[3], (82, 81, 78))
        v.halo_line([a, b], c, 2, ring=1, alpha=170)
    f = _font(22)
    for r in order:
        d = res["regions"][r]
        if d["places"] >= 300 or r in on_way:
            x, y = v.P(d["centre"][0], d["centre"][2])
            t = "R%d" % r + ("（%s变）" % "".join({"wood": "木", "stone": "石", "paper": "纸"}[k] for k in d["transformers"]) if d["transformers"] else "")
            v.pen.rectangle((x - 2, y - 2, x + v.pen.textlength(t, font=f) + 4, y + 26), fill=(252, 252, 251, 220))
            v.pen.text((x, y), t, fill=INK, font=f)
    # the way: from each link's place to the next
    pts = []
    for r, k, b, what, key in res["path"]:
        if key is not None:
            e = res["links"][key]
            pts += [P[e["at"][0]], P[e["at"][1]]]
    sg = res["sg"]
    first = sg.g.pos[res["start_place"]]
    if pts:
        v.halo_line([first] + pts + [sg.g.pos[res["goal"]]], (20, 28, 40), 5, ring=2)
    v.dot(first, "起点", colour=(208, 59, 59))
    v.dot(sg.g.pos[res["goal"]], "目标", colour=(20, 28, 40))
    v.legend([(LINK["launch"], "起飞"), (LINK["drop"], "下落"), (LINK["step"], "台阶"), (LINK["lift"], "风扇上升"), (LINK["mech"], "机关"),
              ((20, 28, 40), "断点最少、连接最少的走法")])
    v.save(path)
