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

Nothing is rolled here: `solve.graph_for` caches the coarse graph, with closed-form launches and lifts still
rolled in the kernel. Mechanism rules come from `coarse.StateGraph`.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import coarse

ROLL = ("roll", "rail", "rail end")
BREAK = ("gap", "bridge")
KINDS = ("wood", "stone", "paper")
BALL = ("木", "石", "纸")


class _Text(str):
    """English report text with the original Chinese wording retained for the figures."""

    def __new__(cls, en, zh):
        text = super().__new__(cls, en)
        text.zh = zh
        return text

    @classmethod
    def format(cls, en, zh, *args):
        return cls(en % args, zh % tuple(_chinese(arg) for arg in args))


def _chinese(value):
    if isinstance(value, _Text):
        return value.zh
    if isinstance(value, dict):
        return {_chinese(k): _chinese(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_chinese(v) for v in value]
    return value


def _ball_list(kinds):
    kinds = sorted(set(kinds))
    return _Text(", ".join(KINDS[k] for k in kinds), "".join(BALL[k] for k in kinds))


ELEVATOR_PAPER = _Text("paper can push with a run-up [WIKI][OWNER]; enough run-up [ASSUMED]",
                       "纸球助跑能推下墙门【wiki】【主人确认】；假定助跑足够【假】")


FLOAT_DIRECTION = {coarse.FLOAT_MINUS: _Text("side -", "侧面−"), coarse.FLOAT_PLUS: _Text("side +", "侧面+"),
                   coarse.FLOAT_END: _Text("end", "末端")}
MECH = {coarse.GATE_THROUGH: "through_gate", coarse.GATE_OPERATE: "push_gate_open", coarse.BOARD_PUSH: "push_board_over",
        coarse.BOARD_OVER: "over_board", coarse.SEESAW: "over_seesaw", coarse.CORK: "over_cork_bridge", coarse.FLOAT_ROW: "over_float_row",
        coarse.ROCK_OVER: "over_falling_rock", coarse.ROCK_PUSH: "drop_falling_rock",
        coarse.T_OVER: "over_t_board", coarse.T_PUSH: "push_t_board", coarse.T_CLEAR: "past_t_board",
        coarse.FLAP_OVER: "over_flap", coarse.FLAP_PUSH: "push_flap", coarse.PENDULUM: "push_pendulum",
        coarse.PENDULUM_BOARD: "pendulum_board", coarse.PENDULUM_BOX: "pendulum_box", coarse.ELEVATOR: "ride_elevator",
        coarse.SWING: "ride_swing", coarse.SANDBAG: "pass_sandbag"}
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


def kind_labels(n: int, roll, tk, allow=None) -> np.ndarray:
    """(3, n): each place's region for each kind of ball (0, 1, 2 ...), -1 where a ball of that kind cannot be (in a
    transformer's range of another kind). roll[u]: the places u rolls to. tk[u]: the kind of transformer whose range
    u is in, or -1."""
    tk = np.asarray(tk)
    out = np.full((3, n), -1, np.int64)
    for k in range(3):
        keep = (tk < 0) | (tk == k)
        lab = components(n, [[v for v in roll[u] if keep[v] and (allow is None or allow(k,u,v))] if keep[u] else [] for u in range(n)])
        lab[~keep] = -1
        _u, out[k][keep] = np.unique(lab[keep], return_inverse=True)
    return out


def roll_labels(sg):
    """Kind-specific regions of the unguarded rolling graph (also used to prune float exits)."""
    roll = [[v for v, _c, kind, _e, mask in sg.fwd[u]
             if kind in ROLL and not mask and (u, v, kind) not in sg.guards] for u in range(sg.n)]
    masks=getattr(sg, "terrain_masks", {})
    return kind_labels(sg.n, roll, sg.tk,
                       lambda k,u,v: masks.get((u,v,"roll"),7)&(1<<k))


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

    def name(self, r: int, lang="en") -> str:
        k = int(self.kind[r])
        return "%s%d" % ((BALL if lang == "zh" else "WSP")[k], r - int(self.base[k]))

    def _label(self, r: int) -> _Text:
        return _Text(self.name(r), self.name(r, "zh"))

    def members(self, r: int) -> np.ndarray:
        k = int(self.kind[r])
        return np.flatnonzero(self.labs[k] == r - self.base[k])


def links_of(sg, R: Regions) -> dict:
    """The links between regions of a sector's state graph, one per (from, to, what, condition): key -> dict(cost (the
    least), count, at (an example: (from place, to place), local)). Keys: ("plain", a, b, edge kind, mask, kind
    from, kind to[, mask of bits that must be clear]), ("lift", a, b, kind to), ("mech", a, b, type, m, side, kind from, kind to). The kind it goes on
    as is that of the transformer's range it lands in, or that a link passes through (ek, lk), or its own."""
    out: dict = {}
    guards = getattr(sg, "guards", {})
    lift_guards = getattr(sg, "lift_guards", {})
    flap_bits = sum(1 << m for m, mc in enumerate(getattr(sg, "mechs", ())) if mc["type"] == "flap")

    def add(key, c, u, v):
        e = out.get(key)
        if e is None:
            out[key] = e = dict(cost=float(c), count=1, at=(int(u), int(v)))
        else:
            e["count"] += 1
            if c < e["cost"]:
                e["cost"], e["at"] = float(c), (int(u), int(v))
        return e

    for k in range(3):
        for u in np.flatnonzero(R.labs[k] >= 0):
            a = R.of(k, u)
            for v, c, kind, ek, mask in sg.fwd[u]:
                prices = getattr(sg, "launch_cost", {}).get((u, v)) if kind == "launch" else None
                if prices is not None:
                    if k not in prices:
                        continue
                    c = prices[k]
                t = int(sg.tk[v]) if sg.tk[v] >= 0 else (int(ek) if ek >= 0 else k)
                b = R.of(t, v)
                if b >= 0 and b != a:
                    off = guards.get((u, v, kind), 0)
                    clear = (off, off & flap_bits) if off & flap_bits else ((off,) if off else ())
                    add(("plain", a, b, kind, int(mask), k, t) + clear, c, u, v)
            if k == coarse.PAPER:
                for v, c, lk in sg.lfwd[u]:
                    t = int(sg.tk[v]) if sg.tk[v] >= 0 else (int(lk) if lk >= 0 else k)
                    b = R.of(t, v)
                    if b >= 0:
                        masks = lift_guards.get((u, v))
                        add(("lift", a, b, t) + (masks or ()), c, u, v)
            for v, c, _kind, m, typ, side in sg.cfwd[u]:
                if (typ == coarse.ROCK_PUSH and k != coarse.STONE) or (typ == coarse.GATE_OPERATE and k != coarse.STONE and not (k == coarse.WOOD and side == coarse.GATE_WOOD)) or (typ in (coarse.BOARD_PUSH, coarse.T_PUSH, coarse.FLAP_PUSH, *coarse.PENDULUM_OPERATE) and k == coarse.PAPER) \
                        or (typ == coarse.CORK and k == coarse.STONE) or (typ == coarse.FLOAT_ROW and not (side & (1 << k))):
                    continue
                if (typ == coarse.ELEVATOR or (typ == coarse.GATE_THROUGH and side)) and not side & (1 << k):
                    continue
                t = int(sg.tk[v]) if sg.tk[v] >= 0 and typ not in coarse.PENDULUM_OPERATE else k
                b = R.of(t, v)
                if b >= 0 and (b != a or typ not in (*coarse.STATE_OVER, coarse.FLAP_OVER)):
                    e = add(("mech", a, b, int(typ), int(m), int(side), k, t), c, u, v)
                    if typ == coarse.ELEVATOR:
                        gu, gv = int(sg.glob[u]), int(sg.glob[v])
                        ride = sg.mechs[m]["rides"][gu, gv, side]
                        e.setdefault("elevator_exits", {})[v] = ride
                        if e["at"] == (u, v):
                            e["elevator"] = ride
                    if typ in (coarse.ROCK_PUSH, coarse.T_PUSH, coarse.FLAP_PUSH, *coarse.PENDULUM_OPERATE):
                        e.setdefault("sites", set()).add(int(u))
    return out


def _moves(key, bits: int, lock: int = 0):
    """What a link does with the mechanism bits `bits`: [(kind it goes on as, bits after, what)] (empty: not there).
    lock: bits that never change (a what-if): a locked gate is not opened, a locked board not pushed over (so not
    crossed that way), a stone ball goes over a locked seesaw without tipping it."""
    got = _moves_free(key, bits)
    if not lock or not got:
        return got
    t, b2, what = got[0]
    if not (b2 ^ bits) & lock:
        return got
    return [(t, bits, what)] if key[0] == "mech" and key[3] == coarse.SEESAW else []


def _moves_free(key, bits: int):
    if key[0] == "plain":
        _p, _a, _b, kind, mask, _k, t, *off = key
        return [(t, bits, kind)] if (bits & mask) == mask and not (off and bits & off[0]) else []
    if key[0] == "lift":
        on, off = key[4:] if len(key) > 4 else (0, 0)
        return [(key[3], bits, "lift")] if bits & on == on and not bits & off else []
    _m, _a, _b, typ, m, side, k, t = key
    bit = (bits >> m) & 1
    if typ == coarse.ELEVATOR and not (bit or bits & (side >> coarse.ELEVATOR_SHIFT)) and side & (1 << k):
        return [(t, bits | (1 << m), MECH[typ])]
    if typ in (coarse.GATE_THROUGH, *coarse.STATE_OVER) and bit == 1:
        return [(t, bits, MECH[typ])]
    if typ in coarse.OPERATE and bit == 0 and (k == coarse.STONE or (k == coarse.WOOD and (typ == coarse.T_PUSH or (typ == coarse.GATE_OPERATE and side == coarse.GATE_WOOD)))):
        return [(t, bits | (1 << m), MECH[typ])]
    if typ in coarse.PENDULUM_OPERATE and bit == 0 and k != coarse.PAPER:
        return [(t, bits | (1 << m), MECH[typ])]
    if bit == side and (typ == coarse.FLAP_OVER or (typ == coarse.FLAP_PUSH and k != coarse.PAPER)):
        return [(t, bits ^ (1 << m) if typ == coarse.FLAP_PUSH else bits, MECH[typ])]
    if typ == coarse.BOARD_PUSH and k != coarse.PAPER:
        return [(t, bits | (1 << m), MECH[typ])]
    if typ == coarse.BOARD_OVER and bit == 1:
        return [(t, bits, MECH[typ])]
    if typ == coarse.SEESAW and bit == side:
        return [(t, (bits & ~(1 << m)) | ((1 - side) << m) if k == coarse.STONE else bits, MECH[typ])]
    if typ in (coarse.PENDULUM, *coarse.TIMED) or (typ == coarse.CORK and k != coarse.STONE) or (typ == coarse.FLOAT_ROW and side & (1 << k)):
        return [(t, bits, MECH[typ])]
    return []


def connect(links: dict, start, goals, lock: int = 0):
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
            for k2, b2, what in _moves(key, bits, lock):
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


_SG_CACHE: dict = {}


def scan(g, info, sector: int, world, entry_kind: str = "wood", operate_cost: float = 20.0, overrides=None) -> dict:
    """Step 1: a sector's regions for each kind of ball, where its reset point and its goal are. overrides (a
    what-if, `Overrides`): placements taken as not there, mechanisms put in another state at the start, mechanisms
    whose state never changes."""
    ov = overrides or Overrides()
    from . import trafo
    from .solve import state_graph_for
    from .coarse import nearest_node
    # the state graph does not depend on the entry kind: `run` scans a sector once per kind it can be entered as,
    # with the same world, so the last one built is kept (one entry only)
    key = (id(g), sector, id(world), operate_cost, frozenset(ov.without))
    if _SG_CACHE.get("key") != key:
        _SG_CACHE.clear()
        _SG_CACHE.update(key=key, value=state_graph_for(g, info, sector, world, operate_cost, cork=True, without=ov.without))
    sg, state_of = _SG_CACHE["value"]
    M = sg.M
    P = sg.g.pos[sg.glob]
    R = Regions(sg.float_roll_labels if sg.float_roll_labels is not None else roll_labels(sg), P)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    goal = g.nearest(anchors[sector], 25.0)
    s0 = state_of(entry_kind, world.scene())
    lock = 0
    for m, mc in enumerate(sg.mechs):
        nm = mc["name"].split("/")[0]
        names = mc.get("placements", (nm,)) if mc["type"] == "box" else (nm,)
        state_name = next((n for n in names if n in ov.state), None)
        if state_name is not None:
            value = ov.state[state_name]
            labels = ()
            if mc["type"] == "seesaw" and str(value).strip().lower().startswith(("low end at ", "低端在")):
                links = links_of(sg, R)
                keys = (k for k in links if k[0] == "mech" and k[3] == coarse.SEESAW and k[4] == m)
                labels = _seesaw_states(keys, links, P)
            s0 = (s0 & ~(1 << m)) | (ov.bit(mc["type"], value, labels) << m)
        if ov.lock.intersection(names):
            lock |= 1 << m
            if mc["type"] == "elevator":
                s0 &= ~(1 << m)
    togo = sg.togo(goal, lock=lock)
    reset = np.array(info["reset_points"][sector - 1], float)
    u0 = nearest_node(g, togo[s0, :len(g.pos)], reset)
    # no place by the reset point reaches the goal in this state: start at the reset point all the same, not at some
    # place far off that does (which would make a way that the ball under the reset point does not have)
    if np.linalg.norm(g.pos[u0] - reset) > 3.0 or sg.loc[u0] < 0:
        real = sg.glob[sg.glob < len(g.pos)]
        u0 = int(real[np.argmin(np.linalg.norm(g.pos[real] - reset, axis=1))])
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
    res = dict(sector=sector, sg=sg, R=R, places=P, regions=regions, transformers=[t for t in trafo.of(info, sector) if t["name"] not in ov.without], lock=lock,
                start=(R.of(k0, l0), k0, s0 & ((1 << M) - 1)), start_place=int(u0), s0=int(s0), togo=togo, goal=int(goal),
                goal_regions=sorted(R.of(k, gl) for k in range(3) if gl >= 0 and R.labs[k][gl] >= 0),
                mechs=[dict(name=m["name"], type=m["type"]) for m in sg.mechs], entry_kind=entry_kind)
    if sg.pendulums or sg.timed:
        res["_placement_context"] = (info, world, operate_cost, ov, g)
    return res


def shared_route(g, starts, goals, *, allow=None, transformations=()):
    """Choose connectivity directly on shared nodes, not projected edges.

    This is a sequence of local representatives, not a continuous replay.
    Unknown portals are excluded. Prefer checked edges, then fewer links.
    A caller with a mechanism state can further restrict allowed transfers.
    """
    from collections import defaultdict
    from heapq import heappop, heappush
    from itertools import count
    shared=g.info['shared_motion_graph']; layout=g.info['conformal_search']
    edges=defaultdict(dict)
    def add(a,b,status,kind,row):
        if a==b or (allow is not None and not allow(a,b,row)):return
        if b in edges[a] and edges[a][b]['status']=='verified':return
        edges[a][b]=dict(source=a,target=b,status=status,kind=kind,witness=row)
    for (a,b),row in shared.get('retained_rolling',{}).items():add(a,b,'retained_verified_roll','roll',row)
    for (a,b),row in shared['rolling'].items():add(a,b,'verified','roll',row)
    for (a,b),rows in shared['witnesses'].items():
        row=min((r for r in rows if allow is None or allow(a,b,r)),key=lambda r:r['ticks'],default=None)
        if row is not None:add(a,b,'verified','rail' if row.get('rail_parts') else 'air',row)
    # The retained archive's coarse indices are only endpoint anchors; all
    # traversal after admission is over the shared nodes.
    support=g.info.get('terrain_support',())
    for (u,v),edge in g.info.get('launches',{}).items():
        if not (0<=u<len(support) and 0<=v<len(support)):continue
        for row in edge.get('variants',(edge,)):
            if row.get('witness_status')!='retained_legacy':continue
            for a in layout.mapped(row['ball'],support[u]):
                for b in layout.mapped(row['ball'],support[v]):add(a,b,'retained_legacy','air',row)
    for a,b,row in transformations:add(a,b,'verified','transform',row)
    prev={a:None for a in starts}; score=dict.fromkeys(prev,(0,0)); order=count()
    heap=[(0,0,next(order),a) for a in prev]; goals=set(goals)
    while heap:
        legacy,hops,_,a=heappop(heap)
        if score[a]!=(legacy,hops):continue
        if a in goals:
            path=[]
            while prev[a] is not None:
                row=prev[a];path.append(row);a=row['source']
            return dict(status='local_witness_sequence',nodes=[a,*[r['target'] for r in reversed(path)]],
                        edges=path[::-1],reached_count=len(prev))
        for b,row in edges.get(a,{}).items():
            cost=(legacy+(row['status']!='verified'),hops+1)
            if b in score and score[b]<=cost:continue
            prev[b]=row;score[b]=cost;heappush(heap,(*cost,next(order),b))
    frontier=[(a,b) for a in prev for b in shared['candidates'].get(a,()) if b not in prev]
    return dict(status='no_route_in_checked_shared_subgraph',nodes=[],edges=[],reached_count=len(prev),
                unresolved_frontier=frontier,reached_nodes=tuple(sorted(prev,key=repr)))


def link(res: dict) -> dict:
    """Step 2: the links between a sector's regions (`scan`), the way through them with the fewest breaks, and the
    coarse level's own cheapest plan as regions. Adds them to res and returns it."""
    sg, R, M = res["sg"], res["R"], len(res["mechs"])
    res["links"] = links_of(sg, R)
    res["breaks"], res["path"], res["free"] = connect(res["links"], res["start"], res["goal_regions"], res.get("lock", 0))
    nodes, kinds, states = sg.plan(res["s0"], res["start_place"], res["togo"], lock=res.get("lock", 0))
    plan = []
    for i, gu in enumerate(nodes):
        st = states[i]
        r = R.of(st >> M, int(sg.loc[gu]))
        if plan and plan[-1]["region"] == r and plan[-1]["state"] == st:
            continue
        plan.append(dict(region=r, state=st, kind=KINDS[st >> M], bits=st & ((1 << M) - 1), how=kinds[i - 1] if i else "start", at=sg.g.pos[gu].tolist()))
    res["plan"] = plan
    return res


def analyse(g, info, sector: int, world, entry_kind: str = "wood", operate_cost: float = 20.0, overrides=None) -> dict:
    """Both steps for one sector."""
    return link(scan(g, info, sector, world, entry_kind, operate_cost, overrides))


def ways_by_exit(res: dict) -> dict:
    """The way with the fewest breaks to the goal as each kind of ball: {kind index: (breaks, path)}; a kind whose
    goal region does not exist or is not reached even with breaks is left out."""
    R = res["R"]
    out = {}
    for r in res["goal_regions"]:
        nb, path, _ = connect(res["links"], res["start"], [r], res.get("lock", 0))
        if nb is not None:
            out[int(R.kind[r])] = (nb, path)
    return out


def choose_kinds(mesh, info, g, overrides=None, first: str = "wood", log=None) -> list:
    """Every sector analysed for each kind of ball it can be entered as, then the kinds chosen for the whole map.

    A sector entered as kind a can end as kind b with breaks(a, b) (its fewest-break way to the goal region of kind b,
    `ways_by_exit`); the next sector is entered as b. The kinds are chosen by dynamic programming over the sectors so
    that the map's breaks, then its links, are fewest; ties go to the kind that comes first in KINDS. A sector whose
    goal no entry kind reaches even with breaks passes the kinds it was entered as on unchanged (as before). Each
    chosen analysis carries its way (`path`, `breaks`), `entry_kind`, `exit_kind` and, for the report, `options`:
    the breaks for every entry and exit kind that was tried."""
    from .solve import world_for
    ov = overrides or Overrides()
    entries = {first}
    analyses, options = [], []
    for sec in range(1, len(info["reset_points"]) + 1):
        world = world_for(mesh, info, "strict", sec)
        tried = {}
        for kind in sorted(entries, key=KINDS.index):
            res = link(scan(g, info, sec, world, entry_kind=kind, overrides=ov))
            res["by_exit"] = ways_by_exit(res)
            tried[kind] = res
        _SG_CACHE.clear()
        opts = {a: {KINDS[b]: (nb, len(path) - 1) for b, (nb, path) in r["by_exit"].items()} for a, r in tried.items()}
        analyses.append(tried)
        options.append(opts)
        entries = {b for o in opts.values() for b in o} or set(entries)
        if log:
            log("   sector %d: entered as %s; can end as %s" % (sec, "/".join(sorted(tried, key=KINDS.index)),
                                                          "/".join(sorted(entries, key=KINDS.index))))
    done = []
    for tried, opts, (entry, out) in zip(analyses, options, pick_kinds(options, first)):
        res = tried[entry]
        if out is not None:
            res["breaks"], res["path"] = res["by_exit"][KINDS.index(out)]
        res["exit_kind"] = out
        res["options"] = {a: {b: v[0] for b, v in o.items()} for a, o in opts.items()}
        done.append(res)
    return done


def pick_kinds(options: list, first: str = "wood") -> list:
    """The kinds of ball for a whole map from each sector's options {entry kind: {exit kind: (breaks, links)}}:
    [(entry kind, exit kind or None), ...] per sector, the map's breaks then links fewest, ties to the kind first in
    KINDS. The next sector is entered as the exit kind. A sector with no option for any of its entry kinds (its goal
    not reached even with breaks) ends as None and passes the kind it was entered as on."""
    order = KINDS.index
    best = {first: ((0, 0), [])}                   # kind the next sector is entered as -> (cost so far, choices)
    for opts in options:
        nxt = {}
        for entry in sorted(best, key=order):
            (b0, l0), how = best[entry]
            for out in sorted(opts.get(entry, {}), key=order):
                nb, nl = opts[entry][out]
                cost = (b0 + nb, l0 + nl)
                if out not in nxt or cost < nxt[out][0]:
                    nxt[out] = (cost, how + [(entry, out)])
        best = nxt or {k: (c, how + [(k, None)]) for k, (c, how) in best.items()}
    return min(best.values(), key=lambda v: (v[0], [order(e) for e, _o in v[1]]))[1]


class Overrides:
    """A what-if for the region analysis: without (placement names taken as not there), state ({mechanism name:
    state at the start}: 0/1, or the state's name: gate 关/开, board 立/倒, seesaw 0 as it falls first / 1 tipped),
    lock (mechanism names whose state never changes)."""

    def __init__(self, without=(), state=None, lock=()):
        self.without, self.state, self.lock = set(without), dict(state or {}), set(lock)

    @classmethod
    def parse(cls, without=(), state=(), lock=()):
        """From the command line: state as NAME=VALUE strings."""
        st = {}
        for s in state or ():
            name, _, v = s.partition("=")
            if not v:
                raise ValueError("--state %a: write it as NAME=STATE" % s)
            st[name.strip()] = v.strip()
        return cls(without or (), st, lock or ())

    @staticmethod
    def bit(typ: str, v, labels=()) -> int:
        v = str(v).strip().lower()
        states = labels or STATES.get(typ, ())
        normalize = (lambda name: "".join(name.lower().split())) if typ == "flap" else str.lower
        names = {normalize(name): bit for bit, state in enumerate(states) for name in (state, _chinese(state))}
        aliases = {"flap": {"a": 0, "b": 1}, "seesaw": {"tipped": 1, "翻": 1, "翻过来": 1}}
        names.update(aliases.get(typ, {}))
        v = normalize(v)
        if v in ("0", "1"):
            return int(v)
        if v in names:
            return names[v]
        options = states or (("tipped",) if typ == "seesaw" else ())
        raise ValueError("a %s has no state %a (0, 1%s)" % (typ, v, "".join(", " + k for k in options)))

    def __bool__(self):
        return bool(self.without or self.state or self.lock)

    def plain(self, mechanisms=()) -> dict:
        """Report state aliases using the same labels as the mechanism report."""
        by_name = {n: m for m in mechanisms if m.get("initial") is not None
                   for n in (m["placements"] if m["type"] == "box" else (m["name"],))}
        states = {}
        for name, value in self.state.items():
            m = by_name.get(name)
            if m is not None and m["type"] not in ("cork_bridge", "float_row"):
                value = m["initial"]
            states[_ascii(name)] = _ascii(value)
        return dict(without=[_ascii(n) for n in sorted(self.without)], state=states, lock=[_ascii(n) for n in sorted(self.lock)])


def describe(key, lang="en", ride=None) -> str:
    """A link's key in words, in English for reports or Chinese for figures."""
    def tr(en, zh):
        return zh if lang == "zh" else en

    if key is None:
        return ""
    ball = tr(KINDS, BALL)
    if key[0] == "plain":
        _p, _a, _b, kind, mask, k, t, *off = key
        text = {
            "launch": ("launch", "起飞"), "drop": ("drop", "下落"), "step": ("step", "台阶"),
            "roll": ("roll", "滚"), "rail": ("rail", "钢轨"), "rail end": ("rail end", "钢轨尽头"),
            "gap": ("break (gap)", "断点(gap)"), "bridge": ("break (bridge)", "断点(bridge)"),
        }
        desc = tr(*text.get(kind, (kind, kind)))
        if kind == "roll" and t == k:
            desc = tr("roll down (one way)", "单向滚下")
        if mask:
            desc += tr(", requires bits in mask %d", "，要机关位 %d") % mask
        if off:
            if len(off) > 1:
                desc += tr(", requires flap pose A (bit mask %d)", "，要求双向挡板在 A 姿态（机关位掩码 %d）") % off[1]
                if off[0] != off[1]:
                    desc += tr(", other bits that must be 0: mask %d", "，其余需为 0 的机关位掩码 %d") % (off[0] ^ off[1])
            else:
                desc += tr(", requires standing falling rock (bit mask %d)", "，要落石未落（机关位 %d）") % off[0]
        if t != k:
            desc += tr(", enters %s transformer range, becomes %s", ("进" if kind == "roll" else "，落进") + "%s变的范围，变%s球") % (ball[t], ball[t])
        return desc
    if key[0] == "lift":
        t = key[3]
        desc = tr("paper rises on fan lift", "纸球乘风扇上升")
        if t != coarse.PAPER:
            desc += tr(", enters %s transformer range, becomes %s", "，进%s变的范围，变%s球") % (ball[t], ball[t])
        if len(key) > 4:
            desc += tr(", requires bits in mask %d set and bits in mask %d clear", "，要求掩码 %d 对应的机关位为 1，掩码 %d 对应的机关位为 0") % key[4:]
        return desc
    _m, _a, _b, typ, m, side, k, t = key
    transform = tr(", becomes %s", "，变%s球") % ball[t] if t != k else ""
    mechanism = tr(" (mechanism %d)", "（机关 %d）") % m
    if typ == coarse.ELEVATOR:
        desc = tr("rides elevator up, pushing off %d blocks", "乘升降机上去，推掉 %d 块") % coarse.elevator_blocks(side)
        if k == coarse.PAPER:
            desc += tr("; ", "；") + tr(ELEVATOR_PAPER, _chinese(ELEVATOR_PAPER))
        if ride is not None and ride["flight"]:
            desc += "; " + _elevator_flight(ride["gap"], lang=lang)
    elif typ == coarse.FLAP_OVER:
        desc = tr("flap in pose %s, crosses lane %s", "双向挡板在 %s 姿态，过 %s 路") % ("AB"[side], "AB"[side])
    elif typ == coarse.FLAP_PUSH:
        desc = (tr("%s pushes the flap from local +X and crosses lane B; lane A closes", "%s球从本地 +X 一侧推翻双向挡板并过 B 路，A 路断开") if side == 0 else
                tr("%s pushes the flap back from local -X and crosses lane A; lane B closes [DERIVED: not measured]",
                   "%s球从本地 −X 一侧推回双向挡板并过 A 路，B 路断开【推：未实测】")) % ball[k]
    elif typ == coarse.PENDULUM_BOARD:
        desc = tr("%s pushes pendulum, its other end pushes board down", "%s球推单摆，另一端推倒推板") % ball[k]
    elif typ == coarse.PENDULUM_BOX:
        desc = tr("%s pushes pendulum, its other end clears fan boxes", "%s球推单摆，另一端推开风扇上的箱子") % ball[k]
    elif typ == coarse.SWING:
        return tr("rides the swing across (wait for it, keys held for wood and paper) (swing %d)",
                  "坐秋千过去（要等它摆过来；木球、纸球要按键）（秋千 %d）") % (m + 1) + transform
    elif typ == coarse.SANDBAG:
        return tr("passes under the sandbag in a window", "趁沙袋摆开时过去") + transform
    elif typ == coarse.PENDULUM:
        return tr("%s pushes past the pendulum", "%s球推开单摆过去") % ball[k] + transform
    elif typ == coarse.CORK:
        return tr("crosses cork bridge", "过软木桥") + transform
    elif typ == coarse.FLOAT_ROW:
        n, direction = coarse.float_exit(side)
        if direction == coarse.FLOAT_THROUGH:
            return tr("%s crosses float row (%d boxes)", "%s球过箱式浮板（%d 个一列）") % (ball[k], n) + transform
        return tr("%s drops from float box %d (%s)", "%s球从箱式浮板第 %d 个（%s）落下") % (
            ball[k], n, tr(FLOAT_DIRECTION[direction], _chinese(FLOAT_DIRECTION[direction]))) + transform
    elif typ == coarse.GATE_OPERATE:
        desc = tr("%s pushes gate open", "%s球推开栅栏") % ball[k]
        if k == coarse.WOOD:
            desc += tr(" (run-up required [ASSUMED])", "（要助跑【假】）")
    elif typ == coarse.SEESAW:
        desc = tr("crosses seesaw from the low end", "从低端过跷跷板")
        if k == coarse.STONE:
            desc += tr(" (tips it)", "（把它翻过来）")
    else:
        desc = tr(*{
            coarse.GATE_THROUGH: ("through open gate", "过开着的栅栏"),
            coarse.BOARD_PUSH: ("pushes board down", "推倒推板"),
            coarse.ROCK_OVER: ("crosses the upper path on the dropped block", "从落下的路块顶面过上层"),
            coarse.ROCK_PUSH: ("stone pushes the crate out to drop the block", "石球顶开箱子，让路块落下"),
            coarse.T_OVER: ("crosses the top of the lowered T-board", "从倒下的 T 型板横头过"),
            coarse.T_PUSH: ("pushes T-board down", "推倒 T 型板"),
            coarse.T_CLEAR: ("crosses the path cleared by the upright T-board", "从 T 型板立姿让出的路面过"),
            coarse.BOARD_OVER: ("crosses lowered board", "从倒下的推板上过"),
        }[typ])
    return desc + mechanism + transform


def _ascii(value):
    """Keep external identifiers readable without emitting non-ASCII characters."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _link_text(key):
    return _Text(describe(key), describe(key, "zh"))


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


def route_steps(res):
    """Expand the selected regional links with actual directed rolling edges.

    The SCC only asserts reachability. Do not draw a straight line across
    that component or use a different ball's route through it.
    """
    from collections import deque
    sg,R,P=res['sg'],res['R'],res['places']
    if not res['path']:
        return []
    result=[]
    here=int(sg.loc[res['start_place']]); r,k,bits=res['start']

    def inside(goal):
        if here == goal:
            return
        todo=deque([here]); prev={here:None}
        while todo and goal not in prev:
            u=todo.popleft()
            for v,_,kind,ek,mask in sg.fwd[u]:
                if (kind not in ROLL or mask or ek not in (-1,k) or (u,v,kind) in sg.guards
                        or v in prev or R.of(k,v)!=r
                        or not (sg.terrain_masks.get((u,v,kind),7)&(1<<k))):
                    continue
                prev[v]=(u,kind); todo.append(v)
        if goal not in prev:
            raise ValueError('regional route endpoints are not joined by the rolling component')
        path=[]; v=goal
        while v!=here:
            u,kind=prev[v]; path.append(dict(source='coarse_roll' if kind=='roll' else kind,
                edge_type=kind, ball=KINDS[k], region=R.name(r), at=[P[u].tolist(),P[v].tolist()],
                anchors=[int(sg.glob[u]),int(sg.glob[v])]))
            v=u
        result.extend(reversed(path))

    way=summary_links(res, only_way=True)
    for (nr,nk,nb,_,key),desc in zip(res['path'][1:], way[1:]):
        u,v=res['links'][key]['at']; inside(u)
        result.append(dict(desc, from_region=R.name(r), ball=KINDS[k],
                           at=[P[u].tolist(),P[v].tolist()],anchors=[int(sg.glob[u]),int(sg.glob[v])]))
        here,r,k,bits=v,nr,nk,nb
    inside(int(sg.loc[res['goal']]))
    return result


def summary_links(res: dict, *, only_way=False) -> dict | list:
    """Step 2 as plain data: the links between the sector's regions, and the way with the fewest breaks."""
    P, R = res["places"], res["R"]
    links = []
    def launch_proof(key,e):
        if key is None or e is None:return {}
        u,v=e['at']; at=[_r1(P[u]),_r1(P[v])]
        sg=res.get('sg')
        if key[0]=='mech' and key[3]==coarse.GATE_THROUGH:
            row=getattr(sg,'gate_route_evidence',{}).get((u,v,key[4],key[6]))
            kind='launch' if row is not None else describe(key)
        elif key[0]!='plain':
            return dict(source=key[0],edge_type=describe(key),at=at)
        else:
            kind=key[3]
            row=getattr(sg,'route_evidence',{}).get((u,v,key[5])) if kind=='launch' else None
        if kind in BREAK:
            return dict(source='missing',edge_type=kind,at=at,missing_witness=True,
                        missing_from=at[0],missing_to=at[1])
        if row is not None:
            status=row.get('witness_status','legacy_unclassified')
            return dict(source=status,edge_type=row.get('route_type',kind),at=at,witness_status=status,
                        reference=row.get('retained_reference'),support_patches=row.get('support_patches'),
                        witness_at=[_r1(row['launch_position']),_r1(row['landing_position'])] if 'launch_position' in row else None)
        return dict(source='coarse_roll' if kind=='roll' else kind,edge_type=kind,at=at)
    way=[dict(region=R.name(r), kind=KINDS[k], mechanism_bits=b, by=what,
              link=describe(key, ride=res['links'].get(key, {}).get('elevator')),
              **launch_proof(key,res['links'].get(key))) for r,k,b,what,key in res['path']]
    if only_way:
        return way
    for key, e in sorted(res["links"].items(), key=lambda kv: (kv[0][1], kv[0][2], str(kv[0]))):
        ride = e.get("elevator")
        desc = describe(key, ride=ride)
        if ride and not ride["flight"]:
            flights = [r["gap"] for r in e["elevator_exits"].values() if r["flight"]]
            if flights:
                desc += "; alternate exits " + _elevator_flight(min(flights), upper=max(flights))
        links.append(dict(**{"from": R.name(key[1]), "to": R.name(key[2])}, what=desc, cost=round(e["cost"], 1), count=e["count"], **{k:v for k,v in launch_proof(key,e).items() if k!="at"},
                          at=[[round(float(x), 2) for x in P[e["at"][0]]], [round(float(x), 2) for x in P[e["at"][1]]]],
                          is_break=bool(key[0] == "plain" and key[3] in BREAK), big=bool(min(res["regions"][key[1]]["places"], res["regions"][key[2]]["places"]) >= BIG)))
    return dict(sector=res["sector"], entry_kind=res["entry_kind"], exit_kind=res.get("exit_kind"), start_region=R.name(res["start"][0]), goal_regions=[R.name(r) for r in res["goal_regions"]],
                fewest_breaks=res["breaks"], breaks_by_entry_and_exit_kind=res.get("options"),
                way=way,
                coarse_plan=[dict(region=R.name(p["region"]) if p["region"] >= 0 else None, kind=p["kind"], mechanism_bits=p["bits"], by=p["how"], at=_r1(p["at"]))
                             for p in res["plan"]],
                links=[{k: v for k, v in l.items() if k not in ("is_break", "big")} for l in links if not l["is_break"]],
                breaks=[{k: v for k, v in l.items() if k not in ("is_break", "big")} for l in links if l["is_break"] and l["big"]],
                breaks_all=sum(1 for l in links if l["is_break"]),
                note=("mechanism_bits: bit m is mechanism m's state (gate open, push board down, seesaw low at its other end, rock dropped, T board down"
                      + (", elevator used" if any(m["type"] == "elevator" for m in res["mechs"]) else "")
                      + (", flap in pose B" if any(m["type"] == "flap" for m in res["mechs"]) else "") + ")") if res["mechs"] else "")


def run(mesh, info, g, name: str, out, png=None, log=print, overrides=None, tag: str | None = None) -> list:
    """The two steps for every sector of a map: step 1 the regions, step 2 the links between them and the way with
    the fewest breaks. Writes <name>-regions.json and <name>-links.json into `out` (and pictures of each step for
    each sector into `png`). The kinds of ball are chosen for the whole map (`choose_kinds`): a sector is entered as
    the kind the way through the one before ended as, and which kind each sector ends as is chosen so that the map's
    breaks, then its links, are fewest (the first sector is entered as a wood ball: ASSUMED, the map file does not
    say). Returns the sectors' analyses."""
    import json
    from pathlib import Path
    ov = overrides or Overrides()
    if ov:
        tag = tag or "what-if"
    stem = name + ("-" + tag if tag else "")
    done = choose_kinds(mesh, info, g, ov)
    for res in done:
        mechanisms(res)
    reported_overrides = ov.plain(m for res in done for m in res["mechanisms"]) if ov else None
    if ov:
        log("what-if: %s" % reported_overrides)
        known = {p["name"] for p in info.get("placements", ())}
        row_names = set().union(*(r["sg"].float_row_names for r in done))
        for nm in (ov.without - row_names) | set(ov.state) | ov.lock:
            if nm not in known:
                log("   (no placement is called %s)" % _ascii(nm))
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
        log("   sector %d (enters as %s, ends as %s; breaks by entry and exit kind %s): %d links, %d breaks between regions; %s" % (
            res["sector"], res["entry_kind"], res.get("exit_kind"), res.get("options"), n_links, len(res["links"]) - n_links,
            "no way to the goal even with breaks" if res["breaks"] is None else
            "the goal is reached with %d break%s, over %d links" % (res["breaks"], "" if res["breaks"] == 1 else "s", len(res["path"]) - 1)))
        for r, k, b, what, key in res["path"]:
            log("      %-6s bits %d  %s" % (R.name(r), b, describe(key, ride=res["links"].get(key, {}).get("elevator")) or what))
    log("step 3, what the mechanisms do to it:")
    for res in done:
        if not res["mechanisms"]:
            log("   sector %d: no mechanism the coarse level keeps" % res["sector"])
            continue
        log("   sector %d:" % res["sector"])
        for m in res["mechanisms"]:
            if m["type"] in ("swing", "sandbag"):
                for row in _timed_notes(m):
                    log("      " + row)
                continue
            if m["type"] == "elevator":
                for row in _elevator_notes(m):
                    log("      " + row)
                continue
            if m["type"] == "pendulum":
                for row in _pendulum_notes(m):
                    log("      " + row)
                continue
            if m["type"] != "float_row":
                log("      %s (%s): %s; starts %s; held so: %s -> %s" % (m["name"], m["type"], " / ".join(m["states"]), m["initial"],
                                                                 "no way" if m["held_breaks"] is None else "%d breaks" % m["held_breaks"], m["need"]))
            if m["type"] == "box":
                log("         " + m["switch_note"])
                log("         " + _stack_note(m))
            for row in m.get("switched", ()):
                if "pendulum" in row:
                    log("      " + _remote_note(row))
            if m["type"] == "float_row":
                log("      %s (box float row): %s Way: %s; without the row: %s." % (
                    m["name"], m["switch_note"], "used" if m["on_way"] else "not used",
                    NEED["unreachable"] if m["without_breaks"] is None else "%d breaks" % m["without_breaks"]))
                for e in m["exits"]:
                    log("         " + _float_exit_note(e))
            elif m["type"] in ("falling_rock", "t_board", "flap"):
                log("         " + m["switch_note"])
                if m["type"] == "flap":
                    for state, rows in m["lets_through"].items():
                        log("         %s actual links: %s" % (state, "; ".join("%s -> %s: %s" % (r["from"], r["to"], r["balls"]) for r in rows) or "none"))
                    for row in m["switched"]:
                        log("         %s -> %s: %s switches it from %s and crosses lane %s to %s; e.g. push at %s (%d points)" % (
                            row["from_state"], row["to_state"], row["balls"], row["from"], row["lane"], row["to"],
                            row["at"][0], len(row["at"])))
        for c in res["chain"]:
            log("      step %d: %s (%s)%s" % (c["step"], c["did"], c["by"], " for step %d: %s" % (c["for"]["step"], c["for"]["link"]) if c["for"] else ", not used after"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    head = dict(map=name, generator="ballistics", what="", note="the coarse level only: nothing is rolled or searched", overrides=reported_overrides)
    json.dump(dict(head, what="step 1: the regions, the places a ball of one kind reaches from each other by rolling", sectors=[summary_regions(r) for r in done]),
              open(out / (stem + "-regions.json"), "w", encoding="utf8"), ensure_ascii=True, indent=1)
    json.dump(dict(head, what="step 2: the links between the regions, and the way with the fewest breaks", sectors=[summary_links(r) for r in done]),
              open(out / (stem + "-links.json"), "w", encoding="utf8"), ensure_ascii=True, indent=1)
    json.dump(dict(head, what="step 3: the mechanisms as simple state machines over the regions", sectors=[summary_mechanisms(r) for r in done]),
              open(out / (stem + "-mechanisms.json"), "w", encoding="utf8"), ensure_ascii=True, indent=1)
    if png is not None:
        png = Path(png)
        png.mkdir(parents=True, exist_ok=True)
        for res in done:
            s = res["sector"]
            k = KINDS.index(res["entry_kind"])
            mine = [d for d in res["regions"].values() if d["kind"] == res["entry_kind"]]
            draw_regions(mesh, res, png / ("%s-regions-s%d.png" % (stem, s)),
                         "第 %d 小节，第一步：%s球只靠滚动互相到得了的区域（%d 个，%d 个点以上的 %d 个）；圈是变球器的范围" % (
                             s, BALL[k], len(mine), BIG, sum(1 for d in mine if d["places"] >= BIG)))
            draw(mesh, res, png / ("%s-links-s%d.png" % (stem, s)),
                 "第 %d 小节，第二步：区域之间的连接；%s" % (s, "到不了目标" if res["breaks"] is None else "到目标最少要跨 %d 个断点" % res["breaks"]))
            if res["mechanisms"]:
                draw_mechanisms(mesh, res, png / ("%s-mechanisms-s%d.png" % (stem, s)),
                                "第 %d 小节，第三步：机关对通断的影响" % s)
    return done


# ---------------- step 3: what the mechanisms do to the connectivity
STATES = {
    "elevator": (_Text("loaded", "满载"), _Text("used", "用过")),
    "gate": (_Text("closed", "关"), _Text("open", "开")),
    "board": (_Text("up", "立"), _Text("down", "倒")),
    "box": (_Text("present", "在"), _Text("cleared", "推开")),
    "cork_bridge": (_Text("whole", "完好"), _Text("broken", "断开")),
    "falling_rock": (_Text("standing", "未落"), _Text("dropped", "已落")),
    "flap": (_Text("pose A", "A 姿态"), _Text("pose B", "B 姿态")),
}
STATES["t_board"] = STATES["board"]
NO_STATE_BIT = _Text("no state bit", "无状态位")
NEED = {
    "unreachable": _Text("goal not reached", "到不了目标"),
    "needed": _Text("needed", "必需"),
    "bypassable": _Text("used, can be bypassed", "用到，可以绕开"),
    "unused": _Text("not used by this way", "这条走法没用到"),
}
SWITCH_NOTES = {
    "elevator": _Text.format(
        "Loaded doorway only. Wood pushes all eight blocks [MEASURED]; stone can push [DERIVED]; %s. "
        "Riding sets used until death; a used elevator cannot be entered. Equilibrium heights [DERIVED], "
        "speed allowance and geometry reach [ASSUMED]. Without it, the platform is gone [READ].",
        "仅满载时从门洞进。木球能推掉八块【测】；石球能推【推】；%s。乘一次即用过，不能再进，只有死亡复位。"
        "平衡高度【推】，带速度容差及粗层几何范围【假】。拿掉时平台不在【读】。", ELEVATOR_PAPER),
    "board": _Text("Wood and stone push it down; it stays down.", "木球、石球推倒；不会再立起来"),
    "seesaw": _Text("Stone tips it when crossing from the low end; wood and paper cross without tipping it.",
                    "石球从低端过板时翻到另一头；木球、纸球过板不翻"),
    "falling_rock": _Text("Stone pushes the crate out along local Z on the lower path (requires a run-up and sustained push; not checked at the coarse level); the upper path opens and the lower path closes; resets only on death.",
                          "石球在下层沿本地 Z 顶开箱子（要助跑并持续顶住，粗层不验）；上层接通、下层封住；只有死亡复位"),
    "t_board": _Text("Push from local +X at the foot. Stone can push; wood needs a run-up of about 12 [ASSUMED: run-up available at the coarse level]; paper cannot push. The crossbar opens the other path. Resets only on death.",
                      "从板脚 +X 一侧推；石球可推，木球要助跑约 12【假：粗层假定助跑可用】，纸球不能推；另一路经横头接通；只有死亡复位"),
    "flap": _Text("In pose A only lane A is open both ways [MEASURED]; in pose B only lane B is open both ways [ASSUMED]; each open lane allows all three ball types. "
                  "A -> B: push toward -X from local +X on lane B and cross it while the flap moves; lane A closes [MEASURED]. "
                  "B -> A: push toward +X from local -X on lane A and cross it while the flap moves; lane B closes [DERIVED: not measured]. "
                  "Stone can push; wood needs a run-up [ASSUMED: not checked at the coarse level]; paper cannot push. Death resets it to pose A.",
                  "A 姿态仅 A 路两向通【测】；B 姿态仅 B 路两向通【假：两向】，通路均放行三种球。"
                  "A → B：从 B 路 +X 一侧朝 −X 推，翻的同时过 B 路，A 路断开【测】；"
                  "B → A：从 A 路 −X 一侧朝 +X 推，翻的同时过 A 路，B 路断开【推：未实测】。"
                  "石球可推，木球要助跑【假：粗层不验】，纸球不能推；死亡复位回 A。"),
    "cork_bridge": _Text("Wood and paper can cross. Stone breaks the bridge on contact and falls with it. The bridge resets only after a ball is lost in this sector, so a live ball never encounters a broken bridge.",
                         "木球、纸球能过，石球一上去就压断、随桥掉下去（断了的桥要在本小节丢一次球才复原），所以活着的球不会遇到断桥"),
    "pendulum": _Text("Wood can push [MEASURED]; stone and paper can push past [DERIVED]. No speed threshold. Only wood and stone can operate linked targets [OWNER][DERIVED]; no swing or force is simulated.",
                      "木球可推【测】；石、纸球可推过【推】，无速度门槛。仅木球、石球可触发联动【项目主人】【推】；不模拟摆动和力。"),
    "box": _Text("The stack blocks this fan's lift until cleared [OWNER][DERIVED].", "箱子推开前封住该风扇的上升边【项目主人】【推】。"),
}
FLOAT_NOTE = (
    "%d boxes in a row. From either end, paper can reach the full row; wood can reach the first %d boxes [ASSUMED: enough run-up]. "
    "Stone is modeled as reaching 0 boxes (the catalogue text says at most 1; the number table gives no limit). Exits from a side or end onto the road are one-way [ASSUMED].",
    "%d 个一列；从任一入口数，纸球可到全列，木球可到前 %d 个（【假】有足够助跑）；"
    "石球按 0 个（目录正文说最多 1 个，数值表未给上限）。侧面或末端到路面的连接是单向的【假】。",
)
NEAR_BREAK = 8.0               # working value: a mechanism this near a break's line (and within 10 in height) is listed by it


def _mech_links(res, typ_m):
    """The links of one mechanism, by index within its stateful or stateless group."""
    out = []
    for key in res["links"]:
        if key[0] != "mech":
            continue
        group = {coarse.CORK: "cork", coarse.FLOAT_ROW: "float", coarse.PENDULUM: "pendulum",
                 coarse.SWING: "timed", coarse.SANDBAG: "timed"}.get(key[3], "m")
        if (group, key[4]) == typ_m:
            out.append(key)
    return out


def _box_lift_links(res, m):
    return [k for k in res["links"] if k[0] == "lift" and len(k) > 4 and k[4] & (1 << m)]


def _flap_deck_links(res, m):
    sg = res["sg"]
    # Cached deck-dependent edges remain plain (including their original rail/launch type).
    deck = sg.mechs[m]["deck_edges"]
    out = {}
    for key, e in res["links"].items():
        if key[0] == "plain" and key[3] not in BREAK:
            edge = (int(sg.glob[e["at"][0]]), int(sg.glob[e["at"][1]]), key[3])
            if edge in deck:
                out[key] = deck[edge]
    return out


def _needs(key, k: int, bits: int):
    """What a link needs of the ball's kind and the mechanism bits, as [("kind", kinds allowed) | ("bit", m, value)]."""
    if key[0] == "plain":
        mask = key[4]
        off = key[7] if len(key) > 7 else 0
        return [("bit", m, value) for value, bits in ((1, mask), (0, off)) for m in range(bits.bit_length()) if (bits >> m) & 1]
    if key[0] == "lift":
        on, off = key[4:] if len(key) > 4 else (0, 0)
        return [("kind", {coarse.PAPER})] + [("bit", m, value) for value, mask in ((1, on), (0, off))
                                            for m in range(mask.bit_length()) if (mask >> m) & 1]
    _m, _a, _b, typ, m, side, kk, _t = key
    if typ in (coarse.GATE_THROUGH, *coarse.STATE_OVER):
        return [("bit", m, 1)]
    if typ in coarse.OPERATE:
        wood = typ == coarse.T_PUSH or (typ == coarse.GATE_OPERATE and side == coarse.GATE_WOOD)
        return [("kind", {coarse.WOOD, coarse.STONE} if wood else {coarse.STONE}), ("bit", m, 0)]
    if typ in coarse.PENDULUM_OPERATE:
        return [("kind", {coarse.WOOD, coarse.STONE}), ("bit", m, 0)]
    if typ == coarse.FLAP_PUSH:
        return [("kind", {coarse.WOOD, coarse.STONE}), ("bit", m, side)]
    if typ == coarse.FLAP_OVER:
        return [("bit", m, side)]
    if typ == coarse.BOARD_PUSH:
        return [("kind", {coarse.WOOD, coarse.STONE})]
    if typ == coarse.BOARD_OVER:
        return [("bit", m, 1)]
    if typ == coarse.SEESAW:
        return [("bit", m, side)]
    if typ == coarse.CORK:
        return [("kind", {coarse.WOOD, coarse.PAPER})]
    if typ == coarse.ELEVATOR:
        target = side >> coarse.ELEVATOR_SHIFT
        return [("kind", {k0 for k0 in range(3) if side & (1 << k0)}), ("bit", m, 0)] + (
            [("bit", target.bit_length() - 1, 0)] if target else [])
    if typ == coarse.FLOAT_ROW:
        return [("kind", {k for k in range(len(KINDS)) if side & (1 << k)})]
    return []


def _elevator_flight(gap, lang="en", upper=None):
    span = "%.2f" % gap if upper is None or round(gap, 2) == round(upper, 2) else "%.2f-%.2f" % (gap, upper)
    return ("must fly a %s gap [DERIVED, not measured]" if lang == "en" else "要飞过 %s 的空档【推，未实测】") % span


def _elevator_notes(m, details=True, lang="en"):
    def tr(en, zh):
        return en if lang == "en" else zh

    wb = m["without_breaks"]
    yield tr("%s (elevator): %s / %s; starts %s; way: %s; without platform: %s.",
             "%s（升降机）：%s / %s；开局%s；走法%s；拿掉平台：%s。") % (
        m["name"], *m["states"], m["initial"], tr("used", "使用") if m["on_way"] else tr("not used", "未用"),
        tr("no way", "到不了目标") if wb is None else tr("%d breaks", "%d 个断点") % wb)
    yield m["switch_note"]
    if not details:
        return
    entries = sorted({r for e in m["entries"] for r in e["regions"]})
    yield tr("doorway entries: %s", "从门洞进：%s") % (", ".join(entries) or tr("none", "无"))
    grouped = {}
    for e in m["exits"]:
        key = (e["from"], e["to"], e["ball"], e["blocks"], e["platform_top"], tuple(e["walls"]),
               e["flight"], e["with_speed"], e["target_elevator"] or "")
        grouped.setdefault(key, []).append(e)
    if not grouped:
        yield tr("no matched exits", "无高度相合的出口")
    for (a, b, ball, n, height, walls, flight, speed, target), options in sorted(grouped.items()):
        g0, g1 = min(e["gap"] for e in options), max(e["gap"] for e in options)
        y0, y1 = min(e["exit_top"] for e in options), max(e["exit_top"] for e in options)
        ball_label = ball if lang == "en" else BALL[KINDS.index(ball)]
        row = tr("%s -> %s: %s, %d blocks, platform top %.2f, exit top %s; clear %s",
                 "%s → %s：%s球，推掉 %d 块，平台顶 %.2f，出口面 %s；推出 %s") % (
            a, b, ball_label, n, height, "%.2f" % y0 if round(y0, 2) == round(y1, 2) else "%.2f-%.2f" % (y0, y1),
            ", ".join(walls) or tr("no wall", "没有挡墙"))
        if ball == "paper":
            row += tr("; ", "；") + tr(ELEVATOR_PAPER, _chinese(ELEVATOR_PAPER))
        if target:
            row += tr("; lands on loaded %s", "；落到满载的 %s") % target
        if speed:
            row += tr("; needs speed [ASSUMED]", "；要带速度【假】")
        if flight:
            row += "; " + _elevator_flight(g0, lang=lang, upper=g1)
        if any(e["on_way"] for e in options):
            row += tr("; on the way", "；走法使用")
        yield row


def _float_exit_note(e):
    return "%s -> box %d (%s) -> %s: %s" % (e["from"], e["box"], e["direction"], e["to"], e["balls"])


def _pendulum_notes(m, details=True, lang="en"):
    def tr(en, zh):
        return zh if lang == "zh" else en

    m = _chinese(m) if lang == "zh" else m
    arrow = tr("%s -> %s", "%s → %s")
    crossing = tr("%s -> %s (%s)", "%s → %s（%s球）")
    sep = tr("; ", "；")
    none = tr("none", "无")
    yield tr("%s (pendulum): %s; way: %s; without the rod: %s. %s", "%s（单摆）：%s；走法%s；拿掉杆：%s。%s") % (
        m["name"], tr(NO_STATE_BIT, _chinese(NO_STATE_BIT)), tr("used", "用到") if m["on_way"] else tr("not used", "没用到"),
        tr(NEED["unreachable"], _chinese(NEED["unreachable"])) if m["without_breaks"] is None else
        tr("%d breaks", "%d 个断点") % m["without_breaks"], m["switch_note"])
    if details:
        pairs = {}
        for e in m["crossings"]:
            pairs.setdefault((e["from"], e["to"]), set()).add(tr(KINDS, BALL).index(e["balls"]))
        yield tr("  Push-through links: ", "  可推过：") + (sep.join(
            crossing % (a, b, tr(", ", "").join(tr(KINDS, BALL)[k] for k in sorted(kinds)))
            for (a, b), kinds in sorted(pairs.items())) if pairs else tr("no supported sections to connect", "无可接的有支撑路段"))
        for e in m["crossings"]:
            if e["on_way"]:
                yield tr("  Link used: %s, %s.", "  走法用边：%s，%s。") % (
                    crossing % (e["from"], e["to"], e["balls"]), arrow % tuple(e["at"]))
    if m["stopper_deg"] is not None:
        yield tr("  Stop at %+.3f deg [DERIVED] blocks %d directed point pairs across the rod [ASSUMED: clearance under one ball diameter] (see details).",
                 "  限位 %+.3f°【推】挡掉 %d 个有向跨杆点对【假：让距不足一球直径】（见明细）。") % (m["stopper_deg"], len(m["blocked"]))
        if details and m["blocked"]:
            groups = sorted({(e["from"], e["to"]) for e in m["blocked"]})
            yield tr("  Blocked directions (wood region IDs; stone and paper are also blocked): ",
                     "  被挡方向（木球区域编号，同样限制石、纸球）：") + sep.join(arrow % pair for pair in groups)
    if not m["linked"]:
        yield tr("  Actual linked operations: none", "  实际联动：无")
    for row in m["linked"]:
        yield _remote_note(row, lang)
    yield tr("  Possible linked mechanisms [ASSUMED: unverified]: ", "  联动候选【假，未验证】：") + (sep.join(
        c["name"] + (tr(" (near a break on the way: %s; may be related)", "（靠近走法断点 %s，可能相关）") %
        ", ".join(arrow % (b["from"], b["to"]) for b in c["near_route_breaks"]) if c["near_route_breaks"] else "")
        for c in m["candidates"]) or none)
    if m["hole_warning"]:
        yield tr("  Hole warning: support is missing near the pivot. Pushing the wooden target or being hit by the returning rod may drop the ball into the hole [DERIVED]; links are unchanged.",
                 "  空洞提醒：轴周有缺支撑的位置，推木靶或被回摆的杆撞到可能掉进洞【推】；不改连通。")


def _near_break(q, a, b):
    d = b - a
    t = float(np.clip(((q - a) @ d) / max(float(d @ d), 1e-9), 0.0, 1.0))
    c = a + t * d
    return np.hypot(*(q - c)[[0, 2]]) <= NEAR_BREAK and abs(q[1] - c[1]) <= 10.0


def _remote_switches(res, keys):
    """Report concrete operation links and all their supported push sites."""
    R, P, sg = res["R"], res["places"], res["sg"]
    used = {key for *_x, key in res["path"]}
    groups = {}
    for key in sorted(keys):
        _, a, b, typ, m, j, k, _ = key
        row = groups.setdefault((a, b, typ, m, j), (set(), set(), set()))
        row[0].add(k)
        row[1].update(res["links"][key]["sites"])
        row[2].add(key)
    for (a, b, typ, m, j), (balls, sites, links) in groups.items():
        rod, target = sg.pendulums[j]["name"], sg.mechs[m]["name"].split("/")[0]
        effect = (_Text.format("remotely pushed down by pendulum %s", "由单摆 %s 远程推倒", rod)
                  if typ == coarse.PENDULUM_BOARD else
                  _Text.format("remotely cleared by pendulum %s", "由单摆 %s 远程推开", rod))
        yield dict(**{"from": R._label(a), "to": R._label(b)}, balls=_ball_list(balls),
                   at=[_r1(P[u]) for u in sorted(sites)], pendulum=rod, target=target,
                   effect=effect, basis=["OWNER", "DERIVED", "ASSUMED"], on_way=bool(links & used),
                   locked=bool(res.get("lock", 0) & (1 << m)))


def _stack_note(m, lang="en"):
    return ("风扇 %s；箱子：%s" % (m["fan"], "、".join(m["placements"])) if lang == "zh" else
            "Fan %s; stack: %s" % (m["fan"], ", ".join(m["placements"])))


def _remote_note(row, lang="en"):
    if lang == "zh":
        r = _chinese(row)
        return "  %s：%s，%s球从 %s 推（例如 %s，共 %d 处）【项目主人】【推】【假：几何】%s" % (
            r["target"], r["effect"], r["balls"], r["from"], r["at"][0], len(r["at"]),
            "；目标锁定" if r["locked"] else "")
    return "  %s: %s; %s can push in %s (e.g. %s; %d sites) [OWNER][DERIVED][ASSUMED: geometry]%s" % (
        row["target"], row["effect"], row["balls"], row["from"], row["at"][0], len(row["at"]),
        "; target locked" if row["locked"] else "")


def _pendulum_reports(res, used):
    sg, R, P = res["sg"], res["R"], res["places"]
    if not getattr(sg, "pendulums", ()):
        return []
    info, world, cost, ov, g = res["_placement_context"]
    placements = {p["name"]: p for p in info["placements"] if p.get("sector") == res["sector"]}
    breaks = [key for key in used if key[0] == "plain" and key[3] in BREAK]
    out = []
    for j, rod in enumerate(sg.pendulums):
        name = rod["name"]
        keys = _mech_links(res, ("pendulum", j))
        remote = [k for k in res["links"] if k[0] == "mech" and k[3] in coarse.PENDULUM_OPERATE and k[5] == j]
        on_way = bool(used.intersection(keys + remote))
        what_if = Overrides(ov.without | {name}, ov.state, ov.lock)
        alternate = scan(g, info, res["sector"], world, res["entry_kind"], cost, what_if)
        nb, _, _ = connect(links_of(alternate["sg"], alternate["R"]), alternate["start"], alternate["goal_regions"], alternate["lock"])
        candidates = []
        for c in rod["candidates"]:
            q = np.asarray(placements[c["name"]]["position"], float)
            near = [dict(**{"from": R._label(k[1]), "to": R._label(k[2])}) for k in sorted(breaks)
                    if _near_break(q, *P[list(res["links"][k]["at"])])]
            candidates.append(dict(c, near_route_breaks=near))
        crossings = [dict(**{"from": R._label(k[1]), "to": R._label(k[2])}, balls=_ball_list([k[6]]), on_way=k in used,
                          at=[_r1(P[u]) for u in res["links"][k]["at"]]) for k in sorted(keys)]
        def wood_region(u):
            r = R.of(coarse.WOOD, int(sg.loc[u]))
            return R._label(r) if r >= 0 else _Text("transformer area", "变球区")

        blocked = [dict(**{"from": wood_region(u), "to": wood_region(v)},
                        at=[_r1(sg.g.pos[u]), _r1(sg.g.pos[v])]) for u, v in rod["blocked"]]
        out.append(dict(name=name, type="pendulum", states=[], on_way=on_way, pos=rod["pos"],
                        need=_Text("used by this way", "走法用到") if on_way else
                        _Text("not used by this way", "走法没用到"), without_breaks=nb,
                        switch_note=SWITCH_NOTES["pendulum"],
                        crossings=crossings, blocked=blocked, stopper_deg=rod["limit"], sweep_radius=rod["reach"],
                        hole_warning=rod["hole"], candidates=candidates, linked=list(_remote_switches(res, remote))))
    return out


def _timed_reports(res, used):
    from .things import load

    sg, R, P = res["sg"], res["R"], res["places"]
    if not getattr(sg, "timed", ()):
        return
    info, world, cost, ov, g = res["_placement_context"]
    for j, mc in enumerate(sg.timed):
        name, typ = mc["name"], mc["type"]
        swing = typ == "swing"
        nums = load()["P_Modul_08" if swing else "P_Modul_26"].numbers
        if swing:
            timing = dict(period_game_seconds=nums["period"]["v"], windows_per_period=2,
                          end_reach=nums["end_reach_record"]["v"], amplitude_degrees=nums["amplitude_record"]["v"],
                          bank_window_steps_132=nums["bank_window_steps_132"]["v"],
                          tas_on_bucket_game_seconds=nums["tas_on_bucket_seconds"]["v"])
            note = _Text("Wait for the bucket; wood and paper must hold keys [MEASURED]. "
                         "Protection floor below [DERIVED]." if mc["safety"] else
                         "Wait for the bucket; wood and paper must hold keys [MEASURED]. Falling off is fatal [DERIVED].",
                         "等斗摆过来，木球、纸球要按键【测】。下方有保护路面【推】。" if mc["safety"] else
                         "等斗摆过来，木球、纸球要按键【测】。掉下去会死【推】。")
        else:
            timing = dict(period_game_seconds=2 * nums["half_period_seconds"]["v"], windows_per_period=2,
                          window_steps_132=nums["window_steps_per_half"]["v"],
                          startup_closed_steps_132=nums["no_window_steps"]["v"],
                          first_window_after_death_steps_132=nums["first_window_after_death"]["v"],
                          impact_ball_speed=nums["impact_ball_speed"]["v"])
            note = (_Text("Wait for a window; a hit counts as failure [DERIVED].", "要等窗口；撞上算失败【推】。") if mc["edges"] else
                    _Text("No added window crossing; a hit counts as failure [DERIVED].", "不补等窗口的横穿边；撞上算失败【推】。"))
        keys = _mech_links(res, ("timed", j))
        on_way = bool(used.intersection(keys))
        nb = res["breaks"]
        if mc["edges"] and (not swing or on_way):
            what_if = Overrides(ov.without | {name}, ov.state, ov.lock)
            other = scan(g, info, res["sector"], world, res["entry_kind"], cost, what_if)
            nb, _, _ = connect(links_of(other["sg"], other["R"]), other["start"], other["goal_regions"], other["lock"])
        crossings = [dict(**{"from": R._label(k[1]), "to": R._label(k[2])}, balls=_ball_list([k[6]]), on_way=k in used,
                          samples=res["links"][k]["count"], at=[_r1(P[u]) for u in res["links"][k]["at"]]) for k in sorted(keys)]
        yield dict(name=name, type=typ, pos=mc["pos"], states=[], on_way=on_way, without_breaks=nb,
                   timing=timing, switch_note=note, crossings=crossings,
                   need=_Text("used by this way", "用到") if on_way else _Text("not used by this way", "没用到"), safety_floor=mc["safety"] if swing else None,
                   status="connected" if crossings else ("fixed_bypass" if mc["bypass"] else "no_matching_crossing"))


def _timed_notes(m, details=True, lang="en"):
    def tr(en, zh):
        return zh if lang == "zh" else en

    swing = m["type"] == "swing"
    label = tr("swing", "秋千") if swing else tr("sandbag", "沙袋")
    wb = m["without_breaks"]
    yield tr("%s (%s): %s; way: %s; without it: %s. %s", "%s（%s）：%s；走法%s；拿掉它：%s。%s") % (
        m["name"], label, tr(NO_STATE_BIT, _chinese(NO_STATE_BIT)), tr("used", "用到") if m["on_way"] else tr("not used", "没用到"),
        tr("%d breaks", "%d 个断点") % wb if wb is not None else tr("unreachable", "到不了目标"), m["switch_note"])
    t = m["timing"]
    if swing:
        w = t["bank_window_steps_132"]
        yield tr("  %.1f game s; one window per bank; empty bucket model at lip %.1f: +X steps %d-%d, -X %d to next cycle %d at 132 fps [DERIVED]; reach %.2f-%.2f [DERIVED].",
                 "  周期 %.1f 游戏秒；两岸各一个窗口；空斗对 %.1f 岸沿的模型：132 fps 下正侧第 %d–%d 步，负侧第 %d 步到下一周第 %d 步【推】；斗端到达 %.2f–%.2f【推】。") % (
                     t["period_game_seconds"], w["lip_abs_x"], *w["plus_x"], *w["minus_x"], *t["end_reach"])
    else:
        w = t["window_steps_132"]
        if m["status"] != "connected":
            yield tr("  The timing below is the transverse centerline reference, not a window for this placement.",
                     "  下列时序仅供横扫中线参考，不是这个摆放的通过窗口。")
        impact = t["impact_ball_speed"]
        yield tr("  %.1f game s, %d windows of %d/%d steps (center/hull), center closed for first %d steps at 132 fps [DERIVED]; impact paper/wood/stone %.1f/%.1f/%.1f per physics s [DERIVED].",
                 "  周期 %.1f 游戏秒；%d 个窗口，各 %d/%d 步（中心/包围盒）；132 fps 下起步前 %d 步中线不通【推】；纸/木/石撞后 %.1f/%.1f/%.1f 每物理秒【推】。") % (
                     t["period_game_seconds"], t["windows_per_period"], w["center"], w["hull"], t["startup_closed_steps_132"],
                     impact["paper"], impact["wood"], impact["stone"])
    if not m["crossings"]:
        yield (tr("  A fixed rolling bypass is present; no added crossing.", "  有固定支撑的绕行路；不补边。") if m["status"] == "fixed_bypass" else
               tr("  No matching pair of banks [ASSUMED].", "  未找到匹配的两岸【假】。") if swing else
               tr("  No supported transverse crossing found [ASSUMED].", "  未找到有支撑的横穿连接【假】。"))
    elif details:
        for e in m["crossings"]:
            yield tr("  %s -> %s: %s (%d samples)%s [ASSUMED].", "  %s → %s：%s（%d 组采样）%s【假】。") % (
                e["from"], e["to"], e["balls"], e["samples"], tr(" (on way)", "（走法用到）") if e["on_way"] else "")
    else:
        yield tr("  %d directed region links [ASSUMED].", "  %d 条有向区域连接【假】。") % len(m["crossings"])


def _gate_note(lang="en") -> str:
    """How a gate is pushed open, from the catalogue (P_Modul_01): a stone ball pushes it all the way; a wood ball's
    push grows with how fast it hits, and it counts as opened from the least measured speed whose push reaches the
    opening of the way past it (gap_opens_from)."""
    from . import things
    n = things.load()["P_Modul_01"].numbers
    opens = float(n["gap_opens_from"]["v"])
    ok = sorted((float(v), w) for v, w in n["wood_push_by_speed"]["v"].items() if min(w) >= opens)
    wood = _Text.format(
        "at a measured impact speed of %.2f or more, wood pushes it %.2f-%.2f, past the %.1f needed to open the way (coarse level does not check run-up [ASSUMED])",
        "木球撞前 %.2f 以上实测推开 %.2f–%.2f，过了路口打开要的 %.1f，也算推得开（粗层不验助跑【假】）",
        ok[0][0], ok[0][1][0], ok[0][1][1], opens) if ok else _Text(
        "wood did not push it far enough to open the way in measurements", "木球实测推不到路口打开")
    note = _Text.format("Stone pushes it open; %s; paper cannot push; it stays open.",
                        "石球推开；%s；纸球推不动；不会再关", wood)
    return _chinese(note) if lang == "zh" else note


def _seesaw_states(keys, links, places):
    # Label each side by the mean position from which that end is boarded.
    ends = {}
    for key in keys:
        ends.setdefault(key[5], []).append(places[links[key]["at"][0]])
    return [_Text.format("low end at (%s)", "低端在 (%s) 那头",
                         ", ".join("%.0f" % x for x in np.mean(ends[side], axis=0)[[0, 2]]))
            if side in ends else _Text("low end at the other end", "低端在另一头") for side in (0, 1)]


def mechanisms(res: dict) -> dict:
    """Step 3: each mechanism of the sector as a simple state machine over the regions: its states, what each state
    lets through, where and by which ball it is switched; whether the way to the goal needs it (it is held in the
    state it starts in, every link that would switch it taken out, and the way looked for again); the order the way
    switches things in and what each switch is for; and the breaks with a mechanism of a kind not taken in yet beside
    them. Adds res["mechanisms"] and res["chain"] and returns res."""
    R, P = res["R"], res["places"]
    b0 = res["start"][2]
    base = res["breaks"]
    used = {key for *_x, key in res["path"] if key is not None}
    out = []
    items = [("m", m, d["name"], d["type"]) for m, d in enumerate(res["mechs"])] + \
            [("cork", j, name, "cork_bridge") for j, name in enumerate(res["sg"].bridges)] + \
            [("float", j, r["name"], "float_row") for j, r in enumerate(getattr(res["sg"], "floats", ()))]
    for what, m, name, typ in items:
        keys = _mech_links(res, (what, m))

        def pair_kinds(sel):
            got = {}
            for key in keys:
                if sel(key):
                    got.setdefault((R._label(key[1]), R._label(key[2])), set()).add(key[5] if key[0] == "plain" else key[6])
            return got

        def rows_of(got):
            return [dict(**{"from": a, "to": b}, balls=_ball_list(v))
                    for (a, b), v in sorted(got.items(), key=lambda item: tuple(map(_chinese, item[0])))]

        def pairs(sel):
            return rows_of(pair_kinds(sel))

        if typ == "gate":
            states = list(STATES["gate"])
            opens = {1: pairs(lambda k: k[3] == coarse.GATE_THROUGH)}
            switch = pairs(lambda k: k[3] == coarse.GATE_OPERATE)
            changes = [k for k in keys if k[3] == coarse.GATE_OPERATE]
        elif typ == "board":
            states = list(STATES["board"])
            opens = {1: pairs(lambda k: k[3] == coarse.BOARD_OVER)}
            switch = pairs(lambda k: k[3] == coarse.BOARD_PUSH)
            changes = [k for k in keys if k[3] == coarse.BOARD_PUSH]
        elif typ == "box":
            states = list(STATES[typ])
            lift_keys = _box_lift_links(res, m)
            opens = {0: [], 1: rows_of({(R._label(k[1]), R._label(k[2])): {coarse.PAPER} for k in lift_keys})}
            switch, changes = [], []
            keys += lift_keys
        elif typ in ("falling_rock", "t_board"):
            states = list(STATES[typ])
            changes = [k for k in keys if k[3] in coarse.OPERATE]
            opens = {0: [], 1: pairs(lambda k: k[3] in coarse.STATE_OVER)}
            switch = pairs(lambda k: k[3] in coarse.OPERATE)
            for row in switch:
                sites = {u for k in changes if R._label(k[1]) == row["from"] for u in res["links"][k]["sites"]}
                row["at"] = [_r1(P[u]) for u in sorted(sites)]
            if typ == "falling_rock":
                # Existing lower edges stay plain for their kind, cost, transformer and break semantics.
                guarded = [k for k in res["links"] if k[0] == "plain" and k[3] not in BREAK
                           and ((k[4] | (k[7] if len(k) > 7 else 0)) & (1 << m))]
                for value in (0, 1):
                    got = pair_kinds(lambda k: value == 1 and k[0] == "mech" and k[3] in coarse.STATE_OVER)
                    for k in guarded:
                        needed = k[4] if value else k[7] if len(k) > 7 else 0
                        if needed & (1 << m):
                            got.setdefault((R._label(k[1]), R._label(k[2])), set()).add(k[5])
                    opens[value] = rows_of(got)
                keys += guarded
        elif typ == "elevator":
            states = list(STATES[typ])
            changes = keys
            opens = {0: pairs(lambda k: True), 1: []}
            switch = opens[0]
        elif typ == "flap":
            states = list(STATES[typ])
            changes = [k for k in keys if k[3] == coarse.FLAP_PUSH]
            deck = _flap_deck_links(res, m)
            keys += list(deck)
            opens = {side: pairs(lambda k: deck.get(k) == side or
                                 (k[0] == "mech" and k[3] == coarse.FLAP_OVER and k[5] == side)) for side in (0, 1)}
            switch = []
            for side in (0, 1):
                for row in pairs(lambda k: k in changes and k[5] == side):
                    sites = {u for k in changes if k[5] == side and (R._label(k[1]), R._label(k[2])) == (row["from"], row["to"])
                             for u in res["links"][k]["sites"]}
                    row.update(from_state=states[side], to_state=states[1 - side], lane="BA"[side],
                               at=[_r1(P[u]) for u in sorted(sites)])
                    switch.append(row)
        elif typ == "seesaw":
            # which end is low is named by where the ball goes up from at that end (the coarse level's own sides,
            # from the seesaw's hinge and mass centre: nothing about which way it lies on the map)
            states = _seesaw_states(keys, res["links"], P)
            opens = {sd: pairs(lambda k, sd=sd: k[5] == sd) for sd in (0, 1)}
            switch = pairs(lambda k: k[6] == coarse.STONE)
            changes = [k for k in keys if k[6] == coarse.STONE]
        elif what == "float":
            states = [NO_STATE_BIT]
            opens = {0: pairs(lambda k: True)}
            switch = []
            changes = keys
        else:
            states = list(STATES["cork_bridge"])
            opens = {0: pairs(lambda k: True)}
            switch = []
            changes = list(keys)                       # held 'as it starts' is not the question for a bridge: take it away
        if typ in ("board", "box"):
            remote = [k for k in keys if k[0] == "mech" and k[3] in coarse.PENDULUM_OPERATE]
            switch.extend(_remote_switches(res, remote))
            changes.extend(remote)
        initial = (b0 >> m) & 1 if what == "m" else 0
        # held: the links that would switch it out (for a bridge: all of its links)
        held = {k: e for k, e in res["links"].items() if k not in set(changes)}
        hb, hpath, _f = connect(held, res["start"], res["goal_regions"], res.get("lock", 0))
        on_way = any(k in used for k in keys)
        if base is None:
            need = NEED["unreachable"]
        elif hb is None or hb > base:
            need = NEED["needed"]
        elif on_way:
            need = NEED["bypassable"]
        else:
            need = NEED["unused"]
        placed = next((p for p in res.get("placements", ()) if p["name"] == name.split("/")[0]), None)
        out.append(dict(name=name.split("/")[0], type=typ, states=states, initial=states[initial],
                        lets_through={states[k]: v for k, v in opens.items()}, switched=switch,
                        switch_note=_gate_note() if typ == "gate" else SWITCH_NOTES.get(typ),
                        held_breaks=hb, need=need, pos=placed["position"] if placed else None))
        if typ == "box":
            d = res["sg"].mechs[m]
            out[-1].update(placements=d["placements"], fan=d["fan"])
        if typ == "elevator":
            d = res["sg"].mechs[m]
            absent = {k: e for k, e in held.items() if not (
                k[0] == "mech" and k[3] == coarse.ELEVATOR and (k[5] >> coarse.ELEVATOR_SHIFT) & (1 << m))}
            wb, _path, _form = connect(absent, res["start"], res["goal_regions"], res.get("lock", 0))
            sg = res["sg"]
            entries = [(int(sg.loc[u]), range(len(KINDS))) for u in d.get("entries", ())]
            chained = {k[7] for k in res["links"] if k[0] == "mech" and k[3] == coarse.ELEVATOR
                       and (k[5] >> coarse.ELEVATOR_SHIFT) & (1 << m)}
            if chained:
                entries.append((int(sg.loc[d["node"]]), sorted(chained)))
            exits = []
            for key in sorted(keys):
                e = res["links"][key]
                for v, ride in sorted(e["elevator_exits"].items()):
                    exits.append(dict(ride, **{"from": R.name(key[1]), "to": R.name(key[2])}, ball=KINDS[key[6]],
                                      at=_r1(P[v]), on_way=key in used and v == e["at"][1]))
            out[-1].update(on_way=on_way, without_breaks=wb, pos=d["pos"], exits=exits,
                           entries=[dict(at=_r1(P[u]), regions=[R.name(r) for k in kinds
                                                               if (r := R.of(k, u)) >= 0]) for u, kinds in entries])
        if what == "float":
            row = res["sg"].floats[m]
            passes = {kind: bool(row["allowed"] & (1 << k)) for k, kind in enumerate(KINDS)}
            note = _Text.format(*FLOAT_NOTE, row["boxes"], min(row["boxes"], row["limits"][coarse.WOOD][0]))
            exits = []
            for key in sorted(keys):
                n, direction = coarse.float_exit(key[5])
                if direction != coarse.FLOAT_THROUGH:
                    exits.append(dict(**{"from": R._label(key[1]), "to": R._label(key[2])}, box=n,
                                      direction=FLOAT_DIRECTION[direction], balls=_ball_list([key[6]]),
                                      at=[_r1(P[u]) for u in res["links"][key]["at"]]))
            out[-1].update(boxes=row["boxes"], placements=row["placements"], passes=passes, on_way=on_way,
                           without_breaks=hb, pos=row["pos"], switch_note=note, exits=exits)
    out.extend(_pendulum_reports(res, used))
    out.extend(_timed_reports(res, used))
    res["mechanisms"] = out
    # the order the way switches things in, and what each switch is for
    chain = []
    path = res["path"]
    for i in range(1, len(path)):
        r, k, bits, what, key = path[i]
        _r0, k0, bits0, _w0, _k0 = path[i - 1]
        changes = []
        if k != k0:
            changes.append(("kind", k))
        for m in range(len(res["mechs"])):
            if (bits >> m) & 1 != (bits0 >> m) & 1:
                changes.append(("bit", m, (bits >> m) & 1))
        for ch in changes:
            for_ = None
            for j in range(i + 1, len(path)):
                need = _needs(path[j][4], path[j - 1][1], path[j - 1][2]) if path[j][4] is not None else []
                for nd in need:
                    if ch[0] == "kind" and nd[0] == "kind" and ch[1] in nd[1] and k0 not in nd[1]:
                        for_ = j
                    if ch[0] == "bit" and nd[0] == "bit" and nd[1] == ch[1] and nd[2] == ch[2]:
                        for_ = j
                    if for_ is not None:
                        break
                if for_ is not None:
                    break
                if ch[0] == "bit" and res["mechs"][ch[1]]["type"] == "flap" and ((path[j][2] >> ch[1]) & 1) != ch[2]:
                    break  # a later flip, not this one, would supply the state after this point
            if ch[0] == "kind":
                did = _Text("becomes %s" % KINDS[ch[1]], "变%s球" % BALL[ch[1]])
            else:
                mm = next(x for x in out if x["name"] == res["mechs"][ch[1]]["name"].split("/")[0])
                did = _Text.format('%s changes to "%s"', "%s 变成“%s”", mm["name"], mm["states"][ch[2]])
                if key is not None and key[0] == "mech" and key[3] == coarse.BOARD_PUSH:
                    did = _Text.format("%s (crosses while pushing it down)", "%s（推倒时就过去了）", did)
                elif key is not None and key[0] == "mech" and key[3] == coarse.FLAP_PUSH:
                    did = _Text.format("%s (crosses lane %s as lane %s closes)", "%s（同一步过 %s 路，%s 路断开）",
                                       did, "AB"[ch[2]], "BA"[ch[2]])
            chain.append(dict(step=i, did=did, by=_link_text(key), **{"for": None if for_ is None else dict(step=for_, link=_link_text(path[for_][4]))}))
    res["chain"] = chain
    return res


def summary_mechanisms(res: dict) -> dict:
    """Step 3 as plain data."""
    return dict(sector=res["sector"], fewest_breaks=res["breaks"], mechanisms=[{k: v for k, v in m.items() if k != "pos"} for m in res["mechanisms"]],
                chain=res["chain"])


PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (57, 59, 121), (99, 121, 57), (140, 109, 49), (132, 60, 57), (123, 65, 115)]
LINK = {"launch": (212, 122, 28), "drop": (16, 132, 120), "step": (82, 81, 78), "roll": (150, 149, 144), "lift": (62, 100, 150), "mech": (196, 60, 48),
        "transform": (148, 52, 160)}
TRAFO = {"wood": (140, 86, 40), "stone": (90, 90, 96), "paper": (230, 230, 236)}


def _link_colour(key):
    if key[0] == "plain" and key[3] in BREAK:
        return (20, 28, 40)
    if (key[6] if key[0] == "plain" else key[-1]) != (key[5] if key[0] == "plain" else key[6] if key[0] == "mech" else coarse.PAPER):
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


def _view(mesh, res, title, pts=None):
    """A top view of the sector's places, or of `pts` and 25 round them."""
    from .render import TopView
    P = res["places"] if pts is None else np.asarray(pts, float)
    m = 0.0 if pts is None else 25.0
    lo, hi = P.min(0) - m, P.max(0) + m
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
            name = res["R"].name(r, "zh")
            v.pen.rectangle((x - 2, y - 2, x + v.pen.textlength(name, font=f) + 4, y + 26), fill=SURFACE + (220,))
            v.pen.text((x, y), name, fill=INK, font=f)


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
        rows.append("%d　%s → %s　%s" % (n, R.name(key[1], "zh"), R.name(key[2], "zh"), describe(key, "zh", ride=res["links"][key].get("elevator"))))
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


def draw_mechanisms(mesh, res: dict, path, title: str = ""):
    """Step 3: the entry kind's regions faint; each mechanism at its place with its name, kind and whether the way
    needs it; what each lets through as thick lines in its own colour (dashed for the state it does not start in),
    where it is switched as rings."""
    from .render import INK, SURFACE, _font
    P = res["places"]
    mechanisms, chain = (_chinese(res[k]) for k in ("mechanisms", "chain"))
    pts = [m["pos"] for m in mechanisms if m["pos"] is not None]
    for row in getattr(res["sg"], "floats", ()):
        pts.extend(row["centers"])
    for rod in getattr(res["sg"], "pendulums", ()):
        pts.extend(rod["sweep"])
    for key in res["links"]:
        if key[0] == "mech":
            pts += [P[res["links"][key]["at"][0]], P[res["links"][key]["at"][1]]]
    v = _view(mesh, res, title, pts if pts else None)
    back = [r for r, d in res["regions"].items() if d["kind"] == res["entry_kind"] and d["places"] >= BIG]
    _fill(v, res, back, lambda r: 60)
    f = _font(22)
    rows = []
    for i, m in enumerate(mechanisms):
        c = PALETTE[i % len(PALETTE)]
        if m["type"] == "float_row":
            j, row = next((j, row) for j, row in enumerate(res["sg"].floats) if row["name"] == m["name"])
            what = ("float", j)
            for q in row["centers"]:
                x, y = v.P(q[0], q[2])
                v.pen.rectangle((x - 6, y - 6, x + 6, y + 6), outline=c + (255,), width=2)
        elif m["type"] == "pendulum":
            j, rod = next((j, rod) for j, rod in enumerate(res["sg"].pendulums) if rod["name"] == m["name"])
            what = ("pendulum", j)
            sweep = rod["sweep"]
            for j0 in range(0, len(sweep) - 1, 6):  # dashes in angle space, visible at any diagram scale
                v.halo_line(sweep[j0:j0 + 4], c, 3, ring=1)
        elif m["type"] in ("swing", "sandbag"):
            what = ("timed", next(j for j, d in enumerate(res["sg"].timed) if d["name"] == m["name"]))
        elif m["type"] == "cork_bridge":
            what = ("cork", res["sg"].bridges.index(m["name"]))
        else:
            what = ("m", next(j for j, d in enumerate(res["mechs"]) if d["name"].split("/")[0] == m["name"]))
        drawn = set()
        keys = _mech_links(res, what)
        if m["type"] == "flap":
            keys += list(_flap_deck_links(res, what[1]))
        elif m["type"] == "box":
            keys += _box_lift_links(res, what[1])
        for key in keys:
            e = res["links"][key]
            a, b = P[e["at"][0]], P[e["at"][1]]
            if key[0] == "mech" and key[3] in (*coarse.OPERATE, coarse.FLAP_PUSH, *coarse.PENDULUM_OPERATE):
                for u in e.get("sites", (e["at"][0],)):
                    a = P[u]
                    x, y = v.P(a[0], a[2])
                    v.pen.ellipse((x - 16, y - 16, x + 16, y + 16), outline=c + (255,), width=5)
                if key[3] == coarse.FLAP_PUSH and e["at"] not in drawn:
                    drawn.add(e["at"])
                    _arrow(v, P[e["at"][0]], b, c, width=4)
                continue
            if key[3] == coarse.PENDULUM:
                if e["at"] not in drawn:
                    drawn.add(e["at"])
                    _arrow(v, a, b, c, width=4)
                continue
            if key[3] == coarse.FLOAT_ROW:
                n, direction = coarse.float_exit(key[5])
                if direction != coarse.FLOAT_THROUGH:
                    exit_key = (e["at"], n, direction)
                    if exit_key not in drawn:
                        drawn.add(exit_key)
                        centers = row["centers"]
                        entry = 0 if (a - row["origin"]) @ row["frame"][0] < 0 else -1
                        q = centers[n - 1 if entry == 0 else -n]
                        v.halo_line([a, q], c, 3, ring=1)
                        _arrow(v, q, b, c, width=4)
                    continue
            starts = key[3] not in coarse.TIMED and bool(_moves(key, res["start"][2], res.get("lock", 0)))
            ends = (tuple(sorted(e["at"])), starts)
            if ends in drawn:
                continue
            drawn.add(ends)
            if starts:
                v.halo_line([a, b], c, 5, ring=1)
            else:
                # halo_line breaks a sampled path at pixel intervals; two endpoints alone draw no dashes.
                pixels = np.linalg.norm(np.subtract(v.P(a[0], a[2]), v.P(b[0], b[2])))
                points = a + np.linspace(0, 1, max(2, int(np.ceil(pixels / 2)) + 1))[:, None] * (b - a)
                v.halo_line(points, c, 5, ring=1, dash=10.0)
        if m["pos"] is not None:
            x, y = v.P(m["pos"][0], m["pos"][2])
            t = "%s %s" % (m["name"], m["need"])
            v.pen.rectangle((x + 10, y - 30, x + 14 + v.pen.textlength(t, font=f), y - 4), fill=SURFACE + (230,), outline=c + (255,))
            v.pen.text((x + 12, y - 30), t, fill=INK, font=f)
        if m["type"] == "float_row":
            rows.append("%s（箱式浮板）：%s。整列拿掉：%s。" % (m["name"], m["need"],
                        "到不了目标" if m["without_breaks"] is None else "%d 个断点" % m["without_breaks"]))
            rows.append("  " + m["switch_note"])
            if m["exits"]:
                rows.append("  %d 条离开连接（箭头为单向；区域与球种见第三步明细）" % len(m["exits"]))
            continue
        if m["type"] in ("swing", "sandbag"):
            rows.extend(_timed_notes(m, details=False, lang="zh"))
            continue
        if m["type"] == "elevator":
            rows.extend(_elevator_notes(m, details=False, lang="zh"))
            continue
        if m["type"] == "pendulum":
            rows.extend(_pendulum_notes(m, details=False, lang="zh"))
            continue
        rows.append("%s（%s）：%s；开局%s；%s。%s" % (m["name"], {"gate": "栅栏", "board": "推板", "box": "箱子", "seesaw": "跷跷板", "cork_bridge": "软木桥", "falling_rock": "落石", "t_board": "T 型板", "flap": "双向挡板"}[m["type"]],
                                                   " / ".join(m["states"]), m["initial"], m["need"], m["switch_note"]))
        if m["type"] == "box":
            rows.append("  " + _stack_note(m, lang="zh"))
    for c in chain:
        rows.append("第 %d 步 %s%s" % (c["step"], c["did"], "，为了第 %d 步：%s" % (c["for"]["step"], c["for"]["link"]) if c["for"] else ""))
    sg = res["sg"]
    v.dot(sg.g.pos[res["start_place"]], "起点", colour=(208, 59, 59))
    v.dot(sg.g.pos[res["goal"]], "目标", colour=(20, 28, 40))
    legend = [(INK, "粗线：机关放行的连接（实线：开局就放行；虚线：要先拨动它）；圈：拨动它的地方")]
    if res["sg"].timed:
        legend.append((INK, "秋千、沙袋虚线：要等窗口，不用拨动"))
    if res["sg"].pendulums:
        legend.append((INK, "大虚圈：单摆扫圈【假】；短箭头：可推过去的方向"))
    v.legend(legend)
    v.save(path)
    _list_under(path, rows, per_column=20)


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
