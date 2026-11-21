"""Shortcuts: ways onto places the coarse level never proposes, tried one by one, each giving another route.

The coarse plan is one cheapest way over the coarse level's own edges, and its links are taken only where nothing
else gets there. So a place that can hold a ball and leads on to the checkpoint, but that no edge leads onto (the
top of a wall, a guard rail), is never on a plan, however much time it would save. Such places are the *islands*
here. For each island that a ball on the route could come down on, the fine level is asked for a way from the
states the ball has on the route to the island; where it finds one, the sector is solved on from the landing, and
the result is another route for the sector, with its own time and its own tolerance.

What a route listed here is: a hypothesis. It is a trajectory of the hand-written kernel, found by a search that
stops at the first way it finds (not the quickest one); whether the game agrees is for the tools that read the
list. Every candidate is listed, also the ones that were tried and not found, came out slower, or left a break.

Nothing here is a rule of the game. The numbers that only rank and prune candidates are working choices and say so.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from . import fine
from . import physics as ph
from .coarse import STEP_UP

OWN = ("roll", "rail", "rail end", "drop", "launch")     # edges a ball takes without a search: rolling, and the falls the kernel itself ran (launch.py)
ROLLS = ("roll", "rail", "rail end", "drop")
TICKS_PER_COST = ph.PSI_HZ / 9.0       # the coarse costs are lengths at the follower's pace of 9 (launch.FOLLOW_SPEED): this turns them into ticks. Only ranks candidates
V_TOP = 22.5                           # MEASURED: the wood ball's top speed on the level with two keys held (docs/physics.md)
BOUNCE = 0.245                         # MEASURED: what is left of the speed along the normal after a landing (docs/physics.md)
HOP = 8.0                              # working value: how far a ball is taken to get across without any drop (a hop, a kerb)
LEFT_BREAK = 100 * 132                 # ranking only: a break that was left counts as 100 game seconds, so a way round one ranks first
MIN_GAIN = 66                          # working value: an island is tried only if it could save half a game second
RUN_UP = 396                           # working value: the search may also start up to 3 game seconds before the first state within reach
VISIT = 1.0                            # a carried ball is at a place when it is this near it: the places are a ball-centre sample every 1 unit (coarse.STEP)
FURTHER = 132                          # working value: a landing further on is looked for only where it is a game second nearer the end
ROUNDS = 2                             # working value: how many times a landing further on is looked for on one island
CHUNK = 8


def route_states(worlds, res) -> list:
    """The states the ball has along a sector's route, every CHUNK ticks, from running each segment's keys again.
    Each: dict(seg, n (ticks into the segment), t (ticks from the sector's start), pos, vel, omega, carried,
    reached (False after a break that was left: the solver only put the ball there), rem (ticks left to the end
    of the route; LEFT_BREAK more for every break left ahead, and if the route does not reach the goal))."""
    P = ph.Params()
    out, t, reached = [], 0, True
    for si, sg in enumerate(res["segments"]):
        if sg["type"] == "assumed":
            reached = False
            out.append(dict(seg=si, n=0, t=t, gap=True))
            continue
        pos, vel, omega = (np.array(x, float) for x in sg["start"])
        n = 0
        for a, hold in sg["tape"]:
            for _ in range(hold // CHUNK):
                _st, done, pos, vel, omega, stats, _ = worlds[sg["mode"]].rollout(ph.WOOD, P, pos, vel, omega, fine.PUSHES[a:a + 1], [CHUNK])
                n += done
                t += done
                out.append(dict(seg=si, n=n, t=t, pos=pos.copy(), vel=vel.copy(), omega=omega.copy(), carried=bool(stats[8] > 0), reached=reached,
                                follow=sg["type"] == "follow"))
    left = 0 if res["reached"] else LEFT_BREAK
    for s in reversed(out):
        if s.get("gap"):
            left += LEFT_BREAK
        else:
            s["rem"] = t - s["t"] + left
    return [s for s in out if not s.get("gap")]


def estimate(g, states) -> np.ndarray:
    """For every coarse place: about how many ticks from there to the end of the sector, going over the coarse
    level's own edges to where the route passes and on with the route. inf where no own edge leads to the route."""
    P = g.pos
    est = np.full(len(P), np.inf)
    grid: dict = {}
    for k, p in enumerate(P):
        grid.setdefault((int(p[0] // 4), int(p[1] // 4), int(p[2] // 4)), []).append(k)
    for s in states:
        if not s["carried"]:
            continue
        p = s["pos"]
        c = (int(p[0] // 4), int(p[1] // 4), int(p[2] // 4))
        near = [k for a in (-1, 0, 1) for b in (-1, 0, 1) for d in (-1, 0, 1) for k in grid.get((c[0] + a, c[1] + b, c[2] + d), ())]
        if not near:
            continue
        dist = np.linalg.norm(P[near] - p, axis=1)
        for k in np.flatnonzero(dist <= VISIT):
            est[near[k]] = min(est[near[k]], s["rem"])
    back = [[] for _ in range(len(P))]
    for u in range(len(P)):
        for v, c, k in g.adj[u]:
            if k in OWN:
                back[int(v)].append((u, c * TICKS_PER_COST))
    heap = [(float(est[k]), int(k)) for k in np.flatnonzero(np.isfinite(est))]
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > est[u]:
            continue
        for v, c in back[u]:
            if d + c < est[v]:
                est[v] = d + c
                heapq.heappush(heap, (d + c, v))
    return est


def _within_reach(s, P, v):
    """(which of the places v a ball in state s could conceivably come down on, how far across each is): not
    higher than a step, and no further across than the top speed carries it while it falls that far and bounces
    on (generous on purpose: it only keeps the search from being asked for the impossible)."""
    drop = s["pos"][1] - P[v, 1]
    across = np.hypot(P[v, 0] - s["pos"][0], P[v, 2] - s["pos"][2])
    fall = np.sqrt(2.0 * np.maximum(drop, 0.0) / ph.GRAVITY)                 # physics seconds
    return (drop >= -STEP_UP) & (across <= V_TOP * fall * (1.0 + 2.0 * BOUNCE / (1.0 - BOUNCE)) + HOP), across


def islands(g, states) -> list:
    """The islands of a sector, best first. Each: dict(places (indices of the island's places a ball on the route
    could come down on), est (about how many ticks from each of them to the end), place_gain (the most a landing
    on each could save), members and size (all its places), gain (ticks it could save at best), starts (indices into `states`
    of the route states within reach), start_gain (the most leaving from each could save), centre, lowest and
    highest place)."""
    P = g.pos
    est = estimate(g, states)
    # everything the route gets to by the coarse level's edges of any kind
    seen = np.isfinite(est) & False
    grid: dict = {}
    for k, p in enumerate(P):
        grid.setdefault((int(p[0] // 4), int(p[1] // 4), int(p[2] // 4)), []).append(k)
    todo = []
    for s in states:
        p = s["pos"]
        c = (int(p[0] // 4), int(p[1] // 4), int(p[2] // 4))
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for d in (-1, 0, 1):
                    for k in grid.get((c[0] + a, c[1] + b, c[2] + d), ()):
                        if not seen[k] and np.linalg.norm(P[k] - p) <= VISIT:
                            seen[k] = True
                            todo.append(k)
    while todo:
        u = todo.pop()
        for e in g.adj[u]:
            v = int(e[0])
            if not seen[v]:
                seen[v] = True
                todo.append(v)
    off = np.isfinite(est) & ~seen
    if not off.any():
        return []
    # islands: the places off the route's reach that roll into each other
    label = np.full(len(P), -1)
    nbr = [[] for _ in range(len(P))]
    for u in np.flatnonzero(off):
        for e in g.adj[u]:
            v = int(e[0])
            if e[2] in ROLLS and off[v]:
                nbr[u].append(v)
                nbr[v].append(int(u))
    n = 0
    for u in np.flatnonzero(off):
        if label[u] >= 0:
            continue
        label[u] = n
        stack = [int(u)]
        while stack:
            w = stack.pop()
            for v in nbr[w]:
                if label[v] < 0:
                    label[v] = n
                    stack.append(v)
        n += 1
    cand = [k for k, s in enumerate(states) if s["reached"] and s["carried"]]
    out = []
    for i in range(n):
        ids = np.flatnonzero(label == i)
        place_gain = np.full(len(ids), -np.inf)             # per place: the most a landing there could save
        starts, start_gain = [], []
        for k in cand:
            s = states[k]
            ok, across = _within_reach(s, P, ids)
            gain = s["rem"] - (across / V_TOP * ph.PSI_HZ + est[ids])
            ok &= gain >= MIN_GAIN
            if ok.any():
                place_gain[ok] = np.maximum(place_gain[ok], gain[ok])
                starts.append(k)
                start_gain.append(float(gain[ok].max()))
        if starts:
            good = np.isfinite(place_gain)
            out.append(dict(places=ids[good], est=est[ids[good]], place_gain=place_gain[good], members=ids, size=int(len(ids)), gain=float(max(start_gain)),
                            starts=starts, start_gain=start_gain, centre=P[ids[good]].mean(0), lo=P[ids].min(0), hi=P[ids].max(0)))
    out.sort(key=lambda d: -d["gain"])
    return out


# ---- trying an island, and the route it gives
def _cut(sg: dict, n: int, end) -> dict:
    """The first n ticks of a segment, ending in the state `end`."""
    chunks = [(a, CHUNK) for a, hold in sg["tape"] for _ in range(hold // CHUNK)][:n // CHUNK]
    out = dict(sg)
    out.update(tape=chunks, trace=sg["trace"][:n], end=end)
    if sg["type"] == "cross" and n < len(sg["trace"]):
        out["cut"] = True                                 # the route leaves in the middle of this found passage
    return out


def search_island(worlds, g, states, island, tries: int, seed: int = 0, least: float = MIN_GAIN, places=None):
    """Ask the fine level for a way from the route onto the island, touching nothing placed.
    least: only landings and departures that could save this many ticks are asked for. The starts are the route's
    states from RUN_UP before the first such departure to the last one.
    places: the places to arrive on, instead of the island's places that could save `least`.
    Returns (search, crossing, the route state it left from or None)."""
    if places is None:
        places = island["places"][island["place_gain"] >= least]
    from_ = [k for k, gain in zip(island["starts"], island["start_gain"]) if gain >= least]
    first, last = states[from_[0]]["t"], states[from_[-1]]["t"]
    pool = [k for k, s in enumerate(states) if s["reached"] and first - RUN_UP <= s["t"] <= last]
    if len(pool) > 120:
        pool = [pool[i] for i in np.unique(np.linspace(0, len(pool) - 1, 120).astype(int))]
    s = fine.Search(worlds["strict"], g.pos[places], seed=seed)
    ids = {}
    for k in pool:
        st = states[k]
        i = s.add_start(ph.State(st["pos"], st["vel"], st["omega"]), grounded=st["carried"])
        if i >= 0:
            ids[i] = k
    c = s.run(max_expansions=tries, max_seconds=1e9)
    return s, c, (states[ids[c.root]] if c.found else None)


def _traced(route):
    """(positions, ticks from the sector's start) along a route, every tick, up to the first break that was left."""
    pos = []
    for sg in route["segments"]:
        if sg["type"] == "assumed":
            break
        pos.append(sg["trace"][:, :3])
    P = np.concatenate(pos) if pos else np.zeros((0, 3))
    return P, np.arange(1, len(P) + 1)


def _rejoin(res, route, depart_t: int):
    """Where a route that left the standard one at tick depart_t is back on it: the first point, after it has been
    4 away, that lies within 2 of where the standard route passes later. Returns dict(pos, t (on this route),
    t_standard, saved) or None. Only the part of the standard route before a break that was left counts: after
    one its clock is not a time."""
    B, tb = _traced(res)
    A, ta = _traced(route)
    B, tb = B[depart_t:][::4], tb[depart_t:][::4]
    A, ta = A[depart_t:][::4], ta[depart_t:][::4]
    if not len(A) or not len(B):
        return None
    away = False
    for lo in range(0, len(A), 256):
        D = np.sqrt(((A[lo:lo + 256, None, :] - B[None, :, :]) ** 2).sum(2))
        for r in range(len(D)):
            if not away:
                away = bool(D[r].min() > 4.0)
                continue
            near = np.flatnonzero(D[r] <= 2.0)
            if len(near):
                return dict(pos=A[lo + r].copy(), t=int(ta[lo + r]), t_standard=int(tb[near[0]]), saved=int(tb[near[0]] - ta[lo + r]))
    return None


def _crossings(route) -> list:
    """The passages of a route that a search found (the follower's stretches are closed loop and not listed):
    dict(no, nudged {size of nudge: how many of `trials` still arrive}, trials, cut)."""
    cut = {sg["no"] for sg in route["segments"] if sg.get("cut")}
    return [dict(no=b["no"], nudged={int(k): int(v) for k, v in (b.get("nudged") or {1: b["arrives"]}).items()}, trials=int(b["trials"]), cut=b["no"] in cut)
            for b in route["breaks"] if b["status"] == "crossed"]


def tolerance(route) -> dict:
    """A route's tolerance, from its searched passages: per size of nudge, the count of its weakest passage
    (of `trials`), and which one that is. `weakest` is None when the route has no searched passage (it is all
    closed loop)."""
    cs = _crossings(route)
    if not cs:
        return dict(passages=0, trials=20, weakest=None)
    sizes = sorted(set.intersection(*(set(c["nudged"]) for c in cs)))
    weakest = {k: min(cs, key=lambda c: c["nudged"][k]) for k in sizes}
    return dict(passages=len(cs), trials=cs[0]["trials"], weakest={k: dict(no=w["no"], arrives=w["nudged"][k]) for k, w in weakest.items()})


def _route_over(mesh, info, g, worlds, sector, res, no, key, about, s, c, st, tries, seed) -> dict:
    """The route that leaves the standard one in the state st, takes the found way c onto an island, and goes on
    from the landing as the solver takes it (its own follower and searches, the breaks it meets named with a ')."""
    from . import staged

    depart = (st["pos"], st["vel"], st["omega"])
    nudged = {n: staged._robust(worlds["strict"], s, depart, c.tape, scale=n) for n in staged.NUDGES}
    tail = staged.solve_sector(mesh, info, g, sector, tries, seed, log=lambda *a: None, start=c.end_state)
    segs = [dict(sg) for sg in res["segments"][:st["seg"]]] + [_cut(res["segments"][st["seg"]], st["n"], depart)]
    used = {sg["no"] for sg in segs if sg["type"] == "cross"}
    breaks = [dict(b) for b in res["breaks"] if b["status"] == "crossed" and b["no"] in used]
    segs.append(dict(type="cross", mode="strict", no=no, start=depart, tape=c.tape, trace=c.trace, end=c.end_state))
    land = np.array(c.end)
    breaks.append(dict(key=key, kind="island", no=no, near=st["pos"].copy(), far=land, gap=float(math.hypot(land[0] - st["pos"][0], land[2] - st["pos"][2])),
                       rise=float(land[1] - st["pos"][1]), between=[], status="crossed", mode="strict", why="an island: no edge of the coarse level leads onto it",
                       attempts=[dict(mode="strict", where="the route", tries=c.expansions, seconds=c.seconds, found=True)], ticks=c.ticks, back=0.0,
                       flights=staged._flights(c.trace), through=[], arrives=nudged[1], nudged=nudged, trials=20, landed=land))
    for sg in tail["segments"]:
        segs.append(dict(sg, no=sg["no"] + "'") if "no" in sg else dict(sg))
    breaks += [dict(b, no=b["no"] + "'") for b in tail["breaks"]]
    route = dict(sector=sector, tries=tries, segments=segs, breaks=breaks, first=[], reached=tail["reached"], unresolved=[n + "'" for n in tail["unresolved"]],
                 solved=tail["solved"], ticks=int(sum(len(sg["trace"]) for sg in segs if sg["type"] != "assumed")), goal=res["goal"], start=res["start"],
                 rolling_start=res.get("rolling_start", False), end_state=tail["end_state"])
    route["flights"] = [f for sg in segs if sg["type"] == "follow" for f in staged._flights(sg["trace"], 20)]
    route["replays"] = staged.replays(worlds, route)
    saved = int(res["ticks"] - route["ticks"]) if res["solved"] and route["solved"] else None
    rejoin = _rejoin(res, route, int(st["t"]))
    if rejoin is None and saved is not None:                 # the two only meet again at the sector's goal
        rejoin = dict(pos=np.array(route["end_state"][0]), t=route["ticks"], t_standard=res["ticks"], saved=saved)
    route["shortcut"] = dict(about, depart=dict(t=int(st["t"]), pos=st["pos"], vel=st["vel"], omega=st["omega"]), land=land, ticks=int(c.ticks), tries=int(c.expansions),
                             nudged=nudged, trials=20, rejoin=rejoin, saved=saved)
    return route


def alternatives(mesh, info, g, sector: int, res: dict, tries: int = 300_000, most: int = 4, seed: int = 0, log=print) -> list:
    """Other routes for a sector: for each of the `most` best islands, a search from the standard route onto it
    and, where one is found, the sector solved on from the landing. The search stops at the first way it finds,
    which lands wherever is easiest. So it is first asked only for landings that could save at least half of what
    the island could save at best (any landing only if that finds nothing), and then for a landing further on
    (FURTHER nearer the end, by the estimate), 1 + ROUNDS searches at most; each landing found is solved on, and
    the island's route is the quickest of them that gets through. Returns one dict per island, best first: found or not, and for a found one the whole route (the
    same fields as a sector result, so it draws and replays the same way) with what it saves and what the other
    landings came to. Islands beyond `most` are listed as not tried."""
    from . import staged
    from .solve import world_for

    worlds = {m: world_for(mesh, info, m, sector) for m in staged.ORDER}
    states = route_states(worlds, res)
    isl = islands(g, states)
    log("sector %d: %d islands a ball on the route could come down on" % (sector, len(isl)))
    out = []
    for k, d in enumerate(isl):
        no = "S%d" % (k + 1)
        about = dict(no=no, centre=d["centre"], lo=d["lo"], hi=d["hi"], places=int(len(d["places"])), size=d["size"], could_save=float(d["gain"]))
        if k >= most:
            out.append(dict(about, tried=False, found=False))
            continue
        # first only what could save at least half of the island's best; any landing at all only if that finds nothing
        least, places, got, tried = max(0.5 * d["gain"], MIN_GAIN), None, [], 0
        for _round in range(1 + ROUNDS):
            s, c, st = search_island(worlds, g, states, d, tries, seed, least, places)
            tried += c.expansions
            log("   %s island at %s (%d places, could save %.1f game s), %s: %s after %d tries, %.0f s" % (
                no, np.round(d["centre"], 1), d["size"], d["gain"] / 132,
                "a landing further on" if places is not None else "landings that could save %.1f" % (least / 132) if least > MIN_GAIN else "any landing",
                "a way onto it" if c.found else "no way onto it", c.expansions, c.seconds))
            if c.found:
                got.append((s, c, st))
                here = d["est"][int(np.linalg.norm(g.pos[d["places"]] - c.end, axis=1).argmin())]
                places = d["places"][(d["est"] <= here - FURTHER) & (d["place_gain"] >= least)]
                if not len(places):
                    break
            elif not got and least > MIN_GAIN:
                least = MIN_GAIN
            else:
                break
        if not got:
            out.append(dict(about, tried=True, found=False, tries=tried, nearest=c.nearest, best_distance=c.best_distance))
            continue
        made = [_route_over(mesh, info, g, worlds, sector, res, no, ("S", k + 1), about, s, c, st, tries, seed) for s, c, st in got]
        best = min(made, key=lambda r: (not r["solved"], r["ticks"]))
        best["shortcut"]["landings"] = [dict(land=r["shortcut"]["land"], solved=r["solved"], ticks=r["ticks"], taken=r is best) for r in made]
        out.append(dict(about, tried=True, found=True, tries=tried, route=best))
    return out



def routes(res: dict) -> list:
    """The routes of a sector result, the standard one first: [(id, route, its island's entry or None)]. A sector
    result carries its other routes under "alternatives" (from `alternatives`)."""
    out = [("R1", res, None)]
    for a in res.get("alternatives", ()):
        if a["found"]:
            out.append(("R%d" % (len(out) + 1), a["route"], a))
    return out


def summary(res: dict, log=print):
    """The routes of a sector in words: what each is expected to take and how much it tolerates."""
    def tol(route):
        t = tolerance(route)
        if t["weakest"] is None:
            return "no searched passage (all closed loop)"
        return "%d searched passage%s, the weakest arrives %s of %d nudged runs" % (
            t["passages"], "" if t["passages"] == 1 else "s",
            ", ".join("%d (nudge x%d, %s)" % (w["arrives"], k, w["no"]) for k, w in t["weakest"].items()), t["trials"])

    def state(route):
        return "reaches the goal, %.1f game s" % (route["ticks"] / 132) if route["solved"] else "does NOT get through: %s left" % (", ".join(route["unresolved"]) or "the way")

    rs = routes(res)
    log("sector %d: %d route%s (hypotheses: the kernel's trajectories and times; a nudge is 0.03 in place, 0.3 in speed, 2 ticks in key timing)" % (
        res["sector"], len(rs), "" if len(rs) == 1 else "s"))
    for rid, route, a in rs:
        if a is None:
            log("   %s the standard route: %s; %s" % (rid, state(route), tol(route)))
            continue
        sc = route["shortcut"]
        log("   %s over the island %s at %s (x %.0f..%.0f, z %.0f..%.0f, height %.1f): %s%s; %s" % (
            rid, sc["no"], np.round(sc["centre"], 1), sc["lo"][0], sc["hi"][0], sc["lo"][2], sc["hi"][2], sc["centre"][1], state(route),
            "" if sc["saved"] is None else (", %.1f less than R1" % (sc["saved"] / 132) if sc["saved"] >= 0 else ", %.1f MORE than R1" % (-sc["saved"] / 132)), tol(route)))
        rj = sc["rejoin"]
        log("      leaves R1 at %s (%.1f game s in), on the island after %.1f game s (%d tries); the way onto it alone arrives %s of %d nudged runs; %s" % (
            np.round(sc["depart"]["pos"], 1), sc["depart"]["t"] / 132, sc["ticks"] / 132, sc["tries"],
            ", ".join("%d (x%d)" % (v, k) for k, v in sc["nudged"].items()), sc["trials"],
            "it does not come back onto R1 before a break R1 leaves or the end" if rj is None else
            "back on R1 at %s, %.1f game s %s" % (np.round(rj["pos"], 1), abs(rj["saved"]) / 132, "ahead" if rj["saved"] >= 0 else "BEHIND")))
        if route.get("extras"):
            from . import extras
            for line in extras.lines(route["extras"]):
                log("   " + line)
    for a in res.get("alternatives", ()):
        if a["found"]:
            continue
        log("   -- island %s at %s (%d places, could save %.1f game s): %s" % (
            a["no"], np.round(a["centre"], 1), a["size"], a["could_save"] / 132,
            "tried %d times from the route, no way onto it found (nearest %.1f away)" % (a["tries"], a["best_distance"]) if a["tried"] else "not tried (beyond the number of islands asked for)"))


def _plain(v):
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def listing(name: str, results: list) -> dict:
    """The routes of every solved sector as plain data, for other tools: per route what it is expected to take,
    its tolerance, and what is needed to run it again (each segment's start state and keys)."""
    def keys(tape):
        out = []
        for a, n in tape:
            if out and out[-1][0] == a:
                out[-1][1] += int(n)
            else:
                out.append([int(a), int(n)])
        return out

    def segments(route):
        out = []
        for sg in route["segments"]:
            if sg["type"] == "assumed":
                out.append(dict(type="left break", no=sg["no"], ball_put_at=sg["to"]))
            else:
                out.append(dict(type="follower" if sg["type"] == "follow" else "searched passage", no=sg.get("no"), ticks=len(sg["trace"]),
                                start=dict(pos=sg["start"][0], vel=sg["start"][1], omega=sg["start"][2]), keys=keys(sg["tape"])))
        return out

    sectors = []
    for res in results:
        rows = []
        for rid, route, a in routes(res):
            row = dict(id=rid, kind="standard" if a is None else "over an island", reaches_goal=bool(route["solved"]), left_breaks=list(route["unresolved"]),
                       game_seconds=route["ticks"] / 132 if route["solved"] else None, replay_agrees=bool(route["replays"]),
                       tolerance=tolerance(route), searched_passages=_crossings(route), segments=segments(route))
            if route.get("extras"):                     # the score and life balls on the way (ballistics.extras)
                row["extras"] = route["extras"]
            if a is not None:
                sc = route["shortcut"]
                row["shortcut"] = dict(island=dict(no=sc["no"], centre=sc["centre"], lo=sc["lo"], hi=sc["hi"], places=sc["size"]),
                                       leaves_standard_at=sc["depart"], lands_at=sc["land"], game_seconds_to_island=sc["ticks"] / 132, tries=sc["tries"],
                                       nudged=sc["nudged"], trials=sc["trials"], rejoins_standard=sc["rejoin"],
                                       game_seconds_saved=None if sc["saved"] is None else sc["saved"] / 132,
                                       game_seconds_ahead_at_rejoin=None if sc["rejoin"] is None else sc["rejoin"]["saved"] / 132)
            rows.append(row)
        other = [dict(island=dict(no=a["no"], centre=a["centre"], lo=a["lo"], hi=a["hi"], places=a["size"]), could_save_game_seconds=a["could_save"] / 132,
                      tried=a["tried"], tries=a.get("tries"), nearest_distance=a.get("best_distance")) for a in res.get("alternatives", ()) if not a["found"]]
        sectors.append(dict(sector=int(res["sector"]), entered="rolling" if res.get("rolling_start") else "at rest", routes=rows, islands_without_a_route=other))
    return _plain(dict(
        map=name, generator="ballistics",
        what="route hypotheses: trajectories of a hand-written physics kernel, not run in the game. Times are the kernel's. "
             "The payoff is given as time only (speedrun); points (highscore) are not computed.",
        ball="wood", ticks_per_game_second=132, pushes=fine.PUSHES, push_names=list(fine.PUSH_NAMES),
        nudge="tolerance: a searched passage is run again 20 times from its start nudged by (0.03 in place, 0.3 in speed) times the size, "
              "with every key change up to 2 ticks times the size early or late; the number is how many still arrive. "
              "A route's tolerance is that of its weakest searched passage; the follower's stretches are closed loop.",
        sectors=sectors))
