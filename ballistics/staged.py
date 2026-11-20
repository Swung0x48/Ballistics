"""Coarse first, fine only where needed.

1. The coarse level plans the way through a sector: its own edges, joined by the links it would need where it
   does not connect. Each such link is a *break* (so are the steps and flights it only proposes), listed with what
   is known about it from the map alone: the gap, the rise, what is placed in between.
2. A follower rolls the ball along the plan in the physics kernel by looking ahead from the ball's own state, its
   speed included (see `follow`). Falls and flights on the plan (launch.py tells the planner they exist) are taken
   with the speed the ball has: nothing stops before them. Where it gets across a listed break by
   itself, the break was the coarse level's own shortcoming. Where it cannot go on (a listed break or not), that
   is a break for step 3.
3. Only there, the fine level searches key tapes, starting from the states the ball had on the stretch it has just
   rolled (so a run-up from far back is available), towards any place nearer the goal than the far side of the
   break. It tries the strictest mode first and says which mode it needed.

A break counts as crossed only without touching anything placed. One that is not crossed within the stated number
of tries stays a break: the search is then run once more with the loose balls taken out, only as a diagnosis of
whether they are what stands in the way; the ball is put at rest on the far side to list what comes after, and
nothing after it counts as reached.
Everything found is a trajectory of the hand-written kernel: a candidate until the engine has run it.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import coarse, fine
from . import physics as ph
from .geometry import BALL_R
from .kernel import DEAD, OK, PLACED, TRIGGER
from .solve import world_for

BREAK_KINDS = ("gap", "bridge", "step")
ROLLS = ("roll", "rail", "rail end", "drop")
ORDER = ("strict", "free")
LOOK_TICKS = 132          # how far ahead the follower runs each key plan: one game second
SWITCH_TICKS = 24         # where a two-part key plan changes key
STEP_TICKS = 8            # how many ticks of the chosen plan the ball takes before looking again
CHUNK = STEP_TICKS
BASE = "strict"           # the follower's world, and the only one whose crossings count: nothing placed may be touched
STATUS = {DEAD: "fell out", TRIGGER: "entered a transformer's range", PLACED: "touched something placed"}


def nearest_node(g, togo, pos, reach: float = 3.0) -> int:
    d = np.sqrt(((g.pos - pos) ** 2).sum(1))
    d[~np.isfinite(togo)] = np.inf
    near = np.flatnonzero(d <= reach)
    return int(near[np.argmin(d[near])]) if len(near) else int(d.argmin())


def placed_between(mesh, a, b, width: float = 3.5) -> list:
    """Types of the placed objects that come within `width` of the straight line from a to b."""
    t = np.flatnonzero(mesh.placed)
    if not len(t):
        return []
    c = (mesh.A[t] + mesh.B[t] + mesh.C[t]) / 3.0
    ab = np.asarray(b, float) - np.asarray(a, float)
    f = np.clip(((c - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    d = np.sqrt(((c - (a + f[:, None] * ab)) ** 2).sum(1))
    return sorted({str(mesh.entities[mesh.owner[k]].get("type", "")) for k in t[d <= width]})


def describe_break(g, mesh, u: int, v: int, kind: str) -> dict:
    a, b = g.pos[u], g.pos[v]
    return dict(key=(int(u), int(v)), kind=kind, near=a.copy(), far=b.copy(), gap=float(math.hypot(b[0] - a[0], b[2] - a[2])),
                rise=float(b[1] - a[1]), between=placed_between(mesh, a, b), status="pending", attempts=[])


def _progress(tr, P, way, lo: int, hi: int):
    """How far along the plan a rollout gets: the largest way-length of a plan place (index lo..hi) that the ball
    is carried at (within 2.5), and the tick at which it first got there; (-1, 0) if none."""
    k = np.r_[np.arange(7, len(tr), 8), len(tr) - 1] if len(tr) else np.zeros(0, np.int64)
    k = k[tr[k, 7] > 0] if len(k) else k
    if not len(k):
        return -1.0, 0
    d = np.sqrt(((tr[k][:, None, :3] - P[None, lo:hi + 1, :]) ** 2).sum(2))
    ok = d <= 2.5
    if not ok.any():
        return -1.0, 0
    reach = np.where(ok, way[None, lo:hi + 1], -1.0).max(1)             # per sample: the furthest place it is at
    best = float(reach.max())
    return best, int(k[int(np.argmax(reach >= best - 1e-9))])


def choose_push(world, pos, vel, omega, P, way, best: int) -> int:
    """The follower's decision for one state: which of the nine pushes to hold next. From that state the kernel runs
    each push LOOK_TICKS ahead (and, if none gets on, each push for SWITCH_TICKS followed by each other push); the
    answer is the first push of the plan that gets furthest along the way P soonest without ending the ball. A plan
    still in the air at its end is run on without keys to see whether the ball comes down.
    way: way-length at each place of P; best: index of the furthest place reached so far."""
    par = ph.Params()
    n = len(P)
    zero = np.zeros((1, 3))
    lo, hi = max(best - 2, 0), min(best + 30, n - 1)

    def fate(a_tape):
        pushes = np.array([fine.PUSHES[a] for a, _ in a_tape])
        st, done, p2, v2, o2, s2, tr = world.rollout(ph.WOOD, par, pos, vel, omega, pushes, [m for _, m in a_tape], trace=True)
        alive = st == OK
        if alive and s2[8] == 0:
            st3, d3, p3, v3, o3, s3, tr3 = world.rollout(ph.WOOD, par, p2, v2, o2, zero, [160], trace=True)
            down = np.flatnonzero(tr3[:, 7] > 0)
            alive = bool(len(down)) and (st3 == OK or int(down[0]) < d3 - 1)
            if alive:
                tr = np.concatenate([tr, tr3[:int(down[0]) + 1]])
        reach, at = _progress(tr, P, way, lo, hi)
        return (reach - 0.02 * at) - (0.0 if alive else 1000.0), reach

    plans = [[(a, LOOK_TICKS)] for a in range(len(fine.PUSHES))]
    scored = [(fate(t), t) for t in plans]
    top = max(scored, key=lambda x: x[0][0])
    if top[0][0] < -500.0 or top[0][1] <= way[best] + 1.0:               # held keys end the ball or get nowhere: try a change of key
        plans = [[(a, SWITCH_TICKS), (b, LOOK_TICKS - SWITCH_TICKS)] for a in range(len(fine.PUSHES)) for b in range(len(fine.PUSHES)) if a != b]
        scored += [(fate(t), t) for t in plans]
        top = max(scored, key=lambda x: x[0][0])
    return top[1][0][0]


def follow(world, strict_world, start, P, max_ticks: int = 60000) -> dict:
    """Roll along the plan places P from the state `start` = (pos, vel, omega) by looking ahead: every STEP_TICKS
    ticks, from the state the ball is in (its speed included), the kernel runs a set of key plans LOOK_TICKS ahead
    (each of the nine pushes held; if none of those gets on, each push for SWITCH_TICKS and then each other push),
    and the ball takes the first STEP_TICKS ticks of the plan that gets furthest along the way soonest without
    ending the ball. Nothing is braked to a stop on purpose and no speed is aimed at: a fall or a flight on the
    plan is taken with the speed the ball has. Returns status 'goal', 'fail' (with the reason and the last plan
    place reached) or 'clear' (only when `world` is not the strict one: clear of everything placed again)."""
    pos, vel, omega = np.array(start[0], float), np.array(start[1], float), np.array(start[2], float)
    n = len(P)
    way = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] if n > 1 else np.zeros(1)
    par = ph.Params()
    best = 0
    since = ticks = 0
    tape, traces, history = [], [], []
    carried = True
    while ticks < max_ticks:
        lo, hi = max(best - 2, 0), min(best + 30, n - 1)
        d = np.sqrt(((P[lo:hi + 1] - pos) ** 2).sum(1))
        near = np.flatnonzero(d <= 2.5)
        if carried and len(near) and lo + int(near[-1]) > best:          # a place counts as reached when the ball is rolling at it
            best, since = lo + int(near[-1]), 0
        if best >= n - 2 and carried and float(np.linalg.norm(P[n - 1] - pos)) <= 2.0:
            return dict(status="goal", index=n - 1, tape=tape, trace=_stack(traces), history=history, end=(pos, vel, omega))
        history.append((ticks, best, pos.copy(), vel.copy(), omega.copy()))
        a = choose_push(world, pos, vel, omega, P, way, best)
        status, done, pos, vel, omega, st, tr = world.rollout(ph.WOOD, par, pos, vel, omega, fine.PUSHES[a:a + 1], [STEP_TICKS], trace=True)
        if status != OK:
            return dict(status="fail", reason=STATUS.get(status, "stopped"), index=best, tape=tape, trace=_stack(traces), history=history, end=None)
        carried = bool(st[8] > 0)
        ticks += done
        since += done
        tape.append((a, done))
        traces.append(tr)
        if since > 528:
            return dict(status="fail", reason="got no further", index=best, tape=tape, trace=_stack(traces), history=history, end=None)
        if world.mesh is not strict_world.mesh and strict_world.mesh.distance(pos, BALL_R + 1.0)[0] - BALL_R > 0.1:
            return dict(status="clear", index=best, tape=tape, trace=_stack(traces), history=history, end=(pos, vel, omega))
    return dict(status="fail", reason="ran out of time", index=best, tape=tape, trace=_stack(traces), history=history, end=None)


def _stack(traces):
    return np.concatenate(traces) if traces else np.zeros((0, 9))


def _flights(trace, least: int = 20) -> list:
    out, i = [], 0
    while i < len(trace):
        if trace[i, 3] != 0:
            i += 1
            continue
        j = i
        while j + 1 < len(trace) and trace[j + 1, 3] == 0:
            j += 1
        if j - i + 1 >= least:
            a, b = trace[max(i - 1, 0)], trace[min(j + 1, len(trace) - 1)]
            out.append(dict(across=float(math.hypot(b[0] - a[0], b[2] - a[2])), down=float(a[1] - b[1]), speed=float(a[6]), ticks=j - i + 1,
                            start=a[:3].copy(), end=b[:3].copy()))
        i = j + 1
    return out


def _robust(world, search, seed, tape, trials: int = 20, rng_seed: int = 1) -> int:
    """Of `trials` runs with the start nudged (0.03 in place, 0.3 in speed) and every key change up to 2 ticks early
    or late: how many still arrive."""
    rng = np.random.default_rng(rng_seed)
    P = ph.Params()
    ok = 0
    for _ in range(trials):
        pos = seed[0] + rng.normal(0, 0.03, 3) * np.array([1, 0, 1])
        vel = seed[1] + rng.normal(0, 0.3, 3)
        omega, good = seed[2], True
        st = np.zeros(10)
        for a, n in tape:
            status, _d, pos, vel, omega, st, _ = world.rollout(ph.WOOD, P, pos, vel, omega, fine.PUSHES[a:a + 1], [max(1, n + int(rng.integers(-2, 3)))])
            if status != OK:
                good = False
                break
        if good and not search.arrived(pos, vel, st):
            status, _d, pos, vel, omega, st, _ = world.rollout(ph.WOOD, P, pos, vel, omega, np.zeros((1, 3)), [33])
            good = status == OK and st[8] > 0 and search.distance(pos) <= search.reach + 2.0
        ok += bool(good)
    return ok


def solve_sector(mesh, info, g, sector: int, tries: int = 300_000, seed: int = 0, log=print) -> dict:
    """sector: 1-based. g: the coarse graph of `mesh` with the loose balls in it (solid)."""
    worlds = {m: world_for(mesh, info, m) for m in ORDER}
    strict = worlds["strict"]
    rel = getattr(g, "_relaxed", None)
    if rel is None:
        rel = g._relaxed = coarse.Relaxed(g)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    goal = g.nearest(anchors[sector], 25.0)
    togo = rel.togo(goal)
    reset = info["reset_points"][sector - 1]
    s0 = fine.rest_state(strict, reset[0], reset[2], reset[1] + 0.5)
    state = (s0.pos, s0.vel, s0.omega)
    loose = [(p["type"], np.array(p["position"], float)) for p in info["placements"] if p["type"].startswith(coarse.LOOSE)]
    coarse_back = [[] for _ in range(len(g.pos))]
    for u in range(len(g.pos)):
        for e in g.adj[u]:
            if e[2] in ROLLS:                                     # only what a ball rolls along; steps and flights are proposals
                coarse_back[e[0]].append(u)
    dropped = 0
    breaks: dict = {}
    order: list = []
    count = {"B": 0, "F": 0}

    def register(nodes, kinds):
        on = set()
        for e, k in enumerate(kinds):
            if k in BREAK_KINDS:
                key = (nodes[e], nodes[e + 1])
                on.add(key)
                if key not in breaks:
                    count["B"] += 1
                    breaks[key] = dict(describe_break(g, mesh, nodes[e], nodes[e + 1], k), no="B%d" % count["B"])
                    order.append(key)
        for key, b in breaks.items():
            if b["status"] == "pending" and key not in on:
                b["status"] = "bypassed"

    def replan(pos):
        nodes, kinds = rel.plan(nearest_node(g, togo, pos), togo)
        register(nodes, kinds)
        return nodes, kinds

    nodes, kinds = replan(state[0])
    first = [dict(breaks[k]) for k in order]                      # the list from the map alone, before anything is rolled
    log("sector %d: the coarse plan has %d places and %d breaks" % (sector, len(nodes), len(first)))
    segs: list = []
    seeds: list = []                                              # (segment index, history entry) since the last crossing
    older: list = []                                              # the same for the stretches before that
    mode_now = BASE
    for _round in range(80):
        if len(nodes) < 2:
            break
        # the way the follower sees: the plan's places, and for a launch on it the run-up and the landing of the
        # kernel run that made the edge (its start place alone may be far from the ledge)
        way, node_of = [g.pos[nodes[0]]], [0]
        for e, k in enumerate(kinds):
            if k == "launch":
                for q in g.info["launches"][(nodes[e], nodes[e + 1])]["way"][1:]:
                    way.append(np.asarray(q, np.float64))
                    node_of.append(e)
            way.append(g.pos[nodes[e + 1]])
            node_of.append(e + 1)
        r = follow(worlds[mode_now], strict, state, np.array(way))
        r["index"] = node_of[min(r["index"], len(node_of) - 1)]
        segs.append(dict(type="follow", mode=mode_now, start=state, tape=r["tape"], trace=r["trace"], end=r["end"]))
        seeds += [(len(segs) - 1, h) for h in r["history"]]
        for e, k in enumerate(kinds):
            if k in BREAK_KINDS and e + 1 <= r["index"] and breaks[(nodes[e], nodes[e + 1])]["status"] == "pending":
                breaks[(nodes[e], nodes[e + 1])]["status"] = "rolled"
        if r["status"] == "goal":
            break
        if r["status"] == "clear":
            state, mode_now = r["end"], BASE
            nodes, kinds = replan(state[0])
            continue
        k = min(r["index"], len(nodes) - 2)
        key, run = None, 0.0
        for e in range(k, len(kinds)):                            # a listed break right ahead is the one it failed at
            if kinds[e] in BREAK_KINDS or (nodes[e], nodes[e + 1]) in breaks:
                key, k_break = (nodes[e], nodes[e + 1]), e
                break
            run += float(np.linalg.norm(g.pos[nodes[e + 1]] - g.pos[nodes[e]]))
            if run > 8.0:
                break
        if key is not None and key in breaks:
            b, kt = breaks[key], k_break + 1
        else:
            kt, run = k + 1, 0.0
            while kt < len(nodes) - 1 and run < 6.0:
                run += float(np.linalg.norm(g.pos[nodes[kt + 1]] - g.pos[nodes[kt]]))
                kt += 1
            count["F"] += 1
            b = dict(describe_break(g, mesh, nodes[k], nodes[kt], "follow"), no="F%d" % count["F"])
            b["key"] = ("F", count["F"])
            breaks[b["key"]] = b
            order.append(b["key"])
        b["why"] = "%s on a '%s' edge" % (r["reason"], kinds[k])
        # where a crossing may arrive: any place as near the goal as the far side, and any place the coarse level's
        # own edges lead from to such a place (a wall top that drops to the far side is as good as the far side)
        beyond = togo <= togo[nodes[kt]] + 1e-9
        listed = isinstance(b["key"][0], (int, np.integer))       # a break of the plan, not a place where the follower failed on a rolling edge
        if listed:
            stack = list(np.flatnonzero(beyond))
            while stack:
                v = stack.pop()
                for u in coarse_back[v]:
                    if not beyond[u]:
                        beyond[u] = True
                        stack.append(u)
        if listed and beyond[nodes[k]] and dropped < 20:
            dropped += 1
            # the coarse level's own edges lead from here to beyond the break: the plan only took a shortcut link.
            # Drop that link and plan again
            u, v = b["key"]
            rel.fwd[u] = [e for e in rel.fwd[u] if e[0] != v or e[2] != b["kind"]]
            rel.back[v] = [e for e in rel.back[v] if e[0] != u] + [(u, c) for w, c, kk in rel.fwd[u] if w == v]
            togo = rel.togo(goal)
            b["status"] = "bypassed"
            state = segs[-1]["end"] if segs[-1]["end"] is not None else state
            if segs[-1]["end"] is None:                           # the stretch failed: go back to the last state it was fine in
                h = r["history"][-1] if len(r["history"]) < 20 else r["history"][-17]
                segs[-1]["tape"], segs[-1]["trace"], segs[-1]["end"] = segs[-1]["tape"][:h[0] // CHUNK], segs[-1]["trace"][:h[0]], (h[2], h[3], h[4])
                state = segs[-1]["end"]
            nodes, kinds = replan(state[0])
            continue
        targets = g.pos[beyond]
        usable = seeds[:-8] if len(seeds) > 12 else seeds         # not the last half second before it went wrong

        def thin(entries, most):
            cand = entries[::2]
            if len(cand) > most:
                pick = np.unique(np.concatenate([np.linspace(0, len(cand) - 21, most - 20).astype(int), np.arange(len(cand) - 20, len(cand))]))
                cand = [cand[q] for q in pick]
            return cand

        # first from the stretch just rolled; if that fails, also from every earlier stretch of the sector (the way
        # across may start far back, before the ball went down into a dead end)
        pools = [("this stretch", thin(usable, 80))]
        if older:
            pools.append(("the whole sector so far", thin(older, 120) + thin(usable, 80)))
        def attempt(mode, where, cand):
            s = fine.Search(worlds[mode], targets, seed=seed, places=g.pos, togo=togo)
            ids = {}
            for si, h in cand:
                idx = s.add_start(ph.State(h[2], h[3], h[4]))
                if idx >= 0:
                    ids[idx] = (si, h)
            if not ids:
                return None
            c = s.run(max_expansions=tries, max_seconds=1e9)
            b["attempts"].append(dict(mode=mode, where=where, tries=c.expansions, seconds=c.seconds, found=c.found))
            log("   %s at %s (%s): %s from %s, %s after %d tries, %.0f s" % (b["no"], np.round(b["near"], 1), b["why"], mode, where,
                                                                           "crossed" if c.found else "not crossed", c.expansions, c.seconds))
            return (mode, s, c, ids[c.root]) if c.found else None

        # A break counts as crossed only in strict mode
        found = None
        for where, cand in pools:
            found = attempt("strict", where, cand)
            if found:
                break
        if not found:
            # It stays a break. One more run with the loose balls taken out only says whether they are what stands
            # in the way: its trajectory is not used, and nothing after this break counts as reached
            b["would_pass"] = None
            for mode in ORDER[1:]:
                got = attempt(mode, pools[0][0], pools[0][1])
                if got:
                    _m, _s, c, _seed = got
                    b["would_pass"] = dict(mode=mode, tries=c.expansions,
                                           through=[ty for ty, q in loose if np.linalg.norm(c.trace[:, :3] - q, axis=1).min() < 2.0 * BALL_R])
                    break
        if found:
            mode, s, c, (si, h) = found
            for sg in segs[si + 1:]:                              # crossings made on a way that is now given up
                if sg["type"] == "cross":
                    for ob in breaks.values():
                        if ob["no"] == sg["no"]:
                            ob["status"] = "abandoned"
            segs = segs[:si + 1]
            older = [e for e in older + seeds if e[0] < si or (e[0] == si and e[1][0] < h[0])]
            n_ticks = h[0]
            segs[si]["tape"] = segs[si]["tape"][:n_ticks // CHUNK]
            segs[si]["trace"] = segs[si]["trace"][:n_ticks]
            segs[si]["end"] = (h[2], h[3], h[4])
            segs.append(dict(type="cross", mode=mode, no=b["no"], start=(h[2], h[3], h[4]), tape=c.tape, trace=c.trace, end=c.end_state))
            through = [(ty, q) for ty, q in loose if np.linalg.norm(c.trace[:, :3] - q, axis=1).min() < 2.0 * BALL_R]
            b.update(status="crossed", mode=mode, ticks=c.ticks, back=float(np.linalg.norm(h[2] - b["near"])), flights=_flights(c.trace),
                     through=[ty for ty, _q in through],
                     arrives=_robust(worlds[mode], s, (h[2], h[3], h[4]), c.tape), trials=20, landed=np.array(c.end))
            state = c.end_state
            clear = strict.mesh.distance(state[0], BALL_R + 1.0)[0] - BALL_R > 0.1
            mode_now = BASE if (mode == "strict" or clear) else mode
            seeds = []
            nodes, kinds = replan(state[0])
        else:
            b["status"] = "unresolved"
            last = segs[-1]                                       # keep the stretch up to the last state it was fine in
            if r["history"]:
                h = r["history"][-1] if len(r["history"]) < 20 else r["history"][-17]
                last["tape"], last["trace"], last["end"] = last["tape"][:h[0] // CHUNK], last["trace"][:h[0]], (h[2], h[3], h[4])
            older = older + seeds
            far = g.pos[nodes[kt]]
            segs.append(dict(type="assumed", no=b["no"], to=far.copy()))
            s1 = fine.rest_state(strict, far[0], far[2], far[1] + 1.0)
            state, mode_now, seeds = (s1.pos, s1.vel, s1.omega), BASE, []
            nodes, kinds = nodes[kt:], kinds[kt:]
    reached = bool(segs and segs[-1]["type"] == "follow" and segs[-1]["end"] is not None and np.linalg.norm(segs[-1]["end"][0] - g.pos[goal]) <= 2.5)
    unresolved = [breaks[k]["no"] for k in order if breaks[k]["status"] == "unresolved"]
    res = dict(sector=sector, tries=tries, segments=segs, breaks=[breaks[k] for k in order], first=first, reached=reached, unresolved=unresolved,
               solved=reached and not unresolved, ticks=int(sum(len(sg["trace"]) for sg in segs if sg["type"] != "assumed")),
               goal=g.pos[goal].copy(), start=np.array(s0.pos))
    res["flights"] = [f for sg in segs if sg["type"] == "follow" for f in _flights(sg["trace"], 20)]
    res["replays"] = replays(worlds, res)
    return res


def replays(worlds, res) -> bool:
    """Run every segment's keys again from its start, in its own mode: the same end states, and each segment starts
    where the one before ended (or at rest on the far side of an unresolved break)."""
    P = ph.Params()
    prev = None
    for sg in res["segments"]:
        if sg["type"] == "assumed":
            prev = None
            continue
        if prev is not None and np.abs(prev[0] - sg["start"][0]).max() > 1e-9:
            return False
        if sg["tape"]:
            pushes = np.array([fine.PUSHES[a] for a, _ in sg["tape"]])
            st, done, pos, vel, omega, s, _ = worlds[sg["mode"]].rollout(ph.WOOD, P, sg["start"][0], sg["start"][1], sg["start"][2], pushes,
                                                                        [n for _, n in sg["tape"]])
            if st != OK or sg["end"] is None or np.abs(pos - sg["end"][0]).max() > 1e-9:
                return False
        prev = sg["end"]
    return True


def summary(res, log=print):
    """The break list of a solved sector, in words."""
    n_rolled = sum(b["status"] == "rolled" for b in res["breaks"])
    log("sector %d: %s; %.1f game s of trajectory; %d breaks in all, %d of them rolled across by the plain follower; replay %s"
        % (res["sector"], "reaches the goal" if res["solved"] else "does NOT get through: %s left as break%s" % (", ".join(res["unresolved"]) or "the way", "s" if len(res["unresolved"]) > 1 else ""),
           res["ticks"] / 132, len(res["breaks"]), n_rolled, "agrees" if res["replays"] else "DISAGREES"))
    log("   the follower's stretches: %.1f game s" % (sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "follow") / 132))
    for f in res.get("flights", ()):
        log("   the follower flew from %s: takes off at %.1f, %.1f across, %.1f down, lands at %s"
            % (np.round(f["start"], 1), f["speed"], f["across"], f["down"], np.round(f["end"], 1)))
    for b in res["breaks"]:
        if b["status"] in ("rolled", "bypassed", "pending", "abandoned"):
            log("   %s %s at %s: gap %.1f, rise %+.1f%s -> %s" % (b["no"], b["kind"], np.round(b["near"], 1), b["gap"], b["rise"],
                                                             ", in between: " + ", ".join(b["between"]) if b["between"] else "",
                                                             {"rolled": "the plain follower rolled across (the coarse level's own shortcoming)",
                                                              "bypassed": "not on the way taken in the end", "pending": "not reached",
                                                              "abandoned": "crossed, on a way that was given up further on"}[b["status"]]))
            continue
        head = "   %s %s at %s: gap %.1f, rise %+.1f%s" % (b["no"], b["kind"], np.round(b["near"], 1), b["gap"], b["rise"],
                                                         ", in between: " + ", ".join(b["between"]) if b["between"] else "")
        if b.get("why"):
            head += " (follower: %s)" % b["why"]
        log(head)
        if b["status"] == "crossed":
            fl = "; ".join("flight %.1f across, %.1f down, take-off %.1f" % (f["across"], f["down"], f["speed"]) for f in b["flights"]) or "no flight"
            log("      crossed, %.1f game s, leaving %.0f back; %s; %d of %d nudged runs still arrive"
                % (b["ticks"] / 132, b["back"], fl, b["arrives"], b["trials"]))
        else:
            log("      LEFT AS A BREAK: not crossed in strict mode (" + "; ".join("from %s, %d tries" % (a["where"], a["tries"]) for a in b["attempts"] if a["mode"] == "strict") + ")")
            w = b.get("would_pass")
            if w:
                log("      diagnosis only: with the loose balls taken out it passes (%d tries)%s" % (w["tries"], ", through " + ", ".join(w["through"]) if w["through"] else ""))
            else:
                log("      diagnosis: not crossed with the loose balls taken out either (" + "; ".join("%d tries" % a["tries"] for a in b["attempts"] if a["mode"] != "strict") + ")")
