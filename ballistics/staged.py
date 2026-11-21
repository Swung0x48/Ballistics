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
from .solve import world_for, push_towards, ride as ride_legs

BREAK_KINDS = ("gap", "bridge", "step")
ROLLS = ("roll", "rail", "rail end", "drop")
ORDER = ("strict", "free")
LOOK_TICKS = 132          # how far ahead the follower runs each key plan: one game second
SWITCH_TICKS = 24         # where a two-part key plan changes key
STEP_TICKS = 8            # how many ticks of the chosen plan the ball takes before looking again
CHUNK = STEP_TICKS
BASE = "strict"           # the follower's world, and the only one whose crossings count: nothing placed may be touched
STATUS = {DEAD: "fell out", TRIGGER: "entered a transformer's range", PLACED: "touched something placed"}
BALLS = {"wood": ph.WOOD, "stone": ph.STONE, "paper": ph.PAPER}
from .trafo import SOLVED as SOLVED_BALLS      # noqa: E402  the kinds a route goes on as


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


def choose_push(world, pos, vel, omega, P, way, best: int, ball: ph.Ball = ph.WOOD, scene=None) -> int:
    """The follower's decision for one state: which of the nine pushes to hold next. From that state the kernel runs
    each push LOOK_TICKS ahead (and, if none gets on, each push for SWITCH_TICKS followed by each other push); the
    answer is the first push of the plan that gets furthest along the way P soonest without ending the ball. A plan
    still in the air at its end is run on without keys to see whether the ball comes down.
    way: way-length at each place of P; best: index of the furthest place reached so far."""
    par = ph.Params()
    n = len(P)
    zero = np.zeros((1, 3))
    lo, hi = max(best - 2, 0), min(best + 30, n - 1)
    # a wood or stone transformer the way itself leads through is not an end: the ball is to be transformed there
    on_way = set()
    if world.trafos is not None:
        for k, t in enumerate(world.trafos):
            if t["kind"] in SOLVED_BALLS and (np.linalg.norm(P[lo:hi + 1] - t["pos"], axis=1) < world.spheres[k, 3]).any():
                on_way.add(k)

    def fate(a_tape):
        pushes = np.array([fine.PUSHES[a] for a, _ in a_tape])
        sc = None if scene is None else scene.copy()          # each plan moves its own copy of the loose balls
        st, done, p2, v2, o2, s2, tr = world.rollout(ball, par, pos, vel, omega, pushes, [m for _, m in a_tape], trace=True, scene=sc)
        if st == TRIGGER and int(s2[9]) in on_way:
            reach, at = _progress(tr, P, way, lo, hi) if len(tr) else (way[best], 0)
            return (reach - 0.02 * at), reach
        alive = st == OK
        if alive and s2[8] == 0:
            st3, d3, p3, v3, o3, s3, tr3 = world.rollout(ball, par, p2, v2, o2, zero, [160], trace=True, scene=sc)
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


def follow(world, strict_world, start, P, max_ticks: int = 60000, ball: ph.Ball = ph.WOOD) -> dict:
    """Roll along the plan places P from the state `start` = (pos, vel, omega) by looking ahead: every STEP_TICKS
    ticks, from the state the ball is in (its speed included), the kernel runs a set of key plans LOOK_TICKS ahead
    (each of the nine pushes held; if none of those gets on, each push for SWITCH_TICKS and then each other push),
    and the ball takes the first STEP_TICKS ticks of the plan that gets furthest along the way soonest without
    ending the ball. Nothing is braked to a stop on purpose and no speed is aimed at: a fall or a flight on the
    plan is taken with the speed the ball has. Returns status 'goal', 'fail' (with the reason and the last plan
    place reached), 'clear' (only when `world` is not the strict one: clear of everything placed again) or
    'transform' (the ball came within a wood or stone transformer of another kind that the way P itself leads
    through: `which` is its index in world.trafos, `end` the state at that tick; one it strays into off the way
    is a failure)."""
    pos, vel, omega = np.array(start[0], float), np.array(start[1], float), np.array(start[2], float)
    scene = _scene_of(world, start)                           # the loose balls: carried along, changed by every rollout
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
            return dict(status="goal", index=n - 1, tape=tape, trace=_stack(traces), history=history, end=(pos, vel, omega, scene))
        history.append((ticks, best, pos.copy(), vel.copy(), omega.copy(), scene.copy()))
        a = choose_push(world, pos, vel, omega, P, way, best, ball, scene)
        status, done, pos, vel, omega, st, tr = world.rollout(ball, par, pos, vel, omega, fine.PUSHES[a:a + 1], [STEP_TICKS], trace=True, scene=scene)
        if status == TRIGGER and world.trafos is not None and world.trafos[int(st[9])]["kind"] in SOLVED_BALLS \
                and (np.linalg.norm(P - world.trafos[int(st[9])]["pos"], axis=1) < world.spheres[int(st[9]), 3]).any():
            # only a transformer the way itself leads through: one the ball strays into is a failure, as before
            return dict(status="transform", which=int(st[9]), index=best, tape=tape + [(a, done)], trace=_stack(traces + [tr]), history=history,
                        end=(pos, vel, omega, scene))
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
            return dict(status="clear", index=best, tape=tape, trace=_stack(traces), history=history, end=(pos, vel, omega, scene))
    return dict(status="fail", reason="ran out of time", index=best, tape=tape, trace=_stack(traces), history=history, end=None)


def _scene_of(world, state):
    """The loose balls' state a state carries, as a copy to roll on; the world's own at rest when it carries none (or
    one of another world: the diagnosis world has no loose balls)."""
    sc = state[3] if len(state) > 3 else None
    return world.scene() if sc is None or len(sc) != len(world.props) else np.array(sc, float)


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


NUDGES = (1, 3, 10)       # the sizes of nudge a found passage is tried with: 1 is (0.03 in place, 0.3 in speed, 2 ticks in key timing)


def _robust(world, search, seed, tape, trials: int = 20, rng_seed: int = 1, scale: int = 1, ball: ph.Ball = ph.WOOD, parts=None) -> int:
    """Of `trials` runs with the start nudged (0.03 in place, 0.3 in speed) and every key change up to 2 ticks early
    or late: how many still arrive. scale: the nudge is that many times as large. parts: the crossing's stretches
    (`fine.Crossing.parts`) instead of one tape: a nudged run must go through the same transformers; after each it
    starts again as the new ball at rest, so only the first stretch carries the nudge of the start."""
    rng = np.random.default_rng(rng_seed)
    P = ph.Params()
    ok = 0
    parts = parts or [dict(type="cross", ball=ball.name, tape=tape)]
    ball = BALLS[parts[-1]["ball"]]
    for _ in range(trials):
        pos = seed[0] + rng.normal(0, 0.03 * scale, 3) * np.array([1, 0, 1])
        vel = seed[1] + rng.normal(0, 0.3 * scale, 3)
        omega, good = seed[2], True
        scene = _scene_of(world, seed)
        st = np.zeros(10)
        for k, part in enumerate(parts):
            if part["type"] == "transform":
                pos, vel, omega = (np.array(x, float) for x in part["end"][:3])
                world.idle(scene, len(part["trace"]) - 1)
                continue
            into = parts[k + 1]["which"] if k + 1 < len(parts) and parts[k + 1]["type"] == "transform" else -1
            went = False
            for a, n in part["tape"]:
                status, _d, pos, vel, omega, st, _ = world.rollout(BALLS[part["ball"]], P, pos, vel, omega, fine.PUSHES[a:a + 1],
                                                                   [max(1, n + int(rng.integers(-2 * scale, 2 * scale + 1)))], scene=scene)
                if status == TRIGGER and int(st[9]) == into:
                    went = True
                    break
                if status != OK:
                    good = False
                    break
            if not good or (into >= 0 and not went):
                good = False
                break
        if good and not search.arrived(pos, vel, st):
            status, _d, pos, vel, omega, st, _ = world.rollout(ball, P, pos, vel, omega, np.zeros((1, 3)), [33], scene=scene)
            good = status == OK and st[8] > 0 and search.distance(pos) <= search.reach + 2.0
        ok += bool(good)
    return ok


MAX_CROSSINGS = 3         # a break crossed this often, failing right after it each time, is left as a break
OPERATE_COST = 20.0       # pushing a gate open, as way at the follower's pace: ASSUMED, only to rank plans
OPERATE_TICKS = 198       # how long the push is held at the gate (1.5 game s: a stone ball pushes it open in 1.20 to 1.39, MEASURED)
MORE_TRIES = 1.0          # after the first way across is found, the search goes on for as many tries again (at least MORE_AT_LEAST)
MORE_AT_LEAST = 6000
MOST_WAYS = 10            # ... or until it has this many ways
REFINE_ROUNDS = 160       # changes tried on the way taken
TICKS_PER_COST = ph.PSI_HZ / 9.0      # the coarse level's costs are way at the follower's pace of 9: this turns what is left into ticks


def _worth(world, search, seed, c, ball, before: int):
    """What a way across is worth: how many of 20 nudged runs still arrive (nudge 1, then 3), then how soon the ball
    is where the way leads (the ticks before it starts, its own, and the coarse level's estimate of what is left)."""
    n1 = _robust(world, search, seed, c.tape, scale=1, ball=ball, parts=c.parts)
    n3 = _robust(world, search, seed, c.tape, scale=3, ball=ball, parts=c.parts)
    end = getattr(c, "end_state", None)
    total = before + c.ticks + search.score(c.end, getattr(c, "end_ball", None), end[3] if end is not None and len(end) > 3 else None) * TICKS_PER_COST
    return (n1, n3, -total)


def _refine(world, search, seed, c, ball, before: int, rounds: int = REFINE_ROUNDS, rng_seed: int = 7):
    """Adjust a way across that is one stretch: change one key's length by a chunk, change one key, or drop one,
    and keep the change when the way is worth more (`_worth`). Returns (the way, its worth, how many changes were
    kept)."""
    rng = np.random.default_rng(rng_seed)
    best, worth, kept = c, _worth(world, search, seed, c, ball, before), 0
    if len(c.parts) != 1:
        return best, worth, kept
    for _ in range(rounds):
        tape = [list(x) for x in best.tape]
        if not tape:
            break
        k = int(rng.integers(0, len(tape)))
        what = int(rng.integers(0, 4))
        if what == 0:
            tape[k][1] += CHUNK
        elif what == 1:
            tape[k][1] -= CHUNK
        elif what == 2:
            tape[k][0] = int(rng.integers(0, len(fine.PUSHES)))
        else:
            tape[k][1] = 0
        merged = []
        for a, n in tape:
            if n <= 0:
                continue
            if merged and merged[-1][0] == a:
                merged[-1][1] += n
            else:
                merged.append([a, n])
        if not merged:
            continue
        cand = fine.rebuild(search, seed, [(a, n) for a, n in merged], ball.name)
        if cand is None:
            continue
        w2 = _worth(world, search, seed, cand, ball, before)
        if w2 > worth:
            best, worth, kept = cand, w2, kept + 1
    return best, worth, kept


def _fan_bounds(world, segs, log) -> int:
    """Run every stretch a paper ball takes near a fan again with the fans' least and most reach for a ball at a box's
    edge; mark it seg["fan_bounds"] = {"least": arrives, "most": arrives} (arrives: not lost, and ending within 2.5 of
    where it ended). Returns how many were checked."""
    from .kernel import OK
    from .solve import _lift_world
    if not len(world.fans):
        return 0
    n = 0
    bounds = {e: _lift_world(world, e) for e in ("least", "most")}
    for sg in segs:
        if sg.get("type") not in ("follow", "cross") or sg.get("ball") != "paper" or sg.get("end") is None or not len(sg.get("trace", ())):
            continue
        tr = np.asarray(sg["trace"])[:, :3]
        hd = np.hypot(tr[:, None, 0] - world.fans[None, :, 0], tr[:, None, 2] - world.fans[None, :, 2])
        if not (hd < 6.0).any():
            continue
        st0 = sg["start"]
        res = {}
        for e, w in bounds.items():
            sc = np.array(st0[3], float) if len(st0) > 3 and st0[3] is not None and len(st0[3]) == len(w.props) else w.scene()
            out = fine.replay(w, BALLS["paper"], ph.Params(), ph.State(np.array(st0[0], float), np.array(st0[1], float), np.array(st0[2], float)), sg["tape"], scene=sc)
            res[e] = bool(out["status"] == OK and np.linalg.norm(out["pos"] - np.asarray(sg["end"][0], float)) <= 2.5)
        sg["fan_bounds"] = res
        n += 1
        log("   a paper ball's stretch by a fan (%s, %.1f game s): with the least fan reach %s, with the most %s" % (
            sg["type"], len(tr) / 132, "arrives" if res["least"] else "does NOT arrive", "arrives" if res["most"] else "does NOT arrive"))
    return n


def solve_sector(mesh, info, g, sector: int, tries: int = 300_000, seed: int = 0, log=print, start=None, ball: str = "wood") -> dict:
    """sector: 1-based. g: the coarse graph of `mesh` with the loose balls in it (solid).
    start: the state (pos, vel, omega) the ball enters the sector with, as in a normal game: the state it reached
    the sector's first checkpoint in. Without it the ball starts at rest under the sector's reset point, which is
    how the level begins and how a ball comes back after it was lost.
    ball: the kind of ball the sector is entered with. Where the ball comes within a wood or stone transformer of
    another kind, it is transformed (`ballistics.trafo`) and the sector goes on with the new ball; a search for a
    crossing only starts from states after the last transformation."""
    from . import trafo
    worlds = {m: world_for(mesh, info, m, sector) for m in ORDER}
    ball_now = BALLS[ball]
    strict = worlds["strict"]
    # the coarse level of this sector over the ball's kind and its mechanisms' states (`solve.state_graph_for`):
    # the plan says which kind the ball is and what state each mechanism is in at each place; the states are read
    # back from the kernel's scene after every stretch
    from .solve import state_graph_for
    sg, state_of = state_graph_for(g, info, sector, strict, OPERATE_COST)
    N = len(g.pos)
    anchors = [info["start"]] + info["checkpoints"] + [info["airship"]]
    goal = g.nearest(anchors[sector], 25.0)
    togo = sg.togo(goal)
    if sg.mechs:
        log("sector %d: the coarse level keeps the states of %s" % (sector, ", ".join("%s (%s)" % (m["name"], m["type"]) for m in sg.mechs)))
    reset = info["reset_points"][sector - 1]
    s0 = fine.rest_state(strict, reset[0], reset[2], reset[1] + 0.5)
    state = (s0.pos, s0.vel, s0.omega) if start is None else tuple(np.array(x, float) for x in start[:3])
    # this sector's loose balls: at rest where they lie, or as the start has them (a route going on within the sector)
    state = state + (_scene_of(strict, start) if start is not None else strict.scene(),)
    entry = np.array(state[0], float)                             # where the ball enters the sector

    def cs():
        """The coarse state the ball is in now: its kind and the mechanisms' states, read from the scene."""
        return state_of(ball_now.name, state[3] if len(state) > 3 else None)

    def tv():
        """How far each place is from the goal, in the coarse state the ball is in now."""
        return togo[cs()]
    loose =[(p["type"], np.array(p["position"], float)) for p in info["placements"] if p["type"].startswith(coarse.LOOSE)]
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
        # in the coarse state the ball is in now, from the place nearest; the plan's states change where it
        # transforms or a mechanism's state changes on the way
        nodes, kinds, _states = sg.plan(cs(), nearest_node(g, tv(), pos), togo)
        register(nodes, kinds)
        return nodes, kinds

    lifts = g.info.get("lifts", {})
    unridden = set()                                              # lifts ridden once that did not get there

    def lift_ahead(nodes, kinds):
        """The first lift on the plan, to be ridden: (its index, its edge), or None (none, or the first one was ridden
        and did not get there: that one is left to the follower and the search, as any way)."""
        for e, k in enumerate(kinds):
            if k == "lift" and (nodes[e], nodes[e + 1]) in lifts:
                return None if (nodes[e], nodes[e + 1]) in unridden else (e, lifts[(nodes[e], nodes[e + 1])])
        return None

    def at_fan(pos, L):
        """Whether the ball is where a lift of this fan is entered from (within 5 of its axis, as its entries are)."""
        o = L["fan"]
        return math.hypot(pos[0] - o[0], pos[2] - o[2]) <= 5.0 and o[1] - 2.0 <= pos[1] <= o[1] + 17.0

    def ride_lift(key, L, state):
        """Ride a lift the way it was made: to the fan's axis, then its own legs, each held until its point is reached
        (`solve.ride`), not chased by the follower. Returns (status, end state, which transformer or -1); the stretch
        is on the route only if it got there."""
        o = L["fan"]
        legs = [(push_towards(o - state[0]), 90, o)] + list(L["legs"])
        sc = _scene_of(strict, state)
        tape, tr, st, (pos, vel, om), stats = ride_legs(strict, ball_now, state, legs, scene=sc, arrive_at=g.pos[key[1]])
        got = st == TRIGGER or (st == OK and len(tr) and ((np.linalg.norm(tr[:, :3] - g.pos[key[1]], axis=1) <= 2.5) & (tr[:, 7] > 0)).any())
        if not got:
            return "missed", state, -1
        end = (pos, vel, om, sc)
        segs.append(dict(type="follow", mode=BASE, ball=ball_now.name, start=state, tape=tape, trace=tr, end=end, lift=key))
        log("   rode the lift to %s, %.1f game s" % (np.round(g.pos[key[1]], 1), len(tr) / 132))
        return "got", end, int(stats[9]) if st == TRIGGER else -1

    def operate(m, state):
        """At gate m's push place: hold the push nearest to its x for OPERATE_TICKS. Returns the state after, or None
        if the ball was lost."""
        mc = sg.mechs[m]
        a = push_towards(mc["axis"])
        sc = _scene_of(strict, state)
        status, done, pos, vel, omega, _st, tr = strict.rollout(ball_now, ph.Params(), state[0], state[1], state[2], fine.PUSHES[a:a + 1], [OPERATE_TICKS],
                                                                 trace=True, scene=sc)
        if status != OK:
            log("   pushing gate %s: the ball was lost" % mc["name"])
            return None
        end = (pos, vel, omega, sc)
        segs.append(dict(type="follow", mode=BASE, ball=ball_now.name, start=state, tape=[(a, done)], trace=tr, end=end, operates=mc["name"]))
        log("   pushed gate %s for %.1f game s" % (mc["name"], done / 132))
        return end

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
        # a step that changes a mechanism's state by itself (pushing a gate open) is not rolled along: the way ends
        # at its place, the step is done there, and the plan goes on from the state it left
        cut = kinds.index("operate") if "operate" in kinds else None
        if cut is not None and (cut == 0 or np.linalg.norm(state[0] - g.pos[nodes[cut]]) <= 2.5):
            m = next(i for i, mc in enumerate(sg.mechs) if mc.get("operate") == nodes[cut])
            before = cs()
            nxt = operate(m, state)
            if nxt is None:
                break
            state = nxt
            if cs() == before:
                # it did not open: that step is not there for the coarse level any more
                log("   gate %s did not open: no longer planned" % sg.mechs[m]["name"])
                sg.drop_mech_edge(m, coarse.GATE_OPERATE)
                togo = sg.togo(goal)
            nodes, kinds = replan(state[0])
            continue
        # a lift is ridden by its own legs once the ball is at its fan; the follower's way ends there
        la = lift_ahead(nodes, kinds)
        if la is not None and (cut is None or la[0] < cut):
            c, L = la
            if c == 0 or at_fan(state[0], L):
                key = (nodes[c], nodes[c + 1])
                how, nxt, which = ride_lift(key, L, state)
                if how == "missed":
                    # ridden from where the ball is it did not get there: the follower and the search take it on, as
                    # before lifts were ridden (they find their own way up, or it becomes a break)
                    log("   the lift to %s did not get there ridden from here: left to the follower" % np.round(g.pos[key[1]], 1))
                    unridden.add(key)
                    continue
                state = nxt
                if which >= 0:                                    # it went into a transformer's range
                    item = strict.trafos[which]
                    new, tr1, end1, _s1, _st1 = trafo.stretch(strict, item, scene=_scene_of(strict, state))
                    segs.append(dict(type="transform", mode=BASE, ball=new.name, was=ball_now.name, at=item["name"], which=which, start=state, tape=[],
                                     trace=tr1, end=end1))
                    log("   transformed %s -> %s by %s, %.2f game s" % (ball_now.name, new.name, item["name"], len(tr1) / 132))
                    ball_now, state, mode_now, seeds, older = new, end1, BASE, [], []
                nodes, kinds = replan(state[0])
                continue
            cut = c
        if cut is not None:
            nodes, kinds = nodes[:cut + 1], kinds[:cut]
        # the way the follower sees: the plan's places, and for a launch on it the run-up and the landing of the
        # kernel run that made the edge (its start place alone may be far from the ledge)
        way, node_of = [g.pos[nodes[0]]], [0]
        for e, k in enumerate(kinds):
            if k in ("launch", "lift"):
                for q in g.info["launches" if k == "launch" else "lifts"][(nodes[e], nodes[e + 1])]["way"][1:]:
                    way.append(np.asarray(q, np.float64))
                    node_of.append(e)
            way.append(g.pos[nodes[e + 1]])
            node_of.append(e + 1)
        r = follow(worlds[mode_now], strict, state, np.array(way), ball=ball_now)
        r["index"] = node_of[min(r["index"], len(node_of) - 1)]
        item = worlds[mode_now].trafos[r["which"]] if r["status"] == "transform" else None
        segs.append(dict(type="follow", mode=mode_now, ball=ball_now.name, start=state, tape=r["tape"], trace=r["trace"], end=r["end"]))
        seeds += [(len(segs) - 1, h) for h in r["history"]]
        if r["status"] == "transform":
            # the jump: nothing for `body_after` ticks, then the new ball at rest at the hover point; the tick after
            # that it only falls, and from the next one the keys act
            new, tr1, end1, _s1, _st1 = trafo.stretch(worlds[mode_now], item, scene=_scene_of(worlds[mode_now], r["end"]))
            segs.append(dict(type="transform", mode=mode_now, ball=new.name, was=ball_now.name, at=item["name"], which=r["which"], start=r["end"], tape=[],
                             trace=tr1, end=end1))
            log("   transformed %s -> %s by %s, %.2f game s" % (ball_now.name, new.name, item["name"], len(tr1) / 132))
            ball_now, state, mode_now, seeds, older = new, end1, BASE, [], []
            nodes, kinds = replan(state[0])
            continue
        for e, k in enumerate(kinds):
            if k in BREAK_KINDS and e + 1 <= r["index"] and breaks[(nodes[e], nodes[e + 1])]["status"] == "pending":
                breaks[(nodes[e], nodes[e + 1])]["status"] = "rolled"
        if la is not None and r["status"] != "goal":
            # it stopped at the lift's fan, or was there before it went wrong: ride it from there next round
            if r["end"] is not None and at_fan(r["end"][0], la[1]):
                state = r["end"]
                nodes, kinds = replan(state[0])
                continue
            hist = r["history"][:-8] if len(r["history"]) > 12 else r["history"]
            h = next((h for h in reversed(hist) if at_fan(h[2], la[1])), None)
            if h is not None:
                last = segs[-1]
                last["tape"], last["trace"], last["end"] = last["tape"][:h[0] // CHUNK], last["trace"][:h[0]], (h[2], h[3], h[4], h[5])
                seeds = [e for e in seeds if not (e[0] == len(segs) - 1 and e[1][0] > h[0])]
                state = last["end"]
                nodes, kinds = replan(state[0])
                continue
        if r["status"] == "goal" and cut is not None:
            state = r["end"]                                      # at the place of a step: do it next round
            nodes, kinds = replan(state[0])
            continue
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
        beyond = tv() <= tv()[nodes[kt]] + 1e-9
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
            sg.drop(u, v, b["kind"])
            togo = sg.togo(goal)
            b["status"] = "bypassed"
            state = segs[-1]["end"] if segs[-1]["end"] is not None else state
            if segs[-1]["end"] is None:                           # the stretch failed: go back to the last state it was fine in
                h = r["history"][-1] if len(r["history"]) < 20 else r["history"][-17]
                segs[-1]["tape"], segs[-1]["trace"], segs[-1]["end"] = segs[-1]["tape"][:h[0] // CHUNK], segs[-1]["trace"][:h[0]], (h[2], h[3], h[4], h[5])
                seeds = [e for e in seeds if not (e[0] == len(segs) - 1 and e[1][0] > h[0])]
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
            s = fine.Search(worlds[mode], targets, ball=ball_now, seed=seed, places=g.pos, togo=dict(by_state=togo, state_of=state_of), transforms=True)
            ids = {}
            for si, h in cand:
                idx = s.add_start(ph.State(h[2], h[3], h[4]), scene=h[5] if mode == "strict" else None)
                if idx >= 0:
                    ids[idx] = (si, h)
            if not ids:
                return None
            # the first way across is seldom the best: in strict mode the search goes on a while and keeps every way it finds
            keep = mode == "strict"
            c = s.run(max_expansions=tries, max_seconds=1e9, more=MORE_TRIES if keep else 0.0, at_least=MORE_AT_LEAST if keep else 0, most=MOST_WAYS if keep else 1)
            b["attempts"].append(dict(mode=mode, where=where, tries=c.expansions, seconds=c.seconds, found=c.found, ways=len(s.goals)))
            log("   %s at %s (%s): %s from %s, %s after %d tries, %.0f s" % (b["no"], np.round(b["near"], 1), b["why"], mode, where,
                                                                           "crossed (%d ways)" % len(s.goals) if c.found else "not crossed", c.expansions, c.seconds))
            return (mode, s, c, ids) if c.found else None

        # A break counts as crossed only in strict mode. One crossed again and again without the ball getting on past it
        # (it fails right after it each time) is not crossed: it stays a break
        b["times"] = b.get("times", 0) + 1
        found = None
        for where, cand in (pools if b["times"] <= MAX_CROSSINGS else ()):
            found = attempt("strict", where, cand)
            if found:
                break
        if b["times"] > MAX_CROSSINGS:
            log("   %s at %s: crossed %d times already and the ball got no further each time: left as a break" % (b["no"], np.round(b["near"], 1), MAX_CROSSINGS))
        if not found:
            # It stays a break. One more run with the loose balls taken out only says whether they are what stands
            # in the way: its trajectory is not used, and nothing after this break counts as reached
            b["would_pass"] = None
            for mode in ORDER[1:]:
                got = attempt(mode, pools[0][0], pools[0][1])
                if got:
                    _m, _s, c, _ids = got
                    b["would_pass"] = dict(mode=mode, tries=c.expansions,
                                           through=[ty for ty, q in loose if np.linalg.norm(c.trace[:, :3] - q, axis=1).min() < 2.0 * BALL_R])
                    break
        if found:
            mode, s, c, ids = found
            # of the ways found, the one worth most; then small changes to it, kept when they make it worth more
            def before(si, h):
                return int(sum(len(sg["trace"]) for sg in segs[:si] if sg["type"] != "assumed")) + int(h[0])
            ways = []
            for cand in s.crossings():
                si, h = ids[cand.root]
                ways.append((_worth(worlds[mode], s, (h[2], h[3], h[4], h[5]), cand, ball_now, before(si, h)), len(ways), cand, si, h))
            first_worth = ways[0][0]
            worth, _k, c, si, h = max(ways, key=lambda x: (x[0], -x[1]))
            c, refined, changes = _refine(worlds[mode], s, (h[2], h[3], h[4], h[5]), c, ball_now, before(si, h))
            b["ways"] = dict(found=len(ways), first=dict(arrives=first_worth[0], arrives3=first_worth[1], ticks=int(ways[0][2].ticks)),
                             chosen=dict(arrives=worth[0], arrives3=worth[1], ticks=int(max(ways, key=lambda x: (x[0], -x[1]))[2].ticks)),
                             refined=dict(arrives=refined[0], arrives3=refined[1], ticks=int(c.ticks), changes=changes))
            log("      %s: %d ways found; the first arrives %d of 20 in %.1f game s, the one chosen %d of 20, after %d changes %d of 20 in %.1f game s" % (
                b["no"], len(ways), first_worth[0], ways[0][2].ticks / 132, worth[0], changes, refined[0], c.ticks / 132))
            for given_up in segs[si + 1:]:                        # crossings made on a way that is now given up
                if given_up["type"] == "cross":
                    for ob in breaks.values():
                        if ob["no"] == given_up["no"]:
                            ob["status"] = "abandoned"
            segs = segs[:si + 1]
            older = [e for e in older + seeds if e[0] < si or (e[0] == si and e[1][0] < h[0])]
            n_ticks = h[0]
            segs[si]["tape"] = segs[si]["tape"][:n_ticks // CHUNK]
            segs[si]["trace"] = segs[si]["trace"][:n_ticks]
            segs[si]["end"] = (h[2], h[3], h[4], h[5])
            for part in c.parts:                                  # one stretch, or several with the transformations between them
                segs.append(dict(part, mode=mode, no=b["no"]) if part["type"] == "cross" else dict(part, mode=mode))
                if part["type"] == "transform":
                    log("   %s: the crossing goes through %s (%s -> %s)" % (b["no"], part["at"], part["was"], part["ball"]))
            through = [(ty, q) for ty, q in loose if np.linalg.norm(c.trace[:, :3] - q, axis=1).min() < 2.0 * BALL_R]
            nudged = {k: _robust(worlds[mode], s, (h[2], h[3], h[4], h[5]), c.tape, scale=k, ball=ball_now, parts=c.parts) for k in NUDGES}
            if len(c.parts) > 1:
                ball_now, older = BALLS[c.end_ball], []            # earlier states are another kind of ball: no search starts from them
            b.update(status="crossed", mode=mode, ticks=c.ticks, back=float(np.linalg.norm(h[2] - b["near"])), flights=_flights(c.trace),
                     through=[ty for ty, _q in through],
                     arrives=nudged[1], nudged=nudged, trials=20, landed=np.array(c.end))
            state = c.end_state
            clear = strict.mesh.distance(state[0], BALL_R + 1.0)[0] - BALL_R > 0.1
            mode_now = BASE if (mode == "strict" or clear) else mode
            seeds = []
            nodes, kinds = replan(state[0])
        else:
            b["status"] = "unresolved"
            last = segs[-1]                                       # keep the stretch up to the last state it was fine in
            cut_at = None
            if r["history"]:
                h = r["history"][-1] if len(r["history"]) < 20 else r["history"][-17]
                last["tape"], last["trace"], last["end"] = last["tape"][:h[0] // CHUNK], last["trace"][:h[0]], (h[2], h[3], h[4], h[5])
                cut_at = h[0]
            # (a later search may start from this stretch: only from what is left of it)
            older = [e for e in older + seeds if cut_at is None or not (e[0] == len(segs) - 1 and e[1][0] > cut_at)]
            far = g.pos[nodes[kt]]
            segs.append(dict(type="assumed", no=b["no"], to=far.copy()))
            s1 = fine.rest_state(strict, far[0], far[2], far[1] + 1.0)
            kept = last.get("end")
            state, mode_now, seeds = (s1.pos, s1.vel, s1.omega, _scene_of(strict, kept) if kept is not None else strict.scene()), BASE, []
            nodes, kinds = replan(state[0])
    # a paper ball by a fan: whether it is pushed at a box's edge depends on its orientation, which the kernel does
    # not know (`fans.Rules.params`). Each stretch of the way taken that a paper ball rolls near a fan is run again
    # with the two bounds: it is sure only if it gets to the same end with both
    fan_bounds_checked = _fan_bounds(strict, segs, log)
    # at the goal: the last stretch ends there, whether the follower rolled it or a searched crossing arrived on it
    reached = bool(segs and segs[-1]["type"] in ("follow", "cross") and segs[-1]["end"] is not None and np.linalg.norm(segs[-1]["end"][0] - g.pos[goal]) <= 2.5)
    unresolved = [breaks[k]["no"] for k in order if breaks[k]["status"] == "unresolved"]
    res = dict(sector=sector, tries=tries, segments=segs, breaks=[breaks[k] for k in order], first=first, reached=reached, unresolved=unresolved,
               solved=reached and not unresolved, ticks=int(sum(len(sg["trace"]) for sg in segs if sg["type"] != "assumed")),
               goal=g.pos[goal].copy(), start=entry, rolling_start=start is not None, ball=ball, end_ball=ball_now.name,
               transforms=[dict(at=sg["at"], was=sg["was"], to=sg["ball"]) for sg in segs if sg["type"] == "transform"],
               end_state=segs[-1]["end"] if reached and not unresolved else None)
    res["flights"] = [f for sg in segs if sg["type"] == "follow" for f in _flights(sg["trace"], 20)]
    res["replays"] = replays(worlds, res)
    return res


def replays(worlds, res) -> bool:
    """Run every segment's keys again from its start, in its own mode: the same end states, and each segment starts
    where the one before ended (or at rest on the far side of an unresolved break)."""
    P = ph.Params()
    prev = None
    for k, sg in enumerate(res["segments"]):
        if sg["type"] == "assumed":
            prev = None
            continue
        if prev is not None and np.abs(prev[0] - sg["start"][0]).max() > 1e-9:
            return False
        if sg["tape"]:
            pushes = np.array([fine.PUSHES[a] for a, _ in sg["tape"]])
            st, done, pos, vel, omega, s, _ = worlds[sg["mode"]].rollout(BALLS[sg.get("ball", "wood")], P, sg["start"][0], sg["start"][1], sg["start"][2], pushes,
                                                                        [n for _, n in sg["tape"]], scene=_scene_of(worlds[sg["mode"]], sg["start"]))
            nxt = res["segments"][k + 1] if k + 1 < len(res["segments"]) else None
            ok = st == OK or (st == TRIGGER and nxt is not None and nxt["type"] == "transform")      # a stretch may end at a transformer
            if not ok or sg["end"] is None or np.abs(pos - sg["end"][0]).max() > 1e-9:
                return False
        prev = sg["end"]
    return True


def summary(res, log=print):
    """The break list of a solved sector, in words."""
    n_rolled = sum(b["status"] == "rolled" for b in res["breaks"])
    log("sector %d (%s): %s; %.1f game s of trajectory; %d breaks in all, %d of them rolled across by the plain follower; replay %s"
        % (res["sector"], "entered rolling at %.1f" % float(np.hypot(res["segments"][0]["start"][1][0], res["segments"][0]["start"][1][2])) if res.get("rolling_start") else "from rest",
           "reaches the goal" if res["solved"] else "does NOT get through: %s left as break%s" % (", ".join(res["unresolved"]) or "the way", "s" if len(res["unresolved"]) > 1 else ""),
           res["ticks"] / 132, len(res["breaks"]), n_rolled, "agrees" if res["replays"] else "DISAGREES"))
    log("   the follower's stretches: %.1f game s" % (sum(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "follow") / 132))
    for tr in res.get("transforms", ()):
        log("   transformed %s -> %s by %s (%.2f game s each)" % (tr["was"], tr["to"], tr["at"], next(len(sg["trace"]) for sg in res["segments"] if sg["type"] == "transform") / 132))
    if res.get("ball", "wood") != "wood" or res.get("end_ball", "wood") != "wood":
        log("   the ball: enters as %s, leaves as %s" % (res.get("ball", "wood"), res.get("end_ball", "wood")))
    if res.get("extras"):
        from . import extras
        for line in extras.lines(res["extras"]):
            log(line)
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
