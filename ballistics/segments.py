"""Write a solved level out as segments: the route file of version 2, `L<nn>.segments.json`, next to the reference
line of version 1 (`L<nn>.npz`, `ballistics.route`), which stays as it is. The format is in docs/files.md.

The solver's segments become pieces of the route: what the follower rolled (consecutive stretches of it in one,
split where the ball flies), each searched crossing, each transformation, each gate push, each fan lift, and each
break that was left. Every piece says which ball it is, where it is on the line, its exit and its corridor, and
what it does to the mechanisms.

The exit and the corridor are rough: envelopes of nudged replays in the kernel, not proved. A crossing is replayed
whole, from its start nudged, with its keys a little early or late, as the solver itself measures one (`staged.
_robust`). A stretch the follower rolled is not: the follower chose its keys closed loop, and its keys replayed
open loop from a nudged start lose the ball within seconds (on level 1, a stretch of 30 game s: 20 of 20 lost).
It is cut into windows of WINDOW ticks, and each window is replayed from the state the route has at its start,
nudged. So its envelope says how far the ball strays in a game second when it was on the route at the start of it,
which is what a closed-loop executor has to keep up with.

Nothing here searches. Apart from these replays it only reads the solver's results.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from . import fine, route
from . import physics as ph

VERSION = 2
WINDOW = 132              # ticks: a stretch the follower rolled is replayed in windows of a game second
TRIALS = 20               # nudged replays per window (as the solver's own count for a crossing)
NUDGE = (0.03, 0.3, 2)    # in place (x and z), in speed (each axis), key changes early or late in ticks: the solver's nudge of size 1
FLIGHT = 40               # working values: touching nothing this many ticks in a row (0.3 game s), and FLIGHT_ACROSS or more
FLIGHT_ACROSS = 4.0       # across from take-off to landing, is a flight, a piece of its own (the hops of a paper ball by a fan are not)
EVERY = 1.0               # the corridor gives a half width every this much arc length of the piece
LEAST = 10                # fewer trials than this get through a piece's last window: its exit has no envelope, only the route's own end
STILL = 0.5               # slower than this, a direction is not given
MOVED = 0.01              # degrees: a push board that was awake has been struck once it has turned this much
# which balls can change a mechanism's state (docs/mechanisms.md, all MEASURED): a stone ball opens a gate (a wood one
# pushes it 1.6 to 3.4 of the 4.5 it needs, paper hardly moves it); a wood or stone ball pushes a push board over (paper
# is thrown back); only a stone ball tips a seesaw
CAN = {"gate": ["stone"], "push_board": ["wood", "stone"], "seesaw": ["stone"]}
FROM_WINDOW = "%d open-loop replays of the piece's last %d ticks of keys, from the route's state then, nudged" % (TRIALS, WINDOW)
FROM_WINDOWS = "%d open-loop replays of each window of %d ticks of the piece's keys, from the route's state at its start, nudged" % (TRIALS, WINDOW)
FROM_WHOLE = "%d open-loop replays of the whole piece, from its start nudged" % TRIALS
NUDGE_TEXT = ("a nudged replay starts %.2f off in x and z and %.1f off in each axis of the speed (normal, standard deviations), "
              "with every key change up to %d ticks early or late (uniform)" % NUDGE)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _r(v, nd=3):
    return [round(float(x), nd) for x in np.asarray(v, float).ravel()]


def mechanisms_of(world) -> list:
    """The mechanisms of a world whose simple state the coarse level keeps, with the solver's thresholds
    (`solve.state_graph_for`): [(name, type, index, reader(scene) -> (state 0/1, value))]. A gate is open once it has
    slid GATE_FULL (value: how far it slid); a push board is down once it has turned BOARD_DOWN (value: its angle in
    degrees); a seesaw is tipped when it lies at its other end, below -10 degrees (value: its angle)."""
    from .kernel import _hinge_angle
    from .solve import BOARD_DOWN, GATE_FULL
    out = []
    for k, p in enumerate(world.props):
        if p.get("leader") != k:
            continue
        kind = str(p.get("kind", ""))
        name = str(p.get("body") or p["name"]).split("/")[0]
        if kind.startswith("P_Modul_01_Pusher"):
            rest = np.array(p["rest"], float)

            def gate(sc, k=k, rest=rest):
                v = float(np.linalg.norm(sc[k, :3] - rest))
                return int(v >= GATE_FULL), v
            out.append((name, "gate", k, gate))
        elif p.get("hinge") and kind.startswith(("P_Modul_25", "P_Modul_30")):
            def angle(sc, k=k):
                return float(np.degrees(_hinge_angle(world.prop_consts, k, sc[k, 0], sc[k, 1], sc[k, 2])))
            if kind.startswith("P_Modul_25"):
                out.append((name, "push_board", k, lambda sc, f=angle: (int(f(sc) >= BOARD_DOWN), f(sc))))
            else:
                out.append((name, "seesaw", k, lambda sc, f=angle: (int(f(sc) < -10.0), f(sc))))
    return out


def _runs(tape, a: int, b: int) -> list:
    """The keys of ticks a to b of a tape, consecutive equal keys as one run."""
    out, t = [], 0
    for k, n in tape:
        lo, hi = max(t, a), min(t + int(n), b)
        if hi > lo:
            if out and out[-1][0] == k:
                out[-1][1] += hi - lo
            else:
                out.append([int(k), hi - lo])
        t += int(n)
    return [tuple(x) for x in out]


def _nudge_keys(runs, rng) -> list:
    """The same keys with every change moved up to NUDGE[2] ticks early or late; the whole as long as before."""
    if len(runs) < 2:
        return list(runs)
    W = sum(n for _a, n in runs)
    cut = np.cumsum([n for _a, n in runs])[:-1] + rng.integers(-NUDGE[2], NUDGE[2] + 1, len(runs) - 1)
    cut = np.maximum.accumulate(np.clip(cut, 0, W))
    b = np.r_[0, cut, W]
    return [(a, int(b1 - b0)) for (a, _n), b0, b1 in zip(runs, b[:-1], b[1:]) if b1 > b0]


def _scene(world, state):
    from .staged import _scene_of
    return np.array(_scene_of(world, state), float)


def _states_at(world, ball, start, tape, cuts) -> tuple:
    """The route's state after each of `cuts` ticks from `start`, by running its tape again: ({cut: (pos, vel, omega,
    scene)}, the status it ended with)."""
    from .kernel import OK
    P = ph.Params()
    pos, vel, om = (np.array(x, float) for x in start[:3])
    sc = _scene(world, start)
    cuts = sorted(set(int(c) for c in cuts))
    out, t, ci, status = {}, 0, 0, OK
    while ci < len(cuts) and cuts[ci] == 0:
        out[0] = (pos.copy(), vel.copy(), om.copy(), sc.copy())
        ci += 1
    for a, n in tape:
        left = int(n)
        while left > 0 and status == OK:
            step = left if ci >= len(cuts) else min(left, cuts[ci] - t)
            status, done, pos, vel, om, _s, _ = world.rollout(ball, P, pos, vel, om, fine.PUSHES[a:a + 1], [step], scene=sc)
            t += done
            left -= done
            while ci < len(cuts) and cuts[ci] == t:
                out[t] = (pos.copy(), vel.copy(), om.copy(), sc.copy())
                ci += 1
            if done < step:
                break
    return out, status


def _trials(world, ball, state, runs, rng, may_trigger: bool) -> list:
    """TRIALS nudged replays of `runs` from `state`: each dict(ok, pos, vel, touching, trace). ok: the ball was not
    lost (and went into no transformer, unless may_trigger: the piece ends at one)."""
    from .kernel import OK, TRIGGER
    P = ph.Params()
    out = []
    for _ in range(TRIALS):
        pos = state[0] + rng.normal(0, NUDGE[0], 3) * np.array([1.0, 0.0, 1.0])
        vel = state[1] + rng.normal(0, NUDGE[1], 3)
        sc = state[3].copy()
        keys = _nudge_keys(runs, rng)
        st, _d, p, v, _o, stats, tr = world.rollout(ball, P, pos, vel, state[2].copy(), fine.PUSHES[[a for a, _n in keys]], [n for _a, n in keys],
                                                    trace=True, scene=sc)
        out.append(dict(ok=st == OK or (may_trigger and st == TRIGGER), pos=p, vel=v, touching=bool(stats[5] > 0), trace=tr[:, :3]))
    return out


def _exit(end, trials, flight: bool, source: str) -> dict:
    """The exit of a piece: the route's own end, and the envelope of the trials that got through its last window."""
    good = [t for t in trials if t["ok"]]
    e = dict(at=_r(end[0]), vel=_r(end[1]), trials=len(trials), not_lost=len(good), rough=True, **{"from": source})
    if len(good) < LEAST:
        e.update(box=None, speed_h=None, vy=None, dir=None, dir_tol_deg=None)
        return e
    Ps = np.vstack([end[0]] + [t["pos"] for t in good])
    Vs = np.vstack([end[1]] + [t["vel"] for t in good])
    e["box"] = dict(min=_r(Ps.min(0)), max=_r(Ps.max(0)))
    sh = np.hypot(Vs[:, 0], Vs[:, 2])
    e["speed_h"] = _r([sh.min(), sh.max()])
    airborne = flight or any(not t["touching"] for t in good)
    e["vy"] = _r([Vs[:, 1].min(), Vs[:, 1].max()]) if airborne else None
    # the direction is horizontal for a flight too: its end is the tick before it lands, and a nudged one that landed
    # a tick sooner is going up again, so the direction in three dimensions says little (its vy is given)
    D = Vs * np.array([1.0, 0.0, 1.0])
    n = np.linalg.norm(D, axis=1)
    if n[0] < STILL:
        e["dir"], e["dir_tol_deg"] = None, None
        return e
    d0 = D[0] / n[0]
    ang = [math.degrees(math.acos(float(np.clip(D[k] @ d0 / n[k], -1.0, 1.0)))) for k in range(1, len(D)) if n[k] >= STILL]
    e["dir"] = _r(d0, 4)
    e["dir_tol_deg"] = round(math.ceil(max(ang, default=0.0) * 10) / 10, 1)
    return e


def _corridor(rows, windows, source: str) -> dict:
    """Half widths every EVERY of the piece's arc length: how far the trials of each window strayed from the route's
    rows of that window (to the nearest of them)."""
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rows, axis=0), axis=1))] if len(rows) else np.zeros(1)
    hw = np.zeros(int(arc[-1] // EVERY) + 1)
    n = lost = 0
    for b0, b1, trials in windows:
        R = rows[b0:b1]
        for t in trials:
            n += 1
            if not t["ok"]:
                lost += 1
                continue
            if not len(t["trace"]) or not len(R):
                continue
            d2 = ((t["trace"][:, None, :] - R[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(1)
            d = np.sqrt(d2[np.arange(len(j)), j])
            np.maximum.at(hw, (arc[b0 + j] // EVERY).astype(int), d)
    return dict(every=EVERY, half_width=_r(hw, 2), trials=n, lost=lost, rough=True, **{"from": source})


def _impact(world, ball, state, keys, k: int, reader, rng) -> dict | None:
    """Where a piece pushes a push board over: the ball's speed the tick before it strikes the board, in the route
    and in TRIALS nudged runs from 66 ticks before; the trials count only if they push it over too. The board is
    struck in the tick it wakes (a board at rest lies still until then, and the ball loses most of its speed in that
    tick; the board's angle changes from the next), or, if it was awake, once it has turned MOVED. keys: the
    piece's keys, one per tick. None if the board is not struck in the route."""
    from .kernel import OK
    P = ph.Params()
    pos, vel, om = (np.array(x, float) for x in state[:3])
    sc = state[3].copy()

    def struck_test(sc):
        a0, awake0 = reader(sc)[1], sc[k, 9] == 1.0
        return lambda sc: (not awake0 and sc[k, 9] == 1.0) or abs(reader(sc)[1] - a0) > MOVED
    struck = struck_test(sc)
    hist, c = [], None
    for t, a in enumerate(keys):
        hist.append((pos.copy(), vel.copy(), om.copy(), sc.copy()))
        st, _d, pos, vel, om, _s, _ = world.rollout(ball, P, pos, vel, om, fine.PUSHES[a:a + 1], [1], scene=sc)
        if st != OK:
            break
        if struck(sc):
            c = t
            break
    if c is None:
        return None
    speeds = [float(np.linalg.norm(hist[c][1]))]
    s0 = max(0, c - 66)
    pushed = 0
    for _ in range(TRIALS):
        p0, v0, o0, sc0 = hist[s0]
        pos = p0 + rng.normal(0, NUDGE[0], 3) * np.array([1.0, 0.0, 1.0])
        vel = v0 + rng.normal(0, NUDGE[1], 3)
        om, sc = o0.copy(), sc0.copy()
        runs = _nudge_keys(_runs([(a, 1) for a in keys], s0, min(len(keys), c + 264)), rng)
        struck = struck_test(sc)
        hit, ok = None, True
        for a, n in runs:
            for _t in range(n):
                v_before = float(np.linalg.norm(vel))
                st, _d, pos, vel, om, _s, _ = world.rollout(ball, P, pos, vel, om, fine.PUSHES[a:a + 1], [1], scene=sc)
                if st != OK:
                    ok = False
                    break
                if hit is None and struck(sc):
                    hit = v_before
            if not ok:
                break
        if ok and hit is not None and reader(sc)[0] == 1:
            pushed += 1
            speeds.append(hit)
    return dict(impact_speed=_r([min(speeds), max(speeds)]), pushed_over=pushed, trials=TRIALS, rough=True,
                **{"from": "the ball's speed the tick before it strikes the board: the route's, and of %d nudged runs from 66 ticks before that push it over too" % TRIALS})


def _pieces(results) -> list:
    """The route's pieces in order, before anything is replayed: consecutive stretches the follower rolled as one
    (then split at flights), the rest one each. Each: dict(kind, sector, ball, mode, segs, start, tape, trace, end,
    r0, r1 (the rows of the run's trace it covers), ...)."""
    out = []
    for res in results:
        sec = int(res["sector"])
        # the solver keeps one count per break, of its last crossing (a break crossed again after the ball failed past
        # it has an earlier crossing on the route too): only the last crossing of each break gets it
        crossed = {b["no"]: b for b in res.get("breaks", ()) if b.get("arrives") is not None}
        last_of = {sg.get("no"): k for k, sg in enumerate(res["segments"]) if sg["type"] == "cross"}
        left = {b["no"]: b for b in res.get("breaks", ()) if b.get("status") == "unresolved"}
        run: list = []
        before = len(out)

        def piece(kind, segs, **kw):
            tr = np.concatenate([s["trace"] for s in segs]) if segs else np.zeros((0, 9))
            d = dict(kind=kind, sector=sec, segs=segs, ball=segs[0].get("ball", "wood") if segs else None, mode=segs[0].get("mode", "strict") if segs else None,
                     start=segs[0]["start"] if segs else None, tape=[x for s in segs for x in s["tape"]], trace=tr, end=segs[-1]["end"] if segs else None,
                     r0=0, r1=len(tr), fan_bounds=None, **kw)
            fb = [s["fan_bounds"] for s in segs if s.get("fan_bounds")]
            if fb:
                d["fan_bounds"] = dict(least=all(f["least"] for f in fb), most=all(f["most"] for f in fb))
            return d

        def flush():
            if not run:
                return
            p = piece("roll", list(run))
            run.clear()
            tr, i, rows = p["trace"], 0, []
            start = 0
            while i < len(tr):                                    # flights of FLIGHT ticks or more touching nothing
                if tr[i, 3] != 0:
                    i += 1
                    continue
                j = i
                while j + 1 < len(tr) and tr[j + 1, 3] == 0:
                    j += 1
                a, b = tr[max(i - 1, 0)], tr[min(j + 1, len(tr) - 1)]
                if j - i + 1 >= FLIGHT and math.hypot(b[0] - a[0], b[2] - a[2]) >= FLIGHT_ACROSS:
                    if i > start:
                        rows.append(("roll", start, i))
                    rows.append(("flight", i, j + 1))
                    start = j + 1
                i = j + 1
            if start < len(tr):
                rows.append(("roll", start, len(tr)))
            for kind, a, b in rows:
                out.append(dict(p, kind=kind, r0=a, r1=b))

        for si, sg in enumerate(res["segments"]):
            t = sg["type"]
            if t == "follow" and not sg.get("lift") and not sg.get("operates"):
                if len(sg["trace"]):
                    run.append(sg)
                continue
            flush()
            if t == "assumed":
                b = left.get(sg["no"], {})
                out.append(dict(piece("break", []), no=sg["no"], near=b.get("near"), far=b.get("far", sg["to"]), to=sg["to"]))
            elif t == "transform":
                out.append(piece("transform", [sg]))
            elif not len(sg["trace"]):
                continue
            elif t == "cross":
                b = crossed.get(sg.get("no"), {}) if last_of.get(sg.get("no")) == si else {}
                out.append(piece("crossing", [sg], no=sg.get("no"), arrives=b.get("arrives"), nudged=b.get("nudged")))
            elif sg.get("lift"):
                out.append(piece("fan_rise", [sg]))
            else:
                out.append(piece("operate", [sg], operates=sg["operates"]))
        flush()
        if len(out) > before:
            out[-1]["sector_end"] = dict(solved=bool(res.get("solved")))
            out[before]["sector_start"] = dict(rolling=bool(res.get("rolling_start")))
    return out


def build(results: list, info: dict, level: int, world_of, line: str = None, line_sha256: str = None, map_sha256: str = None, seed: int = 0,
          log=print) -> dict:
    """The route file of version 2 as plain data. results: the solver's sector results in order (as for
    `route.reference_line`). world_of(sector, mode): the kernel's world a segment was rolled in (`staged.world_for`)."""
    from .shortcut import _plain
    from .staged import BALLS
    from . import fans as fans_mod, trafo
    _P, _sector, _t, meta, tick_s = route.line_and_ticks(results, info, level)
    pieces = _pieces(results)
    # the rows the line was made of, piece by piece (route.line_and_ticks takes every trace in order up to where it ends)
    row, on_line, n_line = 0, True, len(tick_s)
    after_break = False
    segs_out = []
    by_run: dict = {}
    for i, p in enumerate(pieces):
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        out = dict(i=i, sector=p["sector"], kind=p["kind"], ball=p["ball"], source="kernel")
        if p["kind"] == "break":
            if on_line:
                on_line = False
            out.update(ball=None, s=None, t_ms=None, assumed_rest_start=after_break,
                       **{"break": dict(no=p["no"], near=_r(p["near"]) if p.get("near") is not None else None, far=_r(p["far"]))})
            segs_out.append(out)
            after_break = True
            continue
        if p.get("sector_start") is not None and p["sector"] > 1 and not p["sector_start"]["rolling"]:
            out["rest_start"] = "reset_point"                      # the sector before was not got through: the ball is put at rest at this one's reset point
        elif after_break and segs_out and segs_out[-1]["kind"] == "break":
            out["rest_start"] = "far_side_of_break"
        n = p["r1"] - p["r0"]
        if on_line:
            a, b = row, row + n
            assert b <= n_line, "the pieces have more ticks than the line"
            out["s"] = _r([tick_s[max(a - 1, 0)], tick_s[b - 1]], 2)
            out["t_ms"] = _r([max(a - 1, 0) * route.TICK_MS, (b - 1) * route.TICK_MS], 1)
            row = b
        else:
            rows = p["trace"][p["r0"]:p["r1"], :3]
            out["s"], out["t_ms"] = None, None
            out["own_s"] = _r([0.0, float(np.linalg.norm(np.diff(rows, axis=0), axis=1).sum()) if n > 1 else 0.0], 2)
            out["own_t_ms"] = _r([0.0, n * route.TICK_MS], 1)
        out["assumed_rest_start"] = after_break
        out["ticks"] = int(n)
        world = world_of(p["sector"], p["mode"])
        ball = BALLS[p["ball"]]
        mechs = mechanisms_of(world)
        # the route's states at every cut of this run: the piece's ends and its windows' starts
        key = id(p["segs"][0]) if p["segs"] else None
        if p["kind"] == "transform":                               # no keys: the jump and the new ball, as recorded
            sg = p["segs"][0]
            by_run[key] = {0: tuple(np.array(x, float) for x in sg["start"][:3]) + (_scene(world, sg["start"]),),
                           len(p["trace"]): tuple(np.array(x, float) for x in sg["end"][:3]) + (_scene(world, sg["end"]),)}
        if key not in by_run:
            T = len(p["trace"])
            cuts = {0, T}
            for q in pieces:
                if q["segs"] and id(q["segs"][0]) == key:
                    cuts |= {q["r0"], q["r1"]} | set(_windows(q))
            by_run[key], status = _states_at(world, ball, p["start"], p["tape"], sorted(cuts))
            if p["end"] is not None and T in by_run[key] and np.abs(by_run[key][T][0] - np.asarray(p["end"][0], float)).max() > 1e-6:
                log("   piece %d: its keys run again do not end where the route did" % i)
        st = by_run[key]
        s0, s1 = st.get(p["r0"]), st.get(p["r1"])
        if s0 is None or s1 is None:
            log("   piece %d: the ball was lost running its keys again: no exit" % i)
            segs_out.append(out)
            continue
        out["start"] = dict(pos=_r(s0[0]), vel=_r(s0[1]))
        rng = np.random.default_rng([seed, level, i])
        mech_out = []
        if p["kind"] == "transform":
            sg = p["segs"][0]
            it = world.trafos[sg["which"]] if world.trafos is not None and sg.get("which", -1) >= 0 else next(t for t in trafo.of(info, p["sector"]) if t["name"] == sg["at"])
            out["transform"] = {"at": sg["at"], "pos": _r(it["pos"]), "from": sg["was"], "to": sg["ball"]}
            out["exit"] = None                                     # the new ball appears at rest at the hover point: no envelope
            out["end"] = dict(pos=_r(s1[0]), vel=_r(s1[1]))
        else:
            ends_at_trafo = nxt is not None and nxt["kind"] == "transform" and p["r1"] == len(p["trace"])
            wins = []
            for b0, b1 in _window_pairs(p):
                trials = _trials(world, ball, st[p["r0"] + b0], _runs(p["tape"], p["r0"] + b0, p["r0"] + b1), rng,
                                 may_trigger=ends_at_trafo and b1 == p["r1"] - p["r0"])
                wins.append((b0, b1, trials))
            whole = len(wins) == 1
            out["exit"] = _exit(s1, wins[-1][2] if wins else [], p["kind"] == "flight", FROM_WHOLE if whole else FROM_WINDOW)
            out["corridor"] = _corridor(p["trace"][p["r0"]:p["r1"], :3], wins, FROM_WHOLE if whole else FROM_WINDOWS)
        if p["kind"] == "crossing":
            out["no"] = p.get("no")
            out["arrives"] = p.get("arrives")
            if p.get("nudged"):
                out["arrives_by_nudge"] = {str(k): int(v) for k, v in p["nudged"].items()}
        if p["kind"] == "operate":
            sg = p["segs"][0]
            a = int(sg["tape"][0][0]) if sg["tape"] else 0
            m = next((x for x in mechs if x[0] == sg["operates"]), None)
            opened = None if m is None else bool(m[3](s1[3])[0] and not m[3](s0[3])[0])
            mech_out.append(dict(name=sg["operates"], type="gate", does="push_open", ball=CAN["gate"], key=fine.PUSH_NAMES[a],
                                 hold_ms=round(n * route.TICK_MS, 1), opened=opened))
        if p["kind"] == "fan_rise":
            tr = p["trace"][p["r0"]:p["r1"], :3]
            for f in info.get("placements", ()):
                if f["type"] == fans_mod.TYPE and f.get("sector") == p["sector"]:
                    o = np.array(f["position"], float)
                    if (np.hypot(tr[:, 0] - o[0], tr[:, 2] - o[2]) <= 5.0).any():
                        mech_out.append(dict(name=f["name"], type="fan", does="ride"))
        for name, typ, k, reader in mechs:                         # what the piece changed
            b0, b1 = reader(s0[3]), reader(s1[3])
            if b0[0] == b1[0] or (p["kind"] == "operate" and name == p.get("operates")):
                continue
            e = dict(name=name, type=typ, ball=CAN[typ])
            if typ == "push_board" and b1[0] == 1:
                e["does"] = "push_over"
                keys = [a for a, m in _runs(p["tape"], p["r0"], p["r1"]) for _ in range(m)]
                hit = _impact(world, ball, s0, keys, k, reader, rng)
                if hit:
                    e.update(hit)
            elif typ == "gate" and b1[0] == 1:
                e["does"] = "push_open"
            elif typ == "seesaw":
                e["does"], e["tips"] = "cross", True
            else:
                e["does"] = "state_change"
            e["state"] = [int(b0[0]), int(b1[0])]
            mech_out.append(e)
        out["mechanisms"] = mech_out
        if p.get("fan_bounds"):
            out["fan_bounds"] = p["fan_bounds"]
        if p.get("sector_end") is not None and not p["sector_end"]["solved"] and on_line:
            on_line = False                                        # the line ends with a sector that was not got through
        segs_out.append(out)
    assert not on_line or row == n_line, "the line has %d ticks, the pieces on it %d" % (n_line, row)
    return _plain(dict(version=VERSION, level=int(level), generator="ballistics", line=line, line_sha256=line_sha256, map=info.get("map"),
                       map_sha256=map_sha256, time="game ms", ticks_per_game_second=132, coordinates="world, ball centre",
                       source="trajectories of the Ballistics physics kernel: candidates, not run in the game",
                       nudge=NUDGE_TEXT, keys=dict(zip(fine.PUSH_NAMES, fine.PUSHES.tolist())),
                       routes=[dict(id="standard", complete=bool(meta["complete"]), segments=segs_out)]))


def _window_pairs(p) -> list:
    """The windows a piece is replayed in, as (from, to) ticks into the piece: a crossing or a flight whole; anything
    else in windows of WINDOW ticks, the last one ending at the piece's end (the first may be shorter)."""
    n = p["r1"] - p["r0"]
    if n <= 0:
        return []
    if p["kind"] in ("crossing", "flight") or n <= WINDOW:
        return [(0, n)]
    b = sorted({0} | set(range(n - WINDOW, 0, -WINDOW)) | {n})
    return list(zip(b[:-1], b[1:]))


def _windows(p) -> list:
    """The ticks into the piece's run its windows start at."""
    return [p["r0"] + b0 for b0, _b1 in _window_pairs(p)]


def write(out_dir, level: int, data: dict) -> Path:
    import json
    path = Path(out_dir) / ("L%02d.segments.json" % int(level))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read(path) -> dict:
    """A route file of version 2, checked: the fields every piece has, and its line where it names one."""
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise ValueError("not a route file of version %d: %s" % (VERSION, data.get("version")))
    line = Path(path).parent / data["line"] if data.get("line") else None
    if line is not None and line.exists() and data.get("line_sha256") and file_sha256(line) != data["line_sha256"]:
        raise ValueError("%s is not the line this file was written with" % line)
    for r in data["routes"]:
        for k, sg in enumerate(r["segments"]):
            if sg["i"] != k or sg["kind"] not in KINDS:
                raise ValueError("segment %d: %s" % (k, sg.get("kind")))
    return data


KINDS = ("roll", "crossing", "flight", "fan_rise", "operate", "transform", "break")
