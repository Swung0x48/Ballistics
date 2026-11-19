"""How far the hand-written physics is from the engine, measured on a recorded run.

Two measurements, both starting from the engine's own states and using the keys the run was played with:

* one tick ahead: predict the next velocity, compare with the engine's, by what the ball was touching;
* many ticks ahead with no correction (open loop): where the ball is after 0.1 to 2 game seconds, compared with
  where the engine had it, by what happened on the way (rolling on the floor only, rails, flights).

These numbers are what the width of a proof's tube has to be judged against. They describe the kernel as it is;
nothing here tunes it."""
from __future__ import annotations

import collections
import time

import numpy as np

from . import physics as ph
from .geometry import Mesh
from .trace import Run

TICKS_PER_GAME_SECOND = 132.0


def _kind(mesh: Mesh, contacts) -> str:
    if not contacts:
        return "air"
    rail = [bool(mesh.rail[c.triangle]) for c in contacts]
    if any(abs(c.normal[1]) < 0.5 for c in contacts):
        return "wall"
    if all(rail):
        return "rail"
    return "rail+floor" if any(rail) else "floor"


def _start(mesh: Mesh, run: Run, i: int, P: ph.Params):
    ball = ph.BALLS[int(run.ball_type[i])]
    touching = ph.merge_contacts(mesh.contacts(run.pos[i], ball.radius, P.touch))
    # the run does not record spin: a ball that touches something is taken to roll on it without slipping
    spin = ph.rolling_spin(run.vel[i], touching[0].normal, ball.radius) if touching else np.zeros(3)
    return ball, ph.State(run.pos[i].copy(), run.vel[i].copy(), spin), touching


def one_tick(mesh: Mesh, run: Run, params: ph.Params | None = None, every: int = 1) -> dict:
    """{class: velocity errors (units / physics second)} over the run's ticks. Classes: what the model had the
    ball pressing on after the tick; 'landing' when it touched nothing before; '(impact)' when the engine moved the
    ball off its own straight step, i.e. something was hit inside the tick."""
    P = params or ph.Params()
    t = run.ticks()
    out = collections.defaultdict(list)
    for k in range(0, len(t) - 1, every):
        i0, i1 = t[k], t[k + 1]
        if run.ball_type[i0] != run.ball_type[i1] or run.quarter[i1] < 0:
            continue
        ball, s0, touching = _start(mesh, run, i0, P)
        s1, info = ph.step(mesh, ball, s0, ph.key_push(run.keys_before(i1), int(run.quarter[i1])), P)
        pressed = [c for c, j in zip(info["contacts"], info["normal_impulse"]) if j > 0]
        kind = _kind(mesh, pressed)
        if not touching and kind != "air":
            kind = "landing"
        if np.linalg.norm(run.pos[i0] + run.vel[i0] * ph.DT - run.pos[i1]) > 0.01:
            kind += " (impact)"
        out[kind].append(float(np.linalg.norm(s1.vel - run.vel[i1])))
    return {k: np.array(v) for k, v in out.items()}


def open_loop(mesh: Mesh, run: Run, params: ph.Params | None = None, horizons=(13, 33, 66, 132, 264), every: int = 30, first: int = 200) -> dict:
    """Roll the kernel forward from engine states with the recorded keys and no correction. Returns
    {horizon in ticks: [(position error, distance the engine's ball travelled, what happened on the way)]}, the
    last being 'floor' (rolling on the floor all the way), 'rail' (a rail was touched, no flight) or 'flight'."""
    P = params or ph.Params()
    t = run.ticks()
    top = max(horizons)
    out = {h: [] for h in horizons}
    for k in range(first, len(t) - top - 2, every):
        i0 = t[k]
        if len(set(run.ball_type[t[k:k + top + 1]])) != 1:
            continue
        ball, s, _ = _start(mesh, run, i0, P)
        seen = set()
        for j in range(1, top + 1):
            i = t[k + j]
            seen.add(_kind(mesh, ph.merge_contacts(mesh.contacts(run.pos[i], ball.radius, P.touch))))
            s, _ = ph.step(mesh, ball, s, ph.key_push(run.keys_before(i), int(max(run.quarter[i], 0))), P)
            s = ph.settle(mesh, s, ball.radius)
            if j in out:
                way = "flight" if "air" in seen else ("rail" if seen & {"rail", "rail+floor"} else ("floor" if seen <= {"floor"} else "other"))
                out[j].append((float(np.linalg.norm(s.pos - run.pos[i])), float(np.linalg.norm(run.pos[i] - run.pos[i0])), way))
    return out


def report(mesh: Mesh, run: Run, params: ph.Params | None = None, every: int = 30, log=print) -> dict:
    """Both measurements as text tables; returns the numbers."""
    t0 = time.time()
    ticks = one_tick(mesh, run, params)
    total = sum(len(v) for v in ticks.values())
    log("one tick ahead: velocity error in units / physics second, %d ticks (the ball moves at about 12, gains about 0.2 per tick from a key)" % total)
    log("  %-22s %6s %6s | %8s %8s %8s" % ("the ball was on", "ticks", "share", "median", "90%", "99%"))
    for kind, e in sorted(ticks.items(), key=lambda kv: -len(kv[1])):
        log("  %-22s %6d %5.1f%% | %8.4f %8.4f %8.3f" % (kind, len(e), 100 * len(e) / total, np.median(e), np.percentile(e, 90), np.percentile(e, 99)))
    loop = open_loop(mesh, run, params, every=every)
    log("")
    log("open loop: position error in units after running alone from an engine state (%d starts)" % len(next(iter(loop.values()))))
    log("  %-12s %-8s %5s | %7s %7s %7s | %9s | %s" % ("horizon", "way", "runs", "median", "75%", "90%", "travelled", "within 0.5 / 1 / 2 units"))
    table = {}
    for h, rows in loop.items():
        for way in ("all", "floor", "rail", "flight"):
            e = np.array([r[0] for r in rows if way == "all" or r[2] == way])
            d = np.array([r[1] for r in rows if way == "all" or r[2] == way])
            if not len(e):
                continue
            table[(h, way)] = dict(runs=len(e), median=float(np.median(e)), p75=float(np.percentile(e, 75)), p90=float(np.percentile(e, 90)),
                                   travelled=float(np.median(d)), within=[float(np.mean(e < x)) for x in (0.5, 1.0, 2.0)])
            r = table[(h, way)]
            log("  %4d %5.2f s %-8s %5d | %7.3f %7.2f %7.2f | %9.1f | %3.0f%% %3.0f%% %3.0f%%" % (
                h, h / TICKS_PER_GAME_SECOND, way, r["runs"], r["median"], r["p75"], r["p90"], r["travelled"], *(100 * x for x in r["within"])))
    log("(%.0f s)" % (time.time() - t0))
    return dict(one_tick={k: dict(ticks=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99))) for k, v in ticks.items()},
                open_loop={"%d %s" % k: v for k, v in table.items()})
