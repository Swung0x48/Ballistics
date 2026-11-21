"""The fine level: key tapes found by running the physics kernel.

The coarse level proposes where a ball might get to; this level asks the kernel. From states a ball can have (at
rest on a place the coarse level reaches), it searches over which arrow keys to hold for how long until the ball
is rolling on a target place, and returns the keys and the trajectory.

What a result is worth: it is a trajectory of the hand-written kernel, not of the game. The kernel's measured
error grows with time and at impacts (docs/physics.md, calibration tables), and a search finds whatever the kernel allows,
its weak spots included. So every crossing comes with what it leans on (impacts, several contacts at once, flight)
and how often it survives small changes of the start and of the key timing. It stays a *candidate*
until the engine has run it. "Not found" means not found within the stated budget, never "impossible".

The search keeps one state per cell of (position, heading, speed, on the ground or not) and grows from them:
pick a kept state, hold one of the nine pushes for a while, keep what is new or quicker. Half the picks go to the
states nearest the target, half to any, both preferring the less tried.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import physics as ph
from . import trafo
from .geometry import CONTACT_MARGIN
from .kernel import OK, TRIGGER, World

# The nine pushes the arrow keys can give: each held key is one unit push along a camera axis, and the camera's
# four quarter turns map the axes onto each other, so the set is the same for every camera position.
PUSHES = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)], np.float64)
PUSH_NAMES = ("-", "+x", "-x", "+z", "-z", "+x+z", "+x-z", "-x+z", "-x-z")
CHUNK = 8                       # ticks between kept states
HOLDS = (1, 2, 4, 8)            # how many chunks one pick holds its push
CELL = 1.0
WALK_NY = 0.6
SPEED_EDGES = (0.5, 3.0, 8.0, 13.0, 18.0)


@dataclass
class Crossing:
    found: bool
    expansions: int
    seconds: float
    kept: int
    best_distance: float                 # how near the target the search got
    ticks: int = 0
    tape: list = field(default_factory=list)      # [(push index, ticks)]
    trace: np.ndarray | None = None               # ball centre after every tick
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    stats: dict = field(default_factory=dict)
    nearest: np.ndarray | None = None             # the kept state nearest the target when nothing was found
    root: int = -1                                # which start it left from (the index `Search.add_start` returned)
    end_state: tuple | None = None                # (pos, vel, omega) at the end
    # the crossing as stretches, in order: dict(type "cross", ball, start, tape, trace, end) and, where the ball went
    # through a transformer, dict(type "transform", ball (the new kind), was, at, which, start, tape [], trace, end).
    # `tape` above is the whole crossing's only when it is one stretch; `trace` is all of them joined
    parts: list = field(default_factory=list)
    end_ball: str = ""

    def keys(self) -> str:
        return " ".join("%s×%d" % (PUSH_NAMES[a], n) for a, n in self.tape)


def guide(graph, goal: int) -> np.ndarray:
    """How far each coarse place is from the goal place, counted generously (`coarse.Relaxed`). It only steers the
    search, so being generous costs time, never correctness: wherever the coarse level fails to connect, the search
    is still drawn across and the kernel decides."""
    from .coarse import Relaxed
    return Relaxed(graph).togo(goal)


def rest_state(world: World, x: float, z: float, from_y: float) -> ph.State:
    """A ball at rest on whatever is under (x, from_y, z)."""
    y = world.mesh.rest(x, z, from_y)
    if math.isnan(y):
        raise ValueError("nothing under (%.2f, %.2f)" % (x, z))
    return ph.State(np.array([x, y + CONTACT_MARGIN, z]), np.zeros(3), np.zeros(3))


class Search:
    def __init__(self, world: World, targets, ball: ph.Ball = ph.WOOD, params: ph.Params | None = None, reach: float = 1.2, seed: int = 0,
                 places=None, togo=None, transforms: bool = False):
        """targets: the places to arrive on. places + togo (from `guide`), if given, steer the search by how far
        each place still is from the goal; without them it is steered by the straight distance to the targets.
        ball: the kind the starts are. transforms: a state that comes within a wood or stone transformer of another
        kind is transformed (`ballistics.trafo`) and the search goes on from the new ball; without it, as with a
        paper transformer, that is an end."""
        self.w, self.ball, self.P = world, ball, params or ph.Params()
        self.transforms = bool(transforms) and world.trafos is not None
        self.balls = {b.name: b for b in (ph.WOOD, ph.STONE, ph.PAPER)}
        self.kind, self.jump = [], []            # per kept state: the ball's kind; the transformer it went through since its parent (-1: none)
        self.scene = []                          # per kept state: the loose balls' state (shared with its parent while nothing moved them)
        self._rest = world.scene()
        self.T = np.ascontiguousarray(targets, np.float64).reshape(-1, 3)
        self.reach = reach
        self.G = None
        if places is not None:
            ok = np.isfinite(togo)
            self.G, self.togo = np.ascontiguousarray(places, np.float64)[ok], np.asarray(togo, np.float64)[ok]
            self._ggrid: dict = {}
            for i, p in enumerate(self.G):
                self._ggrid.setdefault((int(p[0] // 6), int(p[2] // 6)), []).append(i)
        self.rng = np.random.default_rng(seed)
        # every state ever kept (never changed afterwards: later states hang on them by `parent`)
        self.pos, self.vel, self.omega = [], [], []
        self.parent, self.act, self.hold, self.ticks = [], [], [], []
        # one slot per cell: the quickest state known for it, how near the target it is, how often it was picked
        self.cell: dict = {}
        self.slot, self.h, self.visits, self.ground = [], [], [], []
        self.goal = -1
        self.expansions = 0
        self._tgrid: dict = {}
        for i, p in enumerate(self.T):
            self._tgrid.setdefault((int(p[0] // 4), int(p[2] // 4)), []).append(i)

    # ---- the target
    def distance(self, p) -> float:
        """Distance from p to the nearest target place."""
        i, j = int(p[0] // 4), int(p[2] // 4)
        near = [k for di in (-2, -1, 0, 1, 2) for dj in (-2, -1, 0, 1, 2) for k in self._tgrid.get((i + di, j + dj), ())]
        if near:
            d = float(np.sqrt(((self.T[near] - p) ** 2).sum(1)).min())
            if d <= 8.0:
                return d                           # nothing outside the cells looked at can be nearer
        return float(np.sqrt(((self.T - p) ** 2).sum(1)).min())

    def score(self, p) -> float:
        """What the search steers by: lower is nearer the goal."""
        if self.G is None:
            return self.distance(p)
        i, j = int(p[0] // 6), int(p[2] // 6)
        near = [k for di in (-1, 0, 1) for dj in (-1, 0, 1) for k in self._ggrid.get((i + di, j + dj), ())]
        if not near:
            return float((3.0 * np.sqrt(((self.G - p) ** 2).sum(1)) + self.togo).min())
        return float((3.0 * np.sqrt(((self.G[near] - p) ** 2).sum(1)) + self.togo[near]).min())

    def arrived(self, pos, vel, stats) -> bool:
        """Rolling on a target place: carried (by a rail, or by a face under it) at a target place's height and spot."""
        return bool(stats[8] > 0 and self.distance(pos) <= self.reach)

    # ---- the kept states
    def _key(self, pos, vel, stats, kind, scene):
        hs = math.hypot(vel[0], vel[2])
        heading = 8 if hs < 1.5 else int(((math.atan2(vel[2], vel[0]) + math.pi) / (math.pi / 4)) % 8)
        speed = sum(hs >= e for e in SPEED_EDGES)
        # where the loose balls are, to the unit, is part of which state this is (0 while none has been moved)
        moved = 0 if scene is self._rest or not len(scene) else hash(np.floor(scene[:, :3]).astype(np.int64).tobytes())
        return (int(pos[0] // CELL), int(pos[1] // CELL), int(pos[2] // CELL), heading, speed, int(stats[5] > 0), kind, moved)

    def _keep(self, pos, vel, omega, parent, act, hold, ticks, stats, kind=None, jump: int = -1, scene=None) -> int:
        """Keep the state if its cell is new or it got there sooner; returns its index, or -1."""
        kind = kind or self.ball.name
        scene = self._rest if scene is None else scene
        key = self._key(pos, vel, stats, kind, scene)
        k = self.cell.get(key)
        if k is not None and self.ticks[self.slot[k]] <= ticks:
            return -1
        i = len(self.pos)
        for a, v in ((self.pos, pos.copy()), (self.vel, vel.copy()), (self.omega, omega.copy()), (self.parent, parent), (self.act, act),
                     (self.hold, hold), (self.ticks, ticks), (self.kind, kind), (self.jump, jump), (self.scene, scene)):
            a.append(v)
        if k is None:
            self.cell[key] = len(self.slot)
            for a, v in ((self.slot, i), (self.h, self.score(pos)), (self.visits, 0), (self.ground, bool(stats[5] > 0))):
                a.append(v)
        else:
            self.slot[k], self.h[k] = i, self.score(pos)
        return i

    def add_start(self, s: ph.State, grounded: bool = True, scene=None) -> int:
        """A state to grow from. Returns its index (the `root` of a crossing that leaves from it), or -1 if another
        start already holds its cell. scene: the loose balls' state there (at rest where they lie, without it)."""
        st = np.zeros(10)
        st[5], st[6], st[8] = float(grounded), 1.0, float(grounded)
        if scene is not None and (len(scene) != len(self._rest) or np.array_equal(scene, self._rest)):
            scene = None
        return self._keep(np.array(s.pos, float), np.array(s.vel, float), np.array(s.omega, float), -1, 0, 0, 0, st,
                          scene=None if scene is None else np.array(scene, float))

    def _pick(self) -> int:
        n = len(self.slot)
        if self.rng.random() < 0.5 and len(self._front) > 0:
            pool = self._front[self.rng.integers(0, len(self._front), 6)]
        else:
            pool = self.rng.integers(0, n, 6)
        return int(min(pool, key=lambda k: self.visits[k]))

    def run(self, max_expansions: int = 200_000, max_seconds: float = 120.0) -> Crossing:
        t0 = time.time()
        self._front = np.zeros(0, np.int64)
        zero = np.zeros((1, 3))
        while self.expansions < max_expansions and time.time() - t0 < max_seconds and self.goal < 0:
            if self.expansions % 256 == 0:           # the front: the cells on the ground nearest the target
                on = np.flatnonzero(np.array(self.ground))
                h = np.array(self.h)[on]
                self._front = on[np.argsort(h)[:max(32, len(h) // 20)]]
            self.expansions += 1
            k = self._pick()
            self.visits[k] += 1
            i = self.slot[k]
            a = self.act[i] if (self.parent[i] >= 0 and self.rng.random() < 0.4) else int(self.rng.integers(0, len(PUSHES)))
            pos, vel, omega, parent, ticks, held = self.pos[i], self.vel[i], self.omega[i], i, self.ticks[i], 0
            kind = self.kind[i]
            ball = self.balls[kind]
            before = self.scene[i]                 # the loose balls: a copy to roll on, kept only if it changed
            sc = before.copy()
            for _ in range(HOLDS[int(self.rng.integers(0, len(HOLDS)))]):
                status, done, pos, vel, omega, stats, _ = self.w.rollout(ball, self.P, pos, vel, omega, PUSHES[a:a + 1], [CHUNK], scene=sc)
                if len(sc) and not np.array_equal(sc, before):
                    before = sc.copy()
                if status == TRIGGER and self.transforms and self.w.trafos[int(stats[9])]["kind"] in trafo.SOLVED:
                    # transformed: the state kept is the new ball's, one tick after it appeared; this pick ends here
                    which = int(stats[9])
                    new, tr, end, st1, s1 = trafo.stretch(self.w, self.w.trafos[which], self.P, scene=sc)
                    if s1 == OK:
                        self._keep(end[0], end[1], end[2], parent, a, held + done, ticks + done + len(tr), st1, kind=new.name, jump=which,
                                   scene=sc.copy() if len(sc) and not np.array_equal(sc, self._rest) else None)
                    break
                if status != OK:
                    break
                ticks += done
                held += done
                j = self._keep(pos, vel, omega, parent, a, held, ticks, stats, kind=kind, scene=before)
                if j < 0:
                    continue                       # nothing new at this point of the hold; it may still lead somewhere new
                parent, held = j, 0
                if self.arrived(pos, vel, stats):
                    # it has to stay: let it roll on with no keys for half a physics second
                    s2, _d, p2, _v, _o, st2, _ = self.w.rollout(ball, self.P, pos, vel, omega, zero, [33], scene=sc.copy())
                    if s2 == OK and self.distance(p2) <= self.reach + 2.0:
                        self.goal = j
                        break
        return self._result(time.time() - t0)

    def tape_of(self, i: int):
        """(the start the state i hangs on, its key tapes): one tape per stretch, [(push, ticks)], and for every
        stretch but the last the transformer it ends in."""
        chain = []
        while self.parent[i] >= 0:
            chain.append((self.act[i], self.hold[i], self.jump[i]))
            i = self.parent[i]
        chain.reverse()
        tapes, tape = [], []
        for a, n, jump in chain:
            if tape and tape[-1][0] == a:
                tape[-1][1] += n
            else:
                tape.append([a, n])
            if jump >= 0:
                tapes.append(([(x, m) for x, m in tape], jump))
                tape = []
        tapes.append(([(x, m) for x, m in tape], -1))
        return i, tapes

    def _result(self, seconds: float) -> Crossing:
        h = np.where(np.array(self.ground), np.array(self.h), np.inf)
        best = int(h.argmin())
        if self.goal < 0:
            return Crossing(False, self.expansions, seconds, len(self.slot), float(h[best]), nearest=self.pos[self.slot[best]].copy())
        root, tapes = self.tape_of(self.goal)
        state, kind, parts, out = (self.pos[root], self.vel[root], self.omega[root], self.scene[root].copy()), self.kind[root], [], None
        for tape, jump in tapes:
            sc = state[3].copy()
            out = replay(self.w, self.balls[kind], self.P, ph.State(state[0], state[1], state[2]), tape, scene=sc)
            parts.append(dict(type="cross", ball=kind, start=state, tape=tape, trace=out["trace"], end=(out["pos"], out["vel"], out["omega"], sc)))
            if jump >= 0:
                if out["status"] != TRIGGER or out["ticks"] != sum(n for _a, n in tape):
                    raise RuntimeError("the key tape does not reproduce the transformation the search went through")
                sc = sc.copy()
                new, tr, end, _st, _s = trafo.stretch(self.w, self.w.trafos[jump], self.P, scene=sc)
                parts.append(dict(type="transform", ball=new.name, was=kind, at=self.w.trafos[jump]["name"], which=jump, start=parts[-1]["end"],
                                  tape=[], trace=tr, end=end))
                state, kind = end, new.name
        ticks = int(sum(len(p["trace"]) for p in parts))
        if ticks != self.ticks[self.goal] or np.abs(out["pos"] - self.pos[self.goal]).max() > 1e-6:
            raise RuntimeError("the key tape does not reproduce the state the search arrived at")
        return Crossing(True, self.expansions, seconds, len(self.slot), self.distance(self.pos[self.goal]), ticks=ticks,
                        tape=parts[0]["tape"] if len(parts) == 1 else [], trace=np.vstack([p["trace"] for p in parts]),
                        start=self.pos[root].copy(), end=out["pos"], stats=out["stats"], root=root,
                        end_state=parts[-1]["end"], parts=parts, end_ball=kind)


def replay(world: World, ball: ph.Ball, P: ph.Params, start: ph.State, tape, scene=None) -> dict:
    """Run a key tape from a state: the trajectory and what it met. scene: the loose balls' state at the start,
    changed in place."""
    pushes = np.array([PUSHES[a] for a, _ in tape]).reshape(-1, 3)
    holds = [n for _, n in tape]
    status, done, pos, vel, omega, st, trace = world.rollout(ball, P, start.pos, start.vel, start.omega, pushes, holds, trace=True, scene=scene)
    return dict(status=status, ticks=done, pos=pos, vel=vel, omega=omega, trace=trace,
                stats=dict(impact_ticks=int(st[0]), edge_ticks=int(st[1]), multi_contact_ticks=int(st[2]), air_ticks=int(st[3]),
                           top_speed=float(st[4]), touching_at_end=bool(st[5] > 0)))


def robustness(search: Search, crossing: Crossing, trials: int = 60, shift: float = 0.25, jitter: int = 2, seed: int = 1) -> dict:
    """How often the crossing still arrives when the ball starts up to `shift` away (at rest on the same floor) and
    every key change comes up to `jitter` ticks early or late. A crossing that only works to the tick is a find
    about the kernel, not about the level."""
    rng = np.random.default_rng(seed)
    w, ball, P = search.w, search.ball, search.P
    ok = 0
    for _ in range(trials):
        dx, dz = rng.uniform(-shift, shift, 2)
        try:
            s = rest_state(w, crossing.start[0] + dx, crossing.start[2] + dz, crossing.start[1] + 1.0)
        except ValueError:
            continue
        if abs(s.pos[1] - crossing.start[1]) > 0.5:
            continue                                 # not the same floor any more: counts as a failure
        pos, vel, omega = s.pos, s.vel, s.omega
        good = False
        for a, n in crossing.tape:
            n = max(1, n + int(rng.integers(-jitter, jitter + 1)))
            status, _d, pos, vel, omega, st, _ = w.rollout(ball, P, pos, vel, omega, PUSHES[a:a + 1], [n])
            if status != OK:
                break
        else:
            good = search.arrived(pos, vel, st)
            if not good:                             # a little early or late: give it the half second the search gives
                status, _d, pos, vel, omega, st, _ = w.rollout(ball, P, pos, vel, omega, np.zeros((1, 3)), [33])
                good = status == OK and search.distance(pos) <= search.reach + 2.0 and st[5] > 0
        ok += good
    return dict(trials=trials, arrived=int(ok), shift=shift, jitter=jitter)


def describe(c: Crossing) -> str:
    """What kind of crossing it is, from what the trajectory met."""
    if not c.found:
        return "not found"
    s = c.stats
    if s["air_ticks"] <= 6 and s["impact_ticks"] == 0:
        return "rolling only"
    return "with flight or impact"
