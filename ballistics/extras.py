"""Score balls and life balls (P_Extra_Point, P_Extra_Life) along a route.

They have no rigid body: the ball goes through them, and nothing here changes how the ball moves. What a route gets
from them is points and lives, and that is decided by where the ball's centre is, frame by frame. This module walks
along the positions a route already has (one per physics tick) and says which ones it collects.

Every number is read from the catalogue (`ballistics.things`): the entry P_Extra, the entry level_flow and the
shared rule sector_rebuild. The rules, with their grounds as the catalogue marks them:

* A score ball is collected when the ball's centre is nearer than 3 to the place it was put at, a life ball
  nearer than 4.5 (3D, strictly). MEASURED on the original engine's records: of 196 score balls collected, in the
  166 where the record has the ball's position per frame the points come in the first frame the centre is inside
  3 or in the next one; 43 life balls, the life comes 0 to 2 frames after the first frame inside 4.5. So the tick
  given here is good to one frame, and a path that only grazes the distance is marked `marginal`.
* Only the current sector's balls are there (READ). The current sector changes at a checkpoint: when the centre is
  nearer than 6.5 to its flame, which is 1.4948 above the checkpoint's place (READ; the height MEASURED on the
  rebuilt engine). The old sector's balls go 2 frames after that frame and the next sector's are in place 6
  frames after it (MEASURED on the rebuilt engine at a death; at a checkpoint DERIVED), their detection opening 1
  to 2 frames later (READ): here they count from the 8th frame. The next checkpoint counts from the 12th (DERIVED).
  The solver's own split into sectors is not used for this: a sector's stretch can end short of the flame.
  The records agree on the checkpoint frame (100 of 100, 0 to 2 frames), but they never come within another
  sector's ball nor collect anything sooner than 446 frames after a checkpoint: "only the current sector's" and
  the frame counts rest on the scripts and the rebuilt engine alone. In the 13 levels no ball is nearer than 27
  beyond its distance to where its sector begins, so the frame counts cannot matter there.
* A score ball gives 100 at once and sends 6 small balls after the ball, 20 each when they arrive: 220 in all
  (MEASURED). The small balls can take long to arrive when the ball moves (1.97 to 13.80 game seconds for all six
  in the original records) and are lost when the sector is taken down first. That does not bear on the route:
  pausing with Esc before the checkpoint lets them arrive while the game's clock stands (MEASURED on the original
  records: 31 of 39 pauses with small balls out collected all of them, the others were too short). The project's
  owner settled it: the small balls are collected with Esc. So every score ball collected counts 220 here, and
  the pause is left to whoever runs the route.
* A life ball gives one life. It is shown only while the ball is within 60 of it (READ), which is checked every
  5 to 20 frames; that is not modelled: a ball cannot get from 60 to 4.5 in 20 frames.

Not modelled: death (a collected life ball comes back, a score ball does not), the level's score as a whole, and
anything the search could do to go and get one: the solver does not aim at them, it reports what its routes pass.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from . import things

TICKS_PER_GAME_SECOND = 132.0


@dataclass(frozen=True)
class Rules:
    """The numbers of the rules above, as the catalogue has them."""
    point_radius: float
    life_radius: float
    core: int                    # points for the score ball itself
    follower: int                # points for each small ball
    followers: int               # how many small balls
    checkpoint_radius: float
    flame_height: float
    off_after: int               # frames after the checkpoint frame: the old sector's balls are gone
    on_after: int                # ... the next sector's balls can be collected
    armed_after: int             # ... the next checkpoint counts
    basis: dict                  # per name above: the catalogue's basis

    @classmethod
    def load(cls) -> "Rules":
        cat = things.load()
        ex, flow, sec = cat["P_Extra"].numbers, cat["level_flow"].numbers, cat.rules["sector_rebuild"]["numbers"]
        score = ex["point_score"]["v"]
        core, each = int(score["内核"]), int(score["每个小球"])
        call = int(sec["checkpoint_call"]["v"])
        return cls(point_radius=float(ex["point_pickup_radius"]["v"]), life_radius=float(ex["life_pickup_radius"]["v"]),
                   core=core, follower=each, followers=(int(score["合计"]) - core) // each,
                   checkpoint_radius=float(flow["checkpoint_radius"]["v"]), flame_height=float(flow["flame_height"]["v"]),
                   off_after=call + int(sec["props_torn_down"]["v"]),
                   on_after=call + int(sec["props_built"]["v"]) + int(max(ex["detect_open_frames"]["v"])),
                   armed_after=int(sec["next_checkpoint_armed"]["v"]),
                   basis=dict(point_radius=ex["point_pickup_radius"]["basis"], life_radius=ex["life_pickup_radius"]["basis"],
                              score=ex["point_score"]["basis"], checkpoint_radius=flow["checkpoint_radius"]["basis"],
                              flame_height=flow["flame_height"]["basis"], off_after=sec["props_torn_down"]["basis"],
                              on_after=sec["props_built"]["basis"], armed_after=sec["next_checkpoint_armed"]["basis"]))


KINDS = {"P_Extra_Point": "point", "P_Extra_Life": "life"}


def items(info: dict, rules: Rules) -> list:
    """The score and life balls of a map file: [dict(name, kind, sector, pos, radius)]. One that is in no sector
    has sector None and is never collected here (it is listed, see `Walk.result`)."""
    out = []
    for p in info.get("placements", ()):
        kind = KINDS.get(p["type"])
        if kind:
            out.append(dict(name=p["name"], kind=kind, sector=p.get("sector"), pos=np.array(p["position"], float),
                            radius=rules.point_radius if kind == "point" else rules.life_radius))
    return out


class Walk:
    """The state that decides what is collected, carried along a route: the current sector, a checkpoint that was
    just passed, what was already taken. `feed` takes the ball's centre after each frame, in order; with one
    physics tick per frame (132 fps, what routes are evaluated at) a frame is a tick."""

    def __init__(self, info: dict, sector: int = 1, rules: Rules | None = None, fps: float = TICKS_PER_GAME_SECOND):
        self.rules = rules or Rules.load()
        self.items = items(info, self.rules)
        cp = np.array(info.get("checkpoints") or np.zeros((0, 3)), float).reshape(-1, 3)
        self.flames = cp + np.array([0.0, self.rules.flame_height, 0.0])
        self.last_sector = len(self.flames) + 1
        self.fps = float(fps)
        self.sector = int(sector)
        self.frame = 0
        self.pending = None           # (frame the old sector's balls go, frame the next sector's count)
        self.armed = 0                # the frame from which the current sector's checkpoint counts
        self.taken: dict = {}         # name -> its event
        self.events: list = []
        self.closest: dict = {}       # name -> [distance, frame, the ball's travel in that frame], while it could be collected
        self.inactive: dict = {}      # name -> [distance, frame, the sector that was current]: inside its distance while not there
        self.flame_closest: dict = {} # checkpoint number -> the nearest the centre came to its flame while it counted
        self.sector_off: dict = {}    # sector -> the frame its balls went
        self.frames_in: dict = {}     # sector -> for how many frames it was the current one
        self.left_break = False
        self._prev = None

    def feed(self, positions) -> None:
        R = self.rules
        where = np.array([it["pos"] for it in self.items], float).reshape(-1, 3)
        reach = np.array([it["radius"] for it in self.items], float)
        sector_of = np.array([-1 if it["sector"] is None else it["sector"] for it in self.items], np.int64)
        for p in np.asarray(positions, float).reshape(-1, 3):
            f = self.frame
            if self.pending is not None and f >= self.pending[1]:
                self.sector += 1
                self.pending = None
            active = self.sector if self.pending is None or f < self.pending[0] else None
            step = float(np.linalg.norm(p - self._prev)) if self._prev is not None else 0.0
            if self.pending is None and self.sector < self.last_sector and f >= self.armed:
                d = float(np.linalg.norm(p - self.flames[self.sector - 1]))
                if d < self.flame_closest.get(self.sector, np.inf):
                    self.flame_closest[self.sector] = d
                if d < R.checkpoint_radius:
                    self.pending = (f + R.off_after, f + R.on_after)
                    self.armed = f + R.armed_after
                    self.sector_off[self.sector] = f + R.off_after
                    self.events.append(dict(kind="checkpoint", number=self.sector, frame=f, distance=d, after_left_break=self.left_break))
            if active is not None:
                self.frames_in[active] = self.frames_in.get(active, 0) + 1
            if len(where):
                dist = np.sqrt(((where - p) ** 2).sum(1))
                there = sector_of == (-2 if active is None else active)
                for k in np.flatnonzero(there | (dist < reach)):
                    it, d = self.items[k], float(dist[k])
                    if there[k]:
                        c = self.closest.get(it["name"])
                        if c is None or d < c[0]:
                            self.closest[it["name"]] = [d, f, step]
                        if d < it["radius"] and it["name"] not in self.taken:
                            ev = dict(kind=it["kind"], name=it["name"], sector=it["sector"], frame=f, distance=d, after_left_break=self.left_break)
                            self.taken[it["name"]] = ev
                            self.events.append(ev)
                    elif it["name"] not in self.taken:
                        c = self.inactive.get(it["name"])
                        if c is None or d < c[0]:
                            self.inactive[it["name"]] = [d, f, self.sector]
            self._prev = p
            self.frame += 1

    def result(self, res: dict) -> dict:
        """Walk a sector result's stretches (as `staged.solve_sector` and the other routes hold them) and say what
        they collect. What comes after a break that was left is walked too, but marked and not counted: the ball
        was put there, the route does not reach it."""
        R = self.rules
        f0, n0 = self.frame, len(self.events)
        before = set(self.taken)
        self.closest, self.inactive, self.frames_in = {}, {}, {}      # what this result's stretches pass, not the ones before
        for sg in res["segments"]:
            if sg["type"] == "assumed":
                self.left_break = True
                self._prev = None
                continue
            if len(sg["trace"]):
                self.feed(sg["trace"][:, :3])
        new = self.events[n0:]
        main = max(self.frames_in, key=self.frames_in.get) if self.frames_in else None     # the sector these stretches are mostly in
        rows = []
        for it in self.items:
            if it["name"] in before or it["name"] not in self.closest:
                continue                                 # taken earlier, or its sector was never the current one on these stretches
            ev = self.taken.get(it["name"])
            c = self.closest[it["name"]]
            row = dict(name=it["name"], kind=it["kind"], sector=it["sector"], radius=it["radius"], collected=ev is not None,
                       closest=c[0], marginal=bool(abs(it["radius"] - c[0]) < c[2]))
            if not row["collected"] and not row["marginal"] and it["sector"] != main:
                continue                                 # a ball of the next sector, seen only in the few frames after the checkpoint
            if row["collected"]:
                row.update(tick=ev["frame"] - f0, game_seconds=(ev["frame"] - f0) / self.fps, distance=ev["distance"], after_left_break=ev["after_left_break"])
                if it["kind"] == "point":
                    row.update(points=R.core + R.follower * R.followers)      # its small balls included: Esc before the checkpoint
            rows.append(row)
        rows.sort(key=lambda r: (not r["collected"], r.get("tick", 0), r["closest"]))
        got = [r for r in rows if r["collected"] and not r["after_left_break"]]
        pts = [r for r in got if r["kind"] == "point"]
        cps = [dict(number=e["number"], tick=e["frame"] - f0, distance=e["distance"], after_left_break=e["after_left_break"]) for e in new if e["kind"] == "checkpoint"]
        return dict(extras=rows, game_sector=main,
                    points=sum(r["points"] for r in pts),
                    lives=sum(r["kind"] == "life" for r in got),
                    checkpoints=cps, game_sector_at_end=self.sector,
                    flame_closest={int(k): float(v) for k, v in self.flame_closest.items()},
                    inactive=[dict(name=n, distance=v[0], tick=v[1] - f0, current_sector=v[2]) for n, v in self.inactive.items() if v[1] >= f0],
                    unassigned=[it["name"] for it in self.items if it["sector"] is None],
                    basis=dict(R.basis))


def annotate(results: list, info: dict) -> None:
    """Give every route of the sector results (in sector order, as the solver made them) its `extras`: the standard
    route of each sector, walked on from the one before when the ball rolled on from it, and each other route of
    a sector from the state its standard route began with."""
    walk, last = None, None
    for res in sorted(results, key=lambda r: r["sector"]):
        if walk is None or not res.get("rolling_start") or res["sector"] != last + 1:
            walk = Walk(info, sector=res["sector"])
        start = copy.deepcopy(walk)
        res["extras"] = walk.result(res)
        for a in res.get("alternatives", ()):
            if a.get("found"):
                a["route"]["extras"] = copy.deepcopy(start).result(a["route"])
        last = res["sector"]


def lines(ex: dict) -> list:
    """A route's `extras` in words."""
    out = []
    for r in ex["extras"]:
        what = "score ball" if r["kind"] == "point" else "life ball"
        if r["collected"]:
            s = "   %s %s: collected at %.1f game s (centre %.3f from it, limit %g; closest %.2f%s)" % (
                what, r["name"], r["game_seconds"], r["distance"], r["radius"], r["closest"], ", MARGINAL: within one tick's travel of the limit" if r["marginal"] else "")
            if r["kind"] == "point":
                s += "; %d points" % r["points"]
            if r["after_left_break"]:
                s += " [after a break that was left: not counted]"
        else:
            s = "   %s %s: not collected, passes at %.2f (limit %g)%s" % (what, r["name"], r["closest"], r["radius"],
                                                                        ", MARGINAL: within one tick's travel of the limit" if r["marginal"] else "")
        out.append(s)
    if ex["extras"]:
        out.insert(0, "   on the way: %d points from score balls (their small balls included: Esc before the checkpoint), %d life ball%s" % (
            ex["points"], ex["lives"], "" if ex["lives"] == 1 else "s"))
    for i in ex["inactive"]:
        out.append("   %s: the centre comes within its distance (%.2f) while sector %s is current: not there, not collected" % (i["name"], i["distance"], i["current_sector"]))
    if ex["unassigned"]:
        out.append("   in no sector, left out: " + ", ".join(ex["unassigned"]))
    return out
