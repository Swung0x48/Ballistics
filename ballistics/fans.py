"""Fans (P_Modul_18): a push straight up on the ball while it is in the fan's box.

A fan has no rigid body: neither its rotor nor its box stops a ball. What it does is add an impulse to the player
ball, once per physics tick, along world +y. The kernel's tick does that (`kernel._step`, and the same line in
`physics.step_reference`); this module reads the numbers from the catalogue (`ballistics.things`, the entry
P_Modul_18 and the shared rule fan_law) and turns a map's fans into what the kernel takes.

The rules, with their grounds as the catalogue marks them:

* While the push is on, every tick adds an impulse of 0.1 at the ball's centre along world +y: dv = 0.1 / mass
  (READ; MEASURED on the original engine's records for wood, 0.0526, and paper, 0.5). In the tick the order is
  damping, then this impulse, then gravity (MEASURED on the original records). Several boxes at once add up (READ;
  20 ticks with two in the records).
* The box: in the fan's own axes, x and z within 1 of the axis, y from 1.14 to 17.14 above the fan's origin (READ).
* On or off: the game tests the ball's mesh box, which turns with the ball, against that box, once per frame, on
  the pose the ball had after the tick before, and the push acts from the next tick on (READ). The kernel does
  not carry the ball's orientation, so it uses a distance instead: the push is on when the ball's centre, as it is
  when the tick begins, is nearer to the box than a distance per ball kind. MEASURED on the original records, as
  the distance that agrees best with which ticks were pushed: wood 2.75 (776 of 795 free-flight ticks near a box
  come out right), paper 3.05 (18,642 of 19,303). The ticks it gets wrong are mostly at the box's edge, where the
  orientation decides.
* Only the current sector's fans push (READ; the records never have a ball in another sector's box).

The edge of the box. Whether a ball at the box's edge is pushed depends on how it is turned, and the kernel
cannot know that: the records hold no orientation, and it cannot be rebuilt from their spin (8 ways tried, none
better than the distance). MEASURED on the original records instead, how the push goes with the distance: paper
is always pushed nearer than 2.0 (11,003 ticks) and never beyond 3.6 (3,564 ticks), half the time at 3.05; wood
always nearer than 2.5, never beyond 3.2. So there are three distances per ball kind: the best one, and the two
bounds ("least": the box's inscribed reach, 1.96 for paper; "most": 3.6). A world can be made with either bound
(`World.of(..., fan_edge=...)`): in free flight the real ball is never lower than the "least" ball nor higher
than the "most" one, which the records bear out (see docs/mechanisms.md).

Not modelled: the fan's two proximity checks (within 80 and within 7 of its axis, horizontally), which delay the
first push after the ball arrives from far away by up to 60 frames and latch the ball that came in; and the stone
ball's distance, for which there is no record (wood's is used: their mesh boxes are nearly the same).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import things

TYPE = "P_Modul_18"


@dataclass(frozen=True)
class Rules:
    """The numbers of the rules above, as the catalogue has them."""
    impulse: float               # per tick, along world +y, at the centre
    half: float                  # the box: |x|, |z| <= half in the fan's axes
    bottom: float                # ... and bottom <= y <= top above the fan's origin
    top: float
    reach: dict                  # ball name -> the centre is nearer to the box than this: pushed
    bounds: dict                 # ball name -> (nearer than this: always pushed, farther than this: never), whatever the orientation
    basis: dict

    @classmethod
    def load(cls) -> "Rules":
        cat = things.load()
        law, fan = cat.rules["fan_law"]["numbers"], cat[TYPE].numbers
        box = law["box"]["v"]
        return cls(impulse=float(law["impulse"]["v"]), half=float(box["half_x_z"]), bottom=float(box["bottom"]), top=float(box["top"]),
                   reach={k: float(v) for k, v in fan["push_distance_by_ball"]["v"].items()},
                   bounds={k: (float(v[0]), float(v[1])) for k, v in fan["push_distance_bounds"]["v"].items()},
                   basis=dict(impulse=law["impulse"]["basis"], box=law["box"]["basis"], reach=fan["push_distance_by_ball"]["basis"]))

    def params(self, ball, edge: str = "best") -> np.ndarray:
        """What the kernel's tick takes: [half, bottom, top, reach, impulse]. edge: which distance stands for the
        box test at the box's edge, where the ball's orientation decides and the kernel does not know it:
        "best" the one that agrees best with the records, "least" pushed only where it always is, "most" pushed
        wherever it ever is. A ball kind without a measured distance (stone) gets wood's: ASSUMED."""
        name = ball.name if ball.name in self.reach else "wood"
        reach = {"best": self.reach[name], "least": self.bounds[name][0], "most": self.bounds[name][1]}[edge]
        return np.array([self.half, self.bottom, self.top, reach, self.impulse])


_RULES = None


def rules() -> Rules:
    global _RULES
    if _RULES is None:
        _RULES = Rules.load()
    return _RULES


def of(info: dict, sector) -> np.ndarray:
    """(F, 12) the fans of a map file that push while `sector` is the current one: origin, then the fan's x, y and
    z axes in world coordinates (unit length). sector None: no fans."""
    out = []
    for p in info.get("placements", ()):
        if p["type"] == TYPE and sector is not None and p.get("sector") == sector:
            axes = [np.array(p[k], float) / np.linalg.norm(p[k]) for k in ("x_axis", "y_axis", "z_axis")]
            out.append(np.r_[np.array(p["position"], float), axes[0], axes[1], axes[2]])
    return np.array(out, np.float64).reshape(-1, 12)


def pushing(pos, fans, par) -> int:
    """How many fans push a ball whose centre is at pos: the plain definition of `kernel._fans_pushing`."""
    half, bottom, top, reach = float(par[0]), float(par[1]), float(par[2]), float(par[3])
    n = 0
    for f in np.asarray(fans, float).reshape(-1, 12):
        d = np.asarray(pos, float) - f[:3]
        lx, ly, lz = float(d @ f[3:6]), float(d @ f[6:9]), float(d @ f[9:12])
        ex, ey, ez = max(abs(lx) - half, 0.0), max(bottom - ly, ly - top, 0.0), max(abs(lz) - half, 0.0)
        if ex * ex + ey * ey + ez * ez < reach * reach:
            n += 1
    return n
