"""Transformers (P_Trafo_Wood, P_Trafo_Stone, P_Trafo_Paper): the ball becomes another kind of ball.

A transformer has no rigid body. It acts by distance alone, and what it does is a jump in the route's state: the
ball's kind, place and speed are replaced, and time passes. This module reads the numbers from the catalogue
(`ballistics.things`, the entry P_Trafo and the shared rule transform_timing) and holds the rule; the kernel only
reports the tick at which a transformer is triggered (`kernel.TRIGGER`).

The rules, with their grounds as the catalogue marks them:

* Every frame the game takes the transformer nearest to the ball's centre (3D, to the transformer's origin). If
  that one is nearer than 4.3 and of another kind than the ball, the ball is transformed. A nearest one of the
  ball's own kind does nothing, and keeps a farther one of another kind from acting (READ in the script; the
  distance MEASURED on the rebuilt engine).
* With T the frame in which the game detects it: the ball stays a rigid body for 2 more ticks, then it is gone;
  at T+332 the new ball is there, at the transformer's origin plus 3 along the transformer's own y axis, at rest,
  and gravity acts on it from that tick; from T+333 the keys act (MEASURED on the retail game at 132 fps and on
  the original engine's records, see docs/mechanisms.md). The speed the ball came in with is not kept.
  T is the frame the centre is first inside, or the one after: of the 5 transformations in the 132 fps records, 1
  and 4. The solver takes the later one: the new ball 333 ticks after the first tick inside, the keys from the
  334th.

What is in place: the transformers of the sector that is current. The game moves a few transformer objects
between the rows that use them and leaves each where it was last put, hidden or not, so one of another sector can
be in place too; which, depends on the order the sectors were built in, and is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import things

KINDS = {"P_Trafo_Wood": "wood", "P_Trafo_Stone": "stone", "P_Trafo_Paper": "paper"}
SOLVED = ("wood", "stone")       # the kinds a route can go on as: the paper ball is not a sphere in the game and is not solved yet


@dataclass(frozen=True)
class Rules:
    """The numbers of the rules above, as the catalogue has them."""
    radius: float                # the centre nearer than this to the transformer's origin: triggered
    gone_after: int              # ticks after T: the old ball is no longer a rigid body
    body_after: int              # ticks after T: the new ball is there, at rest
    control_after: int           # ticks after T: the keys act
    hover: float                 # the new ball's place: this far along the transformer's own y axis from its origin
    basis: dict

    @classmethod
    def load(cls) -> "Rules":
        n = things.load().rules["transform_timing"]["numbers"]
        return cls(radius=float(n["radius"]["v"]), gone_after=int(n["old_body_removed"]["v"]), body_after=int(n["new_body"]["v"]),
                   control_after=int(n["control"]["v"]), hover=float(n["hover_height"]["v"]),
                   basis={k: n[k]["basis"] for k in ("radius", "old_body_removed", "new_body", "control", "hover_height")})


_RULES = None


def rules() -> Rules:
    global _RULES
    if _RULES is None:
        _RULES = Rules.load()
    return _RULES


def of(info: dict, sector=None) -> list:
    """The transformers of a map file: [dict(name, kind, sector, pos, up)], `up` the transformer's own y axis (unit
    length). With a sector: only that sector's."""
    out = []
    for p in info.get("placements", ()):
        kind = KINDS.get(p["type"])
        if kind and (sector is None or p.get("sector") == sector):
            up = np.array(p["y_axis"], float)
            out.append(dict(name=p["name"], kind=kind, sector=p.get("sector"), pos=np.array(p["position"], float), up=up / np.linalg.norm(up)))
    return out


def triggered(pos, ball_kind: str, items: list, R: Rules | None = None):
    """The transformer that transforms a ball of `ball_kind` whose centre is at pos, or None: the nearest one, if
    it is nearer than the radius and of another kind."""
    if not items:
        return None
    R = R or rules()
    d = [float(np.linalg.norm(np.asarray(pos, float) - it["pos"])) for it in items]
    k = int(np.argmin(d))
    return items[k] if d[k] < R.radius and items[k]["kind"] != ball_kind else None


def new_state(item: dict, R: Rules | None = None):
    """(pos, vel, omega) of the new ball when it appears: at rest at the hover point."""
    R = R or rules()
    return item["pos"] + R.hover * item["up"], np.zeros(3), np.zeros(3)


def stretch(world, item: dict, P=None, R: Rules | None = None, scene=None):
    """A transformation as a stretch of a route: (the new ball, its trace, its end state). The trace is `body_after`
    rows at the hover point (nothing happens), then one tick of the new ball left alone: the tick after that is the
    first the keys act in. The trace has the columns of a rollout's (`kernel.World.rollout`). scene: the loose
    balls' state, changed in place (they go on moving while the ball is away); the end state carries it."""
    from . import physics as ph
    R = R or rules()
    new = {b.name: b for b in (ph.WOOD, ph.STONE, ph.PAPER)}[item["kind"]]
    p0, v0, o0 = new_state(item, R)
    scene = world.scene() if scene is None else scene
    world.idle(scene, R.body_after, P)
    status, _d, p1, v1, o1, stats, tr = world.rollout(new, P or ph.Params(), p0, v0, o0, np.zeros((1, 3)), [1], trace=True, scene=scene)
    wait = np.zeros((R.body_after, 9))
    wait[:, :3] = p0
    return new, np.vstack([wait, tr]), (p1, v1, o1, scene), stats, status
