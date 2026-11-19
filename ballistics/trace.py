"""Recorded engine runs: what the real game did, used to check the geometry and to calibrate the physics.

Two forms:

* a per-frame trace text file (frame number, ball entity name, 3x3 rotation, position), read by `read_trace`;
* a run file (.npz) with, per observation, the ball's position and velocity, its type, the camera quarter and the
  engine's nearest-contact gap and normal, and per executed frame the frame time and held keys, read by `Run.load`.
  Observation i is the state after `frames_done[i]` executed frames; `keys[j]` belongs to executed frame j."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def read_trace(path, limit: int | None = None):
    """A per-frame ball trace (one line per frame: frame number, ball entity name, 3x3 rotation, position):
    (frames (n,), ball names [n], positions (n, 3))."""
    frames, names, pos = [], [], []
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            q = line.split()
            if len(q) < 14 or not q[0].lstrip("-").isdigit():
                continue
            frames.append(int(q[0]))
            names.append(q[1])
            pos.append((float(q[11]), float(q[12]), float(q[13])))
            if limit and len(frames) >= limit:
                break
    return np.array(frames), names, np.array(pos, np.float64).reshape(-1, 3)


@dataclass
class Run:
    pos: np.ndarray           # (n, 3) ball centre per observation
    vel: np.ndarray           # (n, 3) units per physics second
    ball_type: np.ndarray     # (n,) 0 wood, 1 paper, 2 stone; 255 no ball
    quarter: np.ndarray       # (n,) camera quarter turns; -1 unknown
    keys: np.ndarray          # (frames,) key bits of each executed frame
    frames_done: np.ndarray   # (n,) executed frames before each observation
    min_gap: np.ndarray       # (n,) the engine's own nearest-contact gap (nan none)
    min_normal: np.ndarray    # (n, 3)
    n_friction: np.ndarray    # (n,) the engine's count of friction (rolling / resting) contacts
    meta: dict

    @classmethod
    def load(cls, path) -> "Run":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["pos"], z["vel"], z["ball_type"], z["quarter"], z["keys"], z["frames_done"], z["min_gap"], z["min_normal"],
                       z["n_friction"], json.loads(str(z["meta"])))

    def ticks(self):
        """Indices of the observations at which the physics state changed: one per physics tick. Between ticks the
        engine reports the same state again (several rendered frames per tick)."""
        moving = self.ball_type != 255
        change = np.r_[True, np.any(self.pos[1:] != self.pos[:-1], axis=1) | np.any(self.vel[1:] != self.vel[:-1], axis=1)]
        return np.flatnonzero(moving & change)

    def keys_before(self, i: int) -> int:
        """Keys held in the frame that ran just before observation i (the input the tick ending at i saw last)."""
        f = int(self.frames_done[i])
        return int(self.keys[f - 1]) if f > 0 else 0
