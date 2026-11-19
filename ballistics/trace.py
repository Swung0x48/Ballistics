"""Recorded engine traces: what the real game did, used to check the geometry and to calibrate the physics."""
from __future__ import annotations

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
