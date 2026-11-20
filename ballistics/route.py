"""Write a solved level out as a reference line: a file, for other tools to read.

The format is the one Ballrain already reads for its reference lines (`<dir>/L<nn>.npz`, its `ReferencePath`
asset, version 1), so that nothing on either side has to know about the other:

    points   float32 (N, 3)   ball centre along the route, a point every `spacing` of arc length
    sector   uint8 (N,)       the 1-based sector of each point, never decreasing
    t_ms     float64 (N,)     game time in milliseconds at which the ball passes each point, strictly increasing
                              (a reader gets the route's speed from it)
    meta     JSON string      level, start and checkpoint positions (a reader checks with them that the line
                              belongs to the level it runs), spacing, smoothing, length, per-sector arc ranges,
                              and where this line comes from

The processing is the same as for a line built from a recorded run: the positions are smoothed in time (a
Gaussian of 40 ms, so that a bouncing ball adds no arc length) and resampled by arc length.

What goes into the line, and what does not: the line follows the route from the start of the level to the first
break that was left, and ends there. It never runs across a left break, and what the solver listed after one
(from rest on its far side) is not on the line. `meta` says whether the line is complete and names the breaks.
This module starts nothing and reads only the solver's own results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VERSION = 1
TICK_MS = 1000.0 / 132.0            # one physics tick in game time


def reference_line(results: list, info: dict, level: int, spacing: float = 0.5, time_sigma_ms: float = 40.0, tangent_span: float = 2.0):
    """results: the solver's sector results in order (staged.solve_sector). Returns (points, sector, t_ms, meta)."""
    pos, sec, breaks = [], [], []
    complete = True
    for res in results:
        cut = False
        for sg in res["segments"]:
            if sg["type"] == "assumed":
                cut = True
                break
            if len(sg["trace"]):
                pos.append(sg["trace"][:, :3])
                sec.append(np.full(len(sg["trace"]), res["sector"], np.int64))
        for b in res["breaks"]:
            if b["status"] == "unresolved":
                breaks.append(dict(sector=int(res["sector"]), no=b["no"], near=[float(v) for v in b["near"]], far=[float(v) for v in b["far"]],
                                   between=list(b["between"]), tries=int(sum(a["tries"] for a in b["attempts"] if a["mode"] == "strict"))))
        if cut or not res["solved"]:
            complete = False
            break
    if not pos:
        raise ValueError("no route to write")
    pos, sec = np.concatenate(pos).astype(np.float64), np.concatenate(sec)
    t = np.arange(len(pos)) * TICK_MS
    if time_sigma_ms > 0:                                   # Gaussian in time; ticks are evenly spaced
        half = int(round(4 * time_sigma_ms / TICK_MS))
        g = np.exp(-0.5 * (np.arange(-half, half + 1) * TICK_MS / time_sigma_ms) ** 2)
        norm = np.convolve(np.ones(len(pos)), g, "same")
        sm = np.stack([np.convolve(pos[:, k], g, "same") / norm for k in range(3)], 1)
        sm[0], sm[-1] = pos[0], pos[-1]
        pos = sm
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    keep = np.r_[True, d > 1e-9]
    cum = np.r_[0.0, np.cumsum(d)][keep]
    pos, sec, t = pos[keep], sec[keep], t[keep]
    s_new = np.arange(0.0, cum[-1], spacing)
    if cum[-1] - s_new[-1] > 1e-6:
        s_new = np.r_[s_new, cum[-1]]
    P = np.stack([np.interp(s_new, cum, pos[:, k]) for k in range(3)], 1)
    t_new = np.interp(s_new, cum, t - t[0])
    idx = np.clip(np.searchsorted(cum, s_new, side="right") - 1, 0, len(cum) - 1)
    sector = np.maximum.accumulate(sec[idx])                # sectors never go back along the line
    seg = sector[1:]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    ranges = {str(int(k)): [float(arc[np.flatnonzero(seg == k)[0]]), float(arc[np.flatnonzero(seg == k)[-1] + 1])] for k in np.unique(seg)}
    meta = dict(version=VERSION, level=int(level), start=[float(v) for v in info["start"]], checkpoints=[[float(v) for v in c] for c in info["checkpoints"]],
                airship=[float(v) for v in info["airship"]] if info.get("airship") is not None else None,
                spacing=spacing, time_sigma_ms=time_sigma_ms, tangent_span=tangent_span, length=float(arc[-1]), points=int(len(P)),
                sector_range=ranges, game_ms=float(t_new[-1]), generator="ballistics",
                source="a trajectory of the Ballistics physics kernel: a candidate, not run in the game",
                complete=bool(complete), sectors_on_line=sorted(int(k) for k in np.unique(sector)), left_breaks=breaks)
    return P.astype(np.float32), sector.astype(np.uint8), t_new.astype(np.float64), meta


def write_line(out_dir, level: int, points, sector, t_ms, meta) -> Path:
    path = Path(out_dir) / ("L%02d.npz" % int(level))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=points, sector=sector, meta=np.array(json.dumps(meta, sort_keys=True)), t_ms=t_ms)
    return path
