"""The compiled tick is the same kernel as `physics.step`, and a rollout is step + settle."""
import os
from pathlib import Path

import numpy as np
import pytest

from ballistics import physics as ph
from ballistics.geometry import Mesh
from ballistics.kernel import OK, World

DATA = Path(os.environ.get("BALLISTICS_DATA", Path(__file__).resolve().parent.parent / "data"))
L1_MESH = DATA / "Level_01-mesh.npz"
L1_RUN = DATA / "runs" / "SR01_1_14_457_264.npz"
needs_data = pytest.mark.skipif(not (L1_MESH.exists() and L1_RUN.exists()), reason="needs the local level 1 mesh and run")


@needs_data
def test_compiled_tick_equals_the_python_tick_on_recorded_states():
    from ballistics.calibrate import _start
    from ballistics.trace import Run
    m = Mesh.load(L1_MESH)
    m = m.subset(~m.placed)
    run, P, w = Run.load(L1_RUN), ph.Params(), World.of(m)
    t = run.ticks()
    worst, multi, hits = 0.0, 0, 0
    for k in range(0, len(t) - 1, 3):
        i0, i1 = t[k], t[k + 1]
        if run.quarter[i1] < 0:
            continue
        ball, s, _ = _start(m, run, i0, P)
        push = ph.key_push(run.keys_before(i1), int(run.quarter[i1]))
        a, info = ph.step_reference(m, ball, s, push, P)
        pos, vel, omega, hit, rows, n_imp = w.step(ball, P, s.pos, s.vel, s.omega, push)
        assert hit == info["hit"] and len(rows) == len(info["contacts"])
        assert [r[0] for r in rows] == [c.triangle for c in info["contacts"]]
        worst = max(worst, np.abs(pos - a.pos).max(), np.abs(vel - a.vel).max(), np.abs(omega - a.omega).max())
        multi += len(rows) > 1
        hits += hit
    assert worst < 1e-9 and multi > 500 and hits > 20


@needs_data
def test_a_rollout_is_step_and_settle():
    from ballistics.calibrate import _start
    from ballistics.trace import Run
    m = Mesh.load(L1_MESH)
    m = m.subset(~m.placed)
    run, P, w = Run.load(L1_RUN), ph.Params(), World.of(m)
    t = run.ticks()
    k0 = int(np.searchsorted(t, 10270))                      # off the rail, a flight, a landing beside a step, onto the ramp
    ball, s, _ = _start(m, run, t[k0], P)
    pushes = np.array([ph.key_push(run.keys_before(t[k + 1]), int(run.quarter[t[k + 1]])) for k in range(k0, k0 + 120)])
    ref = s.copy()
    for p in pushes:
        ref, _ = ph.step_reference(m, ball, ref, p, P)
        ref = ph.settle(m, ref, ball.radius)
    status, done, pos, vel, omega, stats, trace = w.rollout(ball, P, s.pos, s.vel, s.omega, pushes, np.ones(len(pushes), np.int64), trace=True)
    assert status == OK and done == 120 and len(trace) == 120
    assert np.abs(pos - ref.pos).max() < 1e-7 and np.abs(vel - ref.vel).max() < 1e-7
    assert stats[3] > 60                                       # most of it was a flight
