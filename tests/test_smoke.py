"""Fast sanity checks for the closed loop (run with pytest or directly).

    pytest -q            # if pytest is installed
    python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustdrive import (build_scenario, run_simulation, compute_metrics,  # noqa: E402
                        METHODS, SCENARIOS, dlqr, Vehicle)


def test_dlqr_stabilises():
    v = Vehicle()
    A, B, _ = v.linear_model()
    K, P = dlqr(A, B, np.diag([1.0, 5.0]), np.array([[8.0]]))
    eig = np.linalg.eigvals(A - B @ K)
    assert np.all(np.abs(eig) < 1.0), f"closed-loop unstable: {eig}"


def test_all_methods_run_all_scenarios():
    for sc in SCENARIOS:
        frames = build_scenario(sc, T=10.0)
        for m in METHODS:
            log = run_simulation(m, frames, seed=0)
            assert len(log["e_y"]) == len(frames)
            assert np.all(np.isfinite(log["e_y"]))
            met = compute_metrics(log)
            assert met["RMSE_ey"] >= 0.0


def test_nis_detects_ood():
    """NIS must rise out of distribution even though sigma barely moves."""
    frames = build_scenario("ood_curve", T=30.0)
    log = run_simulation("proposed", frames, seed=0)
    t = log["t"]
    win = (t >= 13) & (t <= 22)
    out = ~win & (t < 12)
    assert log["nis"][win].mean() > 3 * log["nis"][out].mean() + 1.0
    # integrity should drop meaningfully inside the OOD window
    assert log["integrity"][win].mean() < 0.4


def test_proposed_beats_naive_under_degradation():
    for sc in ("fog", "ood_curve"):
        frames = build_scenario(sc, T=30.0)
        naive = compute_metrics(run_simulation("naive", frames, seed=0))
        prop = compute_metrics(run_simulation("proposed", frames, seed=0))
        assert prop["RMSE_ey"] <= naive["RMSE_ey"] + 1e-9, sc
        assert prop["unsafe_ai_usage_pct"] <= naive["unsafe_ai_usage_pct"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} smoke tests passed.")
