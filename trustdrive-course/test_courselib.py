"""Smoke tests for courselib (run with pytest or directly).

    python test_courselib.py
"""
from __future__ import annotations
import numpy as np

from courselib import (Vehicle, dlqr, build_scenario, run_episode, metrics,
                       METHODS, SCENARIOS)


def test_lqr_stabilises():
    v = Vehicle()
    A, B, _ = v.linear_model()
    K, _ = dlqr(A, B, np.diag([1.0, 5.0]), np.array([[8.0]]))
    assert np.all(np.abs(np.linalg.eigvals(A - B @ K)) < 1.0)


def test_all_methods_run():
    for sc in SCENARIOS:
        fr = build_scenario(sc, T=10.0)
        for m in METHODS:
            log = run_episode(m, fr, seed=0)
            assert np.all(np.isfinite(log["e_y"]))
            assert metrics(log)["RMSE_ey"] >= 0.0


def test_nis_detects_ood():
    log = run_episode("proposed", build_scenario("ood_curve", T=30.0), seed=0)
    t = log["t"]; win = (t >= 13) & (t <= 22); base = t < 12
    assert log["nis"][win].mean() > 3 * log["nis"][base].mean()


def test_proposed_beats_naive_under_degradation():
    for sc in ("fog", "ood_curve"):
        fr = build_scenario(sc, T=30.0)
        assert (metrics(run_episode("proposed", fr, seed=0))["RMSE_ey"]
                <= metrics(run_episode("naive", fr, seed=0))["RMSE_ey"] + 1e-9)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
