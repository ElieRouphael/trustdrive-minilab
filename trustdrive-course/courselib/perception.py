"""Perception: synthetic uncertain sensor and ensemble statistics.

A learned perception front-end is emulated analytically so the whole course
runs with numpy alone.  Its reported uncertainty grows with image degradation,
and out of distribution it becomes over-confident (a systematic bias appears
that the reported covariance does not model) -- the regime where the model
based NIS monitor becomes necessary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Conditions:
    brightness: float = 1.0
    blur: float = 0.0
    occlusion: float = 0.0


def ensemble_stats(predictions):
    """Mean and unbiased covariance of M ensemble predictions (M, d)."""
    P = np.atleast_2d(np.asarray(predictions, float))
    mean = P.mean(0)
    if P.shape[0] < 2:
        return mean, np.zeros((P.shape[1], P.shape[1]))
    d = P - mean
    return mean, (d.T @ d) / (P.shape[0] - 1)


class SyntheticPerception:
    def __init__(self, kappa_train_max=0.02, base_std=(0.02, 0.01), seed=None):
        self.kmax = kappa_train_max
        self.base = np.asarray(base_std, float)
        self.rng = np.random.default_rng(seed)

    def reported_std(self, cond: Conditions, kappa):
        return self.base * (
            (1 + 2.5 * cond.blur)
            * (1 + 1.5 * max(0.0, 1 - cond.brightness))
            * (1 + 3.0 * cond.occlusion)
            * (1 + 4.0 * max(0.0, abs(kappa) - self.kmax)))

    def measure(self, true_state, kappa, cond: Conditions | None = None):
        cond = cond or Conditions()
        srep = self.reported_std(cond, kappa)
        ood = max(0.0, abs(kappa) - self.kmax) / max(self.kmax, 1e-9)
        strue = srep * (1 + 1.2 * ood)
        bias = np.array([0.18 * ood * np.sign(kappa if kappa != 0 else 1.0), 0.05 * ood])
        mean = np.asarray(true_state, float) + bias + self.rng.normal(0, strue)
        cov = np.diag(srep ** 2)
        members = mean + self.rng.normal(0, srep, size=(3, 2))
        return {"mean": mean, "cov": cov, "trace": float(np.trace(cov)), "members": members}
