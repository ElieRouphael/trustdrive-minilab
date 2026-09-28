"""Control: discrete gain-scheduled LQR and integrity-aware shared control.

The discrete algebraic Riccati equation is solved by fixed-point iteration so
the library needs only numpy.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .vehicle import Vehicle


def dlqr(A, B, Q, R, iters: int = 600, tol: float = 1e-12):
    """Infinite-horizon discrete LQR gain via Riccati iteration.  u = -K x."""
    A, B, Q, R = map(lambda M: np.asarray(M, float), (A, B, Q, R))
    P = Q.copy()
    for _ in range(iters):
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class LQRController:
    """Speed-scheduled lane-keeping LQR with map-based curvature feed-forward."""

    def __init__(self, vehicle: Vehicle, Q=None, R=None):
        self.v = vehicle
        self.Q = np.diag([1.0, 5.0]) if Q is None else np.asarray(Q, float)
        self.R = np.array([[8.0]]) if R is None else np.asarray(R, float)
        self._cache = {}

    def gain_at(self, v):
        key = round(float(v), 1)
        if key not in self._cache:
            A, B, _ = self.v.linear_model(key)
            self._cache[key] = dlqr(A, B, self.Q, self.R)[0]
        return self._cache[key]

    def feedforward(self, kappa):
        return Vehicle.curvature_feedforward(kappa, self.v.p.L)

    def feedback(self, x_hat, v=None):
        K = self.gain_at(self.v.p.v if v is None else v)
        return -float((K @ np.asarray(x_hat, float).reshape(2, 1)).ravel()[0])

    def command(self, x_hat, kappa, v=None):
        d = self.feedforward(kappa) + self.feedback(x_hat, v)
        return float(np.clip(d, -self.v.p.delta_max, self.v.p.delta_max))


@dataclass
class SharedControlParams:
    v_nom: float = 15.0
    v_floor_frac: float = 0.4
    lambda_min: float = 0.05
    lambda_max: float = 0.95
    workload_penalty: float = 0.35


class SharedController:
    """Integrity-aware authority arbitration.

        delta = ff + lambda * fb_auto + (1 - lambda) * delta_human
        lambda = clip(I + rho W (I - 0.5), lam_min, lam_max)
        v_ref  = v_nom (0.4 + 0.6 I)
    """

    def __init__(self, params: SharedControlParams | None = None):
        self.p = params or SharedControlParams()

    def authority(self, I, workload, driver_available=True):
        p = self.p
        lam = I + p.workload_penalty * workload * (I - 0.5)
        if not driver_available:
            lam = max(lam, 0.85)
        return float(np.clip(lam, p.lambda_min, p.lambda_max))

    def reference_speed(self, I):
        p = self.p
        return p.v_nom * (p.v_floor_frac + (1 - p.v_floor_frac) * I)

    def arbitrate(self, fb_auto, delta_human, ff, I, workload, driver_available=True):
        lam = self.authority(I, workload, driver_available)
        delta = ff + lam * fb_auto + (1 - lam) * delta_human
        return float(delta), lam, self.reference_speed(I)
