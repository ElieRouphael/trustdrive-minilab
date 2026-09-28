"""Controllers: discrete LQR lane keeping and integrity-aware shared control.

The LQR is designed on the linearised lane-error model.  The discrete
algebraic Riccati equation is solved by iterating the Riccati recursion so the
package stays scipy-free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vehicle import Vehicle


def dlqr(A, B, Q, R, iters: int = 2000, tol: float = 1e-12):
    """Discrete infinite-horizon LQR gain via Riccati iteration.

    Returns ``(K, P)`` with the optimal control ``u = -K x``.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Q = np.asarray(Q, float)
    R = np.asarray(R, float)
    P = Q.copy()
    for _ in range(iters):
        BtPB_R = R + B.T @ P @ B
        K = np.linalg.solve(BtPB_R, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


@dataclass
class LQRGains:
    Q: np.ndarray = None
    R: np.ndarray = None

    def __post_init__(self):
        if self.Q is None:
            self.Q = np.diag([1.0, 5.0])
        if self.R is None:
            self.R = np.array([[8.0]])


class LQRController:
    """Speed-scheduled lane-keeping LQR with curvature feed-forward.

    ``delta = atan(L * kappa) - K(v) x_hat``  (steering saturated by the
    vehicle).  The gain is re-solved and cached as a function of the current
    speed ``v`` (a lightweight LPV / gain-scheduling scheme) so that the
    integrity-driven speed contraction does not leave the controller mistuned.
    """

    def __init__(self, vehicle: Vehicle, gains: LQRGains | None = None,
                 v: float | None = None):
        self.vehicle = vehicle
        self.gains = gains or LQRGains()
        self._cache: dict[float, np.ndarray] = {}
        self._v_nom = vehicle.p.v if v is None else v
        self.K = self.gain_at(self._v_nom)
        self.P = self._P

    def gain_at(self, v: float) -> np.ndarray:
        key = round(float(v), 1)
        if key not in self._cache:
            A, B, _ = self.vehicle.linear_model(key)
            K, P = dlqr(A, B, self.gains.Q, self.gains.R, iters=500)
            self._cache[key] = K
            self._P = P
        return self._cache[key]

    def feedforward(self, kappa: float) -> float:
        """Map/geometry-based steering; independent of the camera."""
        return Vehicle.curvature_feedforward(kappa, self.vehicle.p.L)

    def feedback(self, x_hat: np.ndarray, v: float | None = None) -> float:
        """State-feedback part ``-K(v) x_hat`` (unsaturated)."""
        K = self.K if v is None else self.gain_at(v)
        return -float((K @ np.asarray(x_hat, float).reshape(2, 1)).ravel()[0])

    def command(self, x_hat: np.ndarray, kappa: float, v: float | None = None) -> float:
        p = self.vehicle.p
        delta = self.feedforward(kappa) + self.feedback(x_hat, v)
        return float(np.clip(delta, -p.delta_max, p.delta_max))


@dataclass
class SharedControlParams:
    v_nom: float = 15.0        # nominal reference speed [m/s]
    v_floor_frac: float = 0.4  # minimum speed fraction under low integrity
    # authority shaping
    lambda_min: float = 0.05   # never fully lock out automation
    lambda_max: float = 0.95   # never fully lock out the human
    workload_penalty: float = 0.35  # how much high workload reduces automation trust


class SharedController:
    """Integrity-aware progressive authority arbitration.

    The blended steering command is::

        delta = lambda * delta_auto + (1 - lambda) * delta_human

    where ``lambda`` (automation authority) is a smooth function of the
    integrity score ``I`` and the driver workload ``W``.  In addition the
    reference speed is contracted when integrity drops so the vehicle shrinks
    its operating envelope instead of blindly trusting either agent.
    """

    def __init__(self, params: SharedControlParams | None = None):
        self.p = params or SharedControlParams()

    def authority(self, integrity: float, workload: float,
                  driver_available: bool = True) -> float:
        """Automation authority ``lambda`` in ``[lambda_min, lambda_max]``."""
        p = self.p
        # base authority rises with integrity
        lam = integrity
        # high workload means the human is a less reliable fallback: keep more
        # authority with automation when the human is loaded, but only while
        # integrity is not catastrophic.
        lam = lam + p.workload_penalty * workload * (integrity - 0.5)
        if not driver_available:
            lam = max(lam, 0.85)  # no human to fall back on -> automation holds on
        return float(np.clip(lam, p.lambda_min, p.lambda_max))

    def reference_speed(self, integrity: float) -> float:
        p = self.p
        return p.v_nom * (p.v_floor_frac + (1.0 - p.v_floor_frac) * integrity)

    def arbitrate(self, fb_auto: float, delta_human: float, ff: float,
                  integrity: float, workload: float,
                  driver_available: bool = True):
        """Blend authority over the *feedback* only; feed-forward always applies.

        ``delta = ff + lambda * fb_auto + (1 - lambda) * delta_human``

        The map/geometry feed-forward ``ff`` keeps the vehicle centred on a
        curve regardless of who holds feedback authority, which is exactly what
        a reactive human loses under a hard handover.
        """
        lam = self.authority(integrity, workload, driver_available)
        delta = ff + lam * fb_auto + (1.0 - lam) * delta_human
        mode = self._mode(integrity, workload, driver_available)
        info = {
            "lambda": lam,
            "v_ref": self.reference_speed(integrity),
            "mode": mode,
        }
        return float(delta), info

    @staticmethod
    def _mode(integrity: float, workload: float, driver_available: bool) -> str:
        if integrity > 0.75:
            return "automation"
        if integrity > 0.4:
            return "shared"
        if not driver_available or workload > 0.7:
            return "degrade+shared"   # neither agent ideal -> shrink envelope
        return "human"
