"""Model-based monitoring: Kalman/NIS, ODD, and integrity-score fusion."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# chi-squared thresholds for 2 degrees of freedom
CHI2_2DOF = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}


class KalmanMonitor:
    """Kalman filter over the lane-error state; exposes the NIS statistic.

        nu  = y - C x_pred
        S   = C P_pred C^T + R
        NIS = nu^T S^{-1} nu   ~ chi^2(dim y) under model/measurement consistency
    """

    def __init__(self, A, B, E, Q, C=None, P0=None):
        self.A, self.B, self.E, self.Q = map(lambda M: np.asarray(M, float), (A, B, E, Q))
        self.n = self.A.shape[0]
        self.C = np.eye(self.n) if C is None else np.asarray(C, float)
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) if P0 is None else np.asarray(P0, float)

    def predict(self, delta, kappa):
        self.x = self.A @ self.x + self.B.flatten() * delta + self.E.flatten() * kappa
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy()

    def update(self, y, R):
        y = np.asarray(y, float).reshape(self.n)
        nu = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + np.asarray(R, float)
        nis = float(nu @ np.linalg.solve(S, nu))
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = (np.eye(self.n) - K @ self.C) @ self.P
        return nis


@dataclass
class ODD:
    kappa_max: float = 0.022
    brightness_min: float = 0.45
    blur_max: float = 1.6
    occlusion_max: float = 0.35

    def margins(self, kappa, brightness, blur, occlusion):
        return {
            "kappa": 1 - abs(kappa) / self.kappa_max,
            "brightness": (brightness - self.brightness_min) / (1 - self.brightness_min),
            "blur": 1 - blur / self.blur_max,
            "occlusion": 1 - occlusion / self.occlusion_max,
        }

    def inside(self, kappa, brightness, blur, occlusion):
        return all(m > 0 for m in self.margins(kappa, brightness, blur, occlusion).values())

    def score(self, kappa, brightness, blur, occlusion):
        worst = min(self.margins(kappa, brightness, blur, occlusion).values())
        return float(1.0 / (1.0 + np.exp(-6.0 * worst)))


@dataclass
class IntegrityConfig:
    sigma_ref: float = 0.02
    nis_confidence: float = 0.99
    nis_scale: float = 6.0
    ema: float = 0.4


class IntegrityMonitor:
    """Fuse epistemic uncertainty, NIS and ODD into a score I in [0, 1]."""

    def __init__(self, odd: ODD | None = None, cfg: IntegrityConfig | None = None):
        self.odd = odd or ODD()
        self.cfg = cfg or IntegrityConfig()
        self.I = 1.0

    def update(self, cov_trace, nis, kappa, brightness, blur, occlusion):
        c = self.cfg
        s_unc = np.exp(-cov_trace / c.sigma_ref)
        thr = CHI2_2DOF[c.nis_confidence]
        s_nis = np.exp(-max(0.0, nis - thr) / c.nis_scale)
        s_odd = self.odd.score(kappa, brightness, blur, occlusion)
        raw = s_unc * s_nis * s_odd
        self.I = (1 - c.ema) * self.I + c.ema * raw
        return float(self.I), {"uncertainty": float(s_unc), "nis": float(s_nis),
                               "odd": float(s_odd), "raw": float(raw)}
