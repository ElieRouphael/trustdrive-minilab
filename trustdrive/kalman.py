"""Model-based Kalman predictor and innovation-based consistency monitoring.

The Kalman filter runs the *same* linearised model as the controller and treats
the perception output (whether synthetic or from the CNN) as a noisy
measurement of the full state (``C = I``).  The key quantity for integrity
monitoring is the **normalised innovation squared (NIS)**::

    nu   = y_AI - C x_pred                (innovation)
    S    = C P_pred C^T + R_AI            (innovation covariance)
    NIS  = nu^T S^{-1} nu

Under the hypothesis that model and measurement are mutually consistent,
``NIS`` follows a chi-squared distribution with ``dim(y)`` degrees of freedom.
A sustained NIS above the chi-squared threshold flags a model / perception
mismatch - e.g. the vehicle has entered an out-of-distribution regime - even
when the neural network itself reports high confidence.
"""
from __future__ import annotations

import numpy as np

# Chi-squared thresholds for 2 degrees of freedom.
CHI2_2DOF = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}


class KalmanMonitor:
    """Kalman filter over the lane-error state with NIS output.

    Parameters
    ----------
    A, B, E : linear model matrices (E multiplies the curvature disturbance).
    Q       : process noise covariance.
    C       : measurement matrix (defaults to identity - perception sees x).
    """

    def __init__(self, A, B, E, Q, C=None, P0=None):
        self.A = np.asarray(A, float)
        self.B = np.asarray(B, float)
        self.E = np.asarray(E, float)
        self.Q = np.asarray(Q, float)
        self.C = np.eye(self.A.shape[0]) if C is None else np.asarray(C, float)
        self.n = self.A.shape[0]
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) if P0 is None else np.asarray(P0, float)
        # populated after each update()
        self.last = {}

    def predict(self, delta: float, kappa: float):
        """Time update using the control ``delta`` and curvature ``kappa``."""
        self.x = self.A @ self.x + (self.B.flatten() * delta) + (self.E.flatten() * kappa)
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy(), self.P.copy()

    def innovation(self, y: np.ndarray, R: np.ndarray):
        """Innovation, innovation covariance and NIS for measurement ``y``."""
        y = np.asarray(y, float).reshape(self.n)
        R = np.asarray(R, float)
        nu = y - self.C @ self.x
        S = self.C @ self.P @ self.C.T + R
        nis = float(nu @ np.linalg.solve(S, nu))
        return nu, S, nis

    def update(self, y: np.ndarray, R: np.ndarray):
        """Full measurement update; also stores innovation diagnostics.

        Returns the NIS value for convenience.
        """
        nu, S, nis = self.innovation(y, R)
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = (np.eye(self.n) - K @ self.C) @ self.P
        self.last = {
            "innovation": nu,
            "S": S,
            "nis": nis,
            "x_post": self.x.copy(),
            "P_post": self.P.copy(),
        }
        return nis

    def nis_consistent(self, confidence: float = 0.99) -> bool:
        """Whether the most recent NIS is below the chi-squared threshold."""
        thr = CHI2_2DOF.get(confidence, 9.210)
        return self.last.get("nis", 0.0) <= thr
