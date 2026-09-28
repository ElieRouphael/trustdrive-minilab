"""Kinematic lane-error vehicle model (bicycle model in road-error coordinates).

The state is the lateral/heading error of the vehicle with respect to the lane
centre::

    x = [e_y, e_psi]^T

with

    e_y   : lateral displacement from the lane centre [m]
    e_psi : heading (yaw) error relative to the lane tangent [rad]

The single control input is the front steering angle ``delta`` [rad]. The road
imposes a curvature ``kappa`` [1/m] that acts as a measured disturbance.

Nonlinear discrete update (Euler, sample time ``Ts``)::

    e_y_{k+1}   = e_y_k   + v * sin(e_psi_k) * Ts
    e_psi_{k+1} = e_psi_k + (v / L * tan(delta_k) - v * kappa_k) * Ts

For controller / observer design we also expose the small-angle linearisation
around ``x = 0``::

    x_{k+1} = A x_k + B delta_k + E kappa_k

    A = [[1, v*Ts], [0, 1]]
    B = [[0], [v*Ts/L]]
    E = [[0], [-v*Ts]]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleParams:
    """Physical parameters of the kinematic vehicle model."""

    L: float = 2.7        # wheelbase [m]
    Ts: float = 0.05      # sample time [s]  (20 Hz)
    v: float = 15.0       # longitudinal speed [m/s] (~54 km/h)
    delta_max: float = 0.6  # steering saturation [rad] (~34 deg)


class Vehicle:
    """Closed-form kinematic lane-error model with an integrator step."""

    def __init__(self, params: VehicleParams | None = None,
                 x0: np.ndarray | None = None):
        self.p = params or VehicleParams()
        self.x = np.zeros(2) if x0 is None else np.asarray(x0, dtype=float).copy()

    # ------------------------------------------------------------------ #
    # Continuous-time / discrete nonlinear dynamics
    # ------------------------------------------------------------------ #
    def dynamics(self, x: np.ndarray, delta: float, kappa: float,
                 v: float | None = None) -> np.ndarray:
        """Nonlinear one-step update of the error state."""
        p = self.p
        v = p.v if v is None else v
        delta = float(np.clip(delta, -p.delta_max, p.delta_max))
        e_y, e_psi = float(x[0]), float(x[1])
        e_y_next = e_y + v * np.sin(e_psi) * p.Ts
        e_psi_next = e_psi + (v / p.L * np.tan(delta) - v * kappa) * p.Ts
        return np.array([e_y_next, e_psi_next])

    def step(self, delta: float, kappa: float, v: float | None = None) -> np.ndarray:
        """Advance the internal state and return the new state."""
        self.x = self.dynamics(self.x, delta, kappa, v)
        return self.x.copy()

    # ------------------------------------------------------------------ #
    # Linearised model (used by LQR and the Kalman predictor)
    # ------------------------------------------------------------------ #
    def linear_model(self, v: float | None = None):
        """Return the discrete linear matrices ``(A, B, E)`` at speed ``v``."""
        p = self.p
        v = p.v if v is None else v
        A = np.array([[1.0, v * p.Ts],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [v * p.Ts / p.L]])
        E = np.array([[0.0],
                      [-v * p.Ts]])
        return A, B, E

    @staticmethod
    def curvature_feedforward(kappa: float, L: float) -> float:
        """Steering that exactly cancels a constant curvature in the model.

        ``v/L * tan(delta_ff) = v * kappa``  =>  ``delta_ff = atan(L * kappa)``.
        """
        return float(np.arctan(L * kappa))
