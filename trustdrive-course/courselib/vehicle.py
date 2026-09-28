"""Kinematic lane-error vehicle model (standalone teaching copy).

State x = [e_y, e_psi]:  lateral error [m] and heading error [rad] w.r.t. the
lane centre.  Control is the steering angle delta [rad]; the road imposes a
curvature kappa [1/m] as a measured disturbance.

Nonlinear discrete update (forward Euler, sample time Ts):

    e_y_{k+1}   = e_y_k   + v sin(e_psi_k) Ts
    e_psi_{k+1} = e_psi_k + (v/L tan(delta_k) - v kappa_k) Ts

Small-angle linearisation used for control/observer design:

    x_{k+1} = A x_k + B delta_k + E kappa_k
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    L: float = 2.7        # wheelbase [m]
    Ts: float = 0.05      # sample time [s]
    v: float = 15.0       # speed [m/s]
    delta_max: float = 0.6  # steering saturation [rad]


class Vehicle:
    def __init__(self, params: VehicleParams | None = None, x0=None):
        self.p = params or VehicleParams()
        self.x = np.zeros(2) if x0 is None else np.asarray(x0, float).copy()

    def dynamics(self, x, delta, kappa, v=None):
        p = self.p
        v = p.v if v is None else v
        delta = float(np.clip(delta, -p.delta_max, p.delta_max))
        e_y, e_psi = float(x[0]), float(x[1])
        return np.array([e_y + v * np.sin(e_psi) * p.Ts,
                         e_psi + (v / p.L * np.tan(delta) - v * kappa) * p.Ts])

    def step(self, delta, kappa, v=None):
        self.x = self.dynamics(self.x, delta, kappa, v)
        return self.x.copy()

    def linear_model(self, v=None):
        p = self.p
        v = p.v if v is None else v
        A = np.array([[1.0, v * p.Ts], [0.0, 1.0]])
        B = np.array([[0.0], [v * p.Ts / p.L]])
        E = np.array([[0.0], [-v * p.Ts]])
        return A, B, E

    @staticmethod
    def curvature_feedforward(kappa, L):
        """Steering that cancels a constant curvature: atan(L kappa)."""
        return float(np.arctan(L * kappa))
