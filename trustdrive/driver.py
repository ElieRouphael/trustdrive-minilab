"""Simulated human driver model for the shared-control experiments.

The human is modelled as a delayed, noisy proportional lane-keeper with a
time-varying workload and (optionally) intermittent availability.  It can also
be told to *disagree* with the automation to study authority arbitration under
a driver/automation conflict.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DriverParams:
    K_h: tuple = (0.12, 0.9)   # human proportional gains on (e_y, e_psi)
    delay_steps: int = 4       # reaction delay (~0.2 s at 20 Hz)
    noise_std: float = 0.02    # steering noise [rad]
    base_workload: float = 0.2
    delta_max: float = 0.6


class SimulatedDriver:
    """Human steering command with delay, noise, workload and conflict modes."""

    def __init__(self, params: DriverParams | None = None, seed: int | None = None):
        self.p = params or DriverParams()
        self.rng = np.random.default_rng(seed)
        self.buffer = deque([np.zeros(2)] * (self.p.delay_steps + 1),
                            maxlen=self.p.delay_steps + 1)
        self.workload = self.p.base_workload
        self.available = True
        self.conflict = 0.0  # in [0,1]: fraction of a deliberate wrong-way bias

    def set_state(self, workload: float | None = None,
                  available: bool | None = None, conflict: float | None = None):
        if workload is not None:
            self.workload = float(np.clip(workload, 0.0, 1.0))
        if available is not None:
            self.available = bool(available)
        if conflict is not None:
            self.conflict = float(np.clip(conflict, 0.0, 1.0))

    def command(self, x_hat: np.ndarray) -> float:
        """Delayed, noisy human steering command based on the estimated state."""
        self.buffer.append(np.asarray(x_hat, float).copy())
        x_delayed = self.buffer[0]  # oldest = delayed observation
        Kh = np.asarray(self.p.K_h)
        delta = float(-(Kh @ x_delayed))

        # workload inflates noise and attenuates the correction
        eff_noise = self.p.noise_std * (1.0 + 2.0 * self.workload)
        delta = (1.0 - 0.5 * self.workload) * delta + self.rng.normal(0.0, eff_noise)

        # deliberate conflict: steer the wrong way (e.g. toward a perceived hazard)
        if self.conflict > 0.0:
            delta = delta - self.conflict * 0.4 * np.sign(delta if delta != 0 else 1.0)

        if not self.available:
            delta = 0.0  # hands off
        return float(np.clip(delta, -self.p.delta_max, self.p.delta_max))
