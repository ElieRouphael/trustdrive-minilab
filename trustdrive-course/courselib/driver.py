"""Simulated human driver: delayed, noisy proportional lane-keeper."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class DriverParams:
    K_h: tuple = (0.12, 0.9)   # proportional gains on (e_y, e_psi)
    delay_steps: int = 4       # reaction delay (~0.2 s at 20 Hz)
    noise_std: float = 0.02
    delta_max: float = 0.6


class SimulatedDriver:
    def __init__(self, params: DriverParams | None = None, seed=None):
        self.p = params or DriverParams()
        self.rng = np.random.default_rng(seed)
        self.buf = deque([np.zeros(2)] * (self.p.delay_steps + 1),
                         maxlen=self.p.delay_steps + 1)

    def command(self, x_hat, workload=0.2, available=True, conflict=0.0):
        self.buf.append(np.asarray(x_hat, float).copy())
        xd = self.buf[0]
        delta = -float(np.asarray(self.p.K_h) @ xd)
        eff_noise = self.p.noise_std * (1 + 2 * workload)
        delta = (1 - 0.5 * workload) * delta + self.rng.normal(0, eff_noise)
        if conflict > 0:
            delta -= conflict * 0.4 * np.sign(delta if delta != 0 else 1.0)
        if not available:
            delta = 0.0
        return float(np.clip(delta, -self.p.delta_max, self.p.delta_max))
