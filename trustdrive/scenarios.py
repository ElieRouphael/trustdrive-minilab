"""Time-varying test scenarios that drive the vehicle in and out of the ODD.

Each scenario is a list of per-step "frames"; every frame carries the road
curvature, the image/sensor condition and the driver state for that sample.
The four scenarios are:

1. ``nominal``    - clear road, everything inside the ODD.
2. ``fog``        - image quality gradually deteriorates then recovers.
3. ``ood_curve``  - curvature exceeds the trained envelope (image still looks
   valid, but the system is in an unfamiliar regime).
4. ``conflict``   - driver steers against the automation for a window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .road import RoadConditions


@dataclass
class Frame:
    kappa: float
    cond: RoadConditions
    workload: float
    available: bool
    conflict: float


def _ramp(t, t0, t1, lo, hi):
    """Linear ramp from ``lo`` at ``t0`` to ``hi`` at ``t1`` (clamped)."""
    if t <= t0:
        return lo
    if t >= t1:
        return hi
    return lo + (hi - lo) * (t - t0) / (t1 - t0)


def _pulse(t, t0, t1, lo, hi):
    """Trapezoidal pulse: ``lo`` outside ``[t0, t1]``, ``hi`` inside."""
    return hi if t0 <= t < t1 else lo


def build_scenario(name: str, T: float = 30.0, Ts: float = 0.05,
                   base_kappa_amp: float = 0.012) -> list[Frame]:
    """Return a list of :class:`Frame` for ``name`` over ``T`` seconds."""
    n = int(round(T / Ts))
    ts = np.arange(n) * Ts
    frames: list[Frame] = []

    for t in ts:
        # a gentle, in-ODD curvature baseline (a slowly weaving lane)
        kappa = base_kappa_amp * np.sin(2 * np.pi * t / 20.0)
        cond = RoadConditions(brightness=1.0, blur=0.0, occlusion=0.0, noise=0.02)
        workload, available, conflict = 0.2, True, 0.0

        if name == "nominal":
            pass

        elif name == "fog":
            # blur ramps up (10-15 s), holds, occlusion pulse (15-20 s), recover
            cond.blur = _ramp(t, 10, 15, 0.0, 3.0) if t < 17 else _ramp(t, 20, 25, 3.0, 0.0)
            cond.brightness = 1.0 - 0.4 * (cond.blur / 3.0)
            cond.occlusion = _pulse(t, 15, 20, 0.0, 0.45)

        elif name == "ood_curve":
            # push curvature past the trained envelope (|kappa|=0.04) for 12-22 s
            extra = _pulse(t, 12, 22, 0.0, 0.04)
            kappa = kappa + extra
            workload = 0.3

        elif name == "conflict":
            # driver becomes loaded and steers against automation (14-20 s)
            workload = _pulse(t, 14, 20, 0.2, 0.8)
            conflict = _pulse(t, 14, 20, 0.0, 0.8)
            # a short hands-off availability drop right after
            available = not (20 <= t < 22)

        else:
            raise ValueError(f"unknown scenario: {name!r}")

        frames.append(Frame(kappa=float(kappa), cond=cond,
                            workload=float(workload), available=bool(available),
                            conflict=float(conflict)))
    return frames


SCENARIOS = ["nominal", "fog", "ood_curve", "conflict"]
