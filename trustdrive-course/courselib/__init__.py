"""courselib -- a small, self-contained reference library for the TrustDrive course.

The notebooks build each idea from scratch; this package is the clean,
consolidated version they culminate in.  It depends only on numpy and
matplotlib and is independent of the main ``trustdrive`` research package.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .vehicle import Vehicle, VehicleParams
from .control import dlqr, LQRController, SharedController, SharedControlParams
from .perception import SyntheticPerception, Conditions, ensemble_stats
from .monitor import KalmanMonitor, ODD, IntegrityMonitor, IntegrityConfig, CHI2_2DOF
from .driver import SimulatedDriver, DriverParams
from .sim import (build_scenario, run_episode, metrics, Frame, METHODS, SCENARIOS)
from .viz import use_course_style, trust_panel, METHOD_COLORS

__all__ = [
    "Vehicle", "VehicleParams", "dlqr", "LQRController", "SharedController",
    "SharedControlParams", "SyntheticPerception", "Conditions", "ensemble_stats",
    "KalmanMonitor", "ODD", "IntegrityMonitor", "IntegrityConfig", "CHI2_2DOF",
    "SimulatedDriver", "DriverParams", "build_scenario", "run_episode", "metrics",
    "Frame", "METHODS", "SCENARIOS", "use_course_style", "trust_panel", "METHOD_COLORS",
]
