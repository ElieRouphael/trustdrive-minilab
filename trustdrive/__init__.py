"""TrustDrive MiniLab - integrity-aware shared control under perception uncertainty.

A compact research demonstrator: a kinematic lane-error vehicle, an (optionally
learned) perception front-end with epistemic uncertainty, a model-based
integrity monitor (Kalman/NIS + ODD), and integrity-aware shared human-vehicle
control.  See the README for the full architecture.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .vehicle import Vehicle, VehicleParams
from .road import RoadRenderer, RoadConditions, CameraParams
from .perception import SyntheticPerception, PerceptionOutput
from .uncertainty import ensemble_stats, total_variance
from .kalman import KalmanMonitor, CHI2_2DOF
from .integrity import IntegrityMonitor, ODD, IntegrityConfig
from .controllers import LQRController, SharedController, dlqr
from .driver import SimulatedDriver, DriverParams
from .scenarios import build_scenario, SCENARIOS
from .simulation import run_simulation, compute_metrics, METHODS

__all__ = [
    "Vehicle", "VehicleParams",
    "RoadRenderer", "RoadConditions", "CameraParams",
    "SyntheticPerception", "PerceptionOutput",
    "ensemble_stats", "total_variance",
    "KalmanMonitor", "CHI2_2DOF",
    "IntegrityMonitor", "ODD", "IntegrityConfig",
    "LQRController", "SharedController", "dlqr",
    "SimulatedDriver", "DriverParams",
    "build_scenario", "SCENARIOS",
    "run_simulation", "compute_metrics", "METHODS",
]
