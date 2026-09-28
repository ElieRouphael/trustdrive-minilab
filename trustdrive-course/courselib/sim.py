"""Scenarios, closed-loop simulation and metrics (standalone teaching copy)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .vehicle import Vehicle, VehicleParams
from .control import LQRController, SharedController, SharedControlParams
from .perception import SyntheticPerception, Conditions
from .monitor import KalmanMonitor, ODD, IntegrityMonitor
from .driver import SimulatedDriver

METHODS = ("naive", "handover", "proposed")
SCENARIOS = ("nominal", "fog", "ood_curve", "conflict")


@dataclass
class Frame:
    kappa: float
    cond: Conditions
    workload: float
    available: bool
    conflict: float


def _pulse(t, t0, t1, lo, hi):
    return hi if t0 <= t < t1 else lo


def _ramp(t, t0, t1, lo, hi):
    if t <= t0:
        return lo
    if t >= t1:
        return hi
    return lo + (hi - lo) * (t - t0) / (t1 - t0)


def build_scenario(name, T=30.0, Ts=0.05, base_kappa_amp=0.012):
    n = int(round(T / Ts))
    frames = []
    for k in range(n):
        t = k * Ts
        kappa = base_kappa_amp * np.sin(2 * np.pi * t / 20.0)
        cond = Conditions()
        workload, available, conflict = 0.2, True, 0.0
        if name == "fog":
            cond.blur = _ramp(t, 10, 15, 0, 3.0) if t < 17 else _ramp(t, 20, 25, 3.0, 0)
            cond.brightness = 1 - 0.4 * (cond.blur / 3.0)
            cond.occlusion = _pulse(t, 15, 20, 0, 0.45)
        elif name == "ood_curve":
            kappa += _pulse(t, 12, 22, 0, 0.04)
            workload = 0.3
        elif name == "conflict":
            workload = _pulse(t, 14, 20, 0.2, 0.8)
            conflict = _pulse(t, 14, 20, 0, 0.8)
            available = not (20 <= t < 22)
        elif name != "nominal":
            raise ValueError(name)
        frames.append(Frame(float(kappa), cond, float(workload), bool(available), float(conflict)))
    return frames


def run_episode(method, frames, perception=None, x0=(0.3, 0.02),
                handover_sigma_thr=0.0045, seed=0):
    vp = VehicleParams()
    veh = Vehicle(vp, x0=np.asarray(x0, float))
    perception = perception or SyntheticPerception(seed=seed)
    lqr = LQRController(veh)
    shared = SharedController(SharedControlParams(v_nom=vp.v))
    driver = SimulatedDriver(seed=seed + 1)
    integ = IntegrityMonitor(ODD())
    A, B, E = veh.linear_model()
    kf = KalmanMonitor(A, B, E, np.diag([1e-4, 1e-4]), P0=np.diag([0.05, 0.02]))
    kf.x = np.asarray(x0, float).copy()

    keys = ("t", "e_y", "e_psi", "y_ey", "nis", "cov_trace", "integrity",
            "lambda", "v_ref", "delta", "delta_auto", "delta_human",
            "kappa", "inside_odd")
    log = {k: [] for k in keys}
    prev_delta = 0.0
    for k, fr in enumerate(frames):
        true = veh.x.copy()
        po = perception.measure(true, fr.kappa, fr.cond)
        y, R = po["mean"], po["cov"]
        kf.predict(prev_delta, fr.kappa)
        nis = kf.update(y, R)
        inside = integ.odd.inside(fr.kappa, fr.cond.brightness, fr.cond.blur, fr.cond.occlusion)
        I, _ = integ.update(po["trace"], nis, fr.kappa, fr.cond.brightness,
                            fr.cond.blur, fr.cond.occlusion)
        x_hat = kf.x.copy() if method == "proposed" else y
        v_cmd = shared.reference_speed(I) if method == "proposed" else vp.v
        ff = lqr.feedforward(fr.kappa)
        fb = lqr.feedback(x_hat, v=v_cmd)
        delta_auto = float(np.clip(ff + fb, -vp.delta_max, vp.delta_max))
        delta_human = driver.command(true, fr.workload, fr.available, fr.conflict)
        if method == "naive":
            lam, v_ref, delta = 1.0, vp.v, delta_auto
        elif method == "handover":
            unsafe = po["trace"] > handover_sigma_thr
            lam, v_ref = (0.0 if unsafe else 1.0), vp.v
            delta = delta_human if unsafe else delta_auto
        else:
            delta, lam, v_ref = shared.arbitrate(fb, delta_human, ff, I, fr.workload, fr.available)
            delta = float(np.clip(delta, -vp.delta_max, vp.delta_max))
        veh.step(delta, fr.kappa, v=v_ref)
        prev_delta = delta
        vals = (k * vp.Ts, true[0], true[1], y[0], nis, po["trace"], I,
                lam, v_ref, delta, delta_auto, delta_human, fr.kappa, inside)
        for kk, vv in zip(keys, vals):
            log[kk].append(vv)
    return {k: np.asarray(v) for k, v in log.items()}


def metrics(log, lane_exit_thr=0.7):
    e = log["e_y"]
    over = np.abs(e) > lane_exit_thr
    exits = int(np.sum(over[1:] & ~over[:-1])) + int(over[0])
    trusted = log["lambda"] > 0.5
    return {
        "RMSE_ey": float(np.sqrt(np.mean(e ** 2))),
        "max_ey": float(np.max(np.abs(e))),
        "lane_exits": exits,
        "steering_jerk": float(np.sum(np.abs(np.diff(log["delta"])))),
        "unsafe_ai_usage_pct": float(np.mean(trusted & ~log["inside_odd"].astype(bool)) * 100),
    }
