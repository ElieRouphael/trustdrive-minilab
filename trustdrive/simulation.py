"""Closed-loop simulation tying every component together.

Three control architectures share the exact same plant, perception and driver
so that differences in the results come only from *how the AI output is used*:

* ``naive``    - perception -> LQR, full automation, no monitoring.
* ``handover`` - perception -> LQR, hard switch to the human when the reported
  uncertainty exceeds a fixed threshold.
* ``proposed`` - perception -> Kalman/NIS + ODD -> integrity score -> smooth
  authority arbitration and speed contraction.

``run_simulation`` returns a dict of equal-length numpy arrays (a "log") that
the benchmark and the notebook turn into metrics and figures.
"""
from __future__ import annotations

import numpy as np

from .controllers import LQRController, SharedController, SharedControlParams
from .driver import DriverParams, SimulatedDriver
from .integrity import ODD, IntegrityConfig, IntegrityMonitor
from .kalman import KalmanMonitor
from .perception import SyntheticPerception
from .vehicle import Vehicle, VehicleParams

METHODS = ("naive", "handover", "proposed")


def run_simulation(method: str, frames, *, perception=None, vehicle=None,
                   x0=(0.3, 0.02), handover_sigma_thr: float = 0.0045,
                   odd: ODD | None = None, seed: int = 0):
    """Run one closed-loop episode and return a log of arrays.

    Parameters
    ----------
    method : one of :data:`METHODS`.
    frames : list of :class:`trustdrive.scenarios.Frame`.
    perception : perception backend (defaults to :class:`SyntheticPerception`).
    x0 : initial ``[e_y, e_psi]`` error.
    handover_sigma_thr : uncertainty (cov-trace) threshold for the ``handover``
        baseline's hard switch.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    vp = VehicleParams()
    vehicle = vehicle or Vehicle(vp, x0=np.asarray(x0, float))
    vehicle.x = np.asarray(x0, float).copy()

    perception = perception or SyntheticPerception(
        kappa_train_max=0.02, seed=seed, base_std=(0.02, 0.01))
    odd = odd or ODD()

    lqr = LQRController(vehicle)
    shared = SharedController(SharedControlParams(v_nom=vp.v))
    driver = SimulatedDriver(DriverParams(), seed=seed + 1)
    integ = IntegrityMonitor(odd=odd, cfg=IntegrityConfig())

    A, B, E = vehicle.linear_model()
    Q_kf = np.diag([1e-4, 1e-4])
    kf = KalmanMonitor(A, B, E, Q_kf, P0=np.diag([0.05, 0.02]))
    kf.x = np.asarray(x0, float).copy()

    log = {k: [] for k in (
        "t", "e_y", "e_psi", "y_ey", "y_epsi", "xhat_ey", "xhat_epsi",
        "nis", "cov_trace", "integrity", "s_unc", "s_nis", "s_odd",
        "lambda", "v_ref", "delta", "delta_auto", "delta_human", "mode",
        "kappa", "brightness", "blur", "occlusion", "inside_odd")}

    prev_delta = 0.0
    Ts = vp.Ts

    for k, fr in enumerate(frames):
        kappa = fr.kappa
        cond = fr.cond
        driver.set_state(workload=fr.workload, available=fr.available,
                         conflict=fr.conflict)

        true_state = vehicle.x.copy()

        # --- perception -------------------------------------------------
        po = perception.measure(true_state, kappa, cond)
        y = po.mean
        R = po.cov
        cov_trace = po.trace

        # --- model-based monitor (runs for every method, for logging) ---
        kf.predict(prev_delta, kappa)
        nis = kf.update(y, R)
        x_post = kf.x.copy()

        # --- integrity score -------------------------------------------
        inside = odd.inside(kappa, cond.brightness, cond.blur, cond.occlusion)
        I, comps = integ.update(cov_trace, nis, kappa, cond.brightness,
                                cond.blur, cond.occlusion)

        # --- choose the state estimate used for control ----------------
        if method == "proposed":
            x_hat = x_post           # filtered estimate
        else:
            x_hat = y                # raw perception

        # proposed contracts speed with integrity; the LQR is scheduled on it
        v_cmd = shared.reference_speed(I) if method == "proposed" else vp.v
        ff = lqr.feedforward(kappa)
        fb_auto = lqr.feedback(x_hat, v=v_cmd)
        delta_auto = float(np.clip(ff + fb_auto, -vp.delta_max, vp.delta_max))
        delta_human = driver.command(true_state)  # human uses their own eyes

        # --- arbitration ------------------------------------------------
        if method == "naive":
            lam, v_ref, mode = 1.0, vp.v, "automation"
            delta = delta_auto
        elif method == "handover":
            # hard switch: when flagged unsafe the human takes over completely
            # (and thereby loses the automation's curvature feed-forward).
            unsafe = cov_trace > handover_sigma_thr
            lam = 0.0 if unsafe else 1.0
            v_ref = vp.v
            mode = "human" if unsafe else "automation"
            delta = delta_auto if not unsafe else delta_human
        else:  # proposed
            delta, info = shared.arbitrate(fb_auto, delta_human, ff, I,
                                           fr.workload, fr.available)
            lam, v_ref, mode = info["lambda"], info["v_ref"], info["mode"]
            delta = float(np.clip(delta, -vp.delta_max, vp.delta_max))

        # --- advance the plant at the (possibly contracted) speed -------
        vehicle.step(delta, kappa, v=v_ref)
        prev_delta = delta

        # --- record -----------------------------------------------------
        log["t"].append(k * Ts)
        log["e_y"].append(true_state[0]); log["e_psi"].append(true_state[1])
        log["y_ey"].append(y[0]); log["y_epsi"].append(y[1])
        log["xhat_ey"].append(x_hat[0]); log["xhat_epsi"].append(x_hat[1])
        log["nis"].append(nis); log["cov_trace"].append(cov_trace)
        log["integrity"].append(I)
        log["s_unc"].append(comps["uncertainty"]); log["s_nis"].append(comps["nis"])
        log["s_odd"].append(comps["odd"])
        log["lambda"].append(lam); log["v_ref"].append(v_ref)
        log["delta"].append(delta); log["delta_auto"].append(delta_auto)
        log["delta_human"].append(delta_human); log["mode"].append(mode)
        log["kappa"].append(kappa); log["brightness"].append(cond.brightness)
        log["blur"].append(cond.blur); log["occlusion"].append(cond.occlusion)
        log["inside_odd"].append(inside)

    return {k: (np.asarray(v) if k != "mode" else np.asarray(v, dtype=object))
            for k, v in log.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(log, lane_exit_thr: float = 0.7) -> dict:
    """Scientific metrics summarising one episode."""
    e_y = log["e_y"]
    delta = log["delta"]
    lam = log["lambda"]
    inside = log["inside_odd"].astype(bool)

    rmse = float(np.sqrt(np.mean(e_y ** 2)))
    max_abs = float(np.max(np.abs(e_y)))
    # a lane exit is counted per crossing (rising edge above threshold)
    over = np.abs(e_y) > lane_exit_thr
    lane_exits = int(np.sum(over[1:] & ~over[:-1])) + int(over[0])
    jerk = float(np.sum(np.abs(np.diff(delta))))

    # "unsafe AI usage": automation trusted (lambda > 0.5) while outside the ODD
    trusted = lam > 0.5
    unsafe_usage = float(np.mean(trusted & ~inside)) * 100.0

    auth_auto = float(np.mean(lam > 0.75)) * 100.0
    auth_shared = float(np.mean((lam >= 0.25) & (lam <= 0.75))) * 100.0
    auth_human = float(np.mean(lam < 0.25)) * 100.0

    return {
        "RMSE_ey": rmse,
        "max_ey": max_abs,
        "lane_exits": lane_exits,
        "steering_jerk": jerk,
        "unsafe_ai_usage_pct": unsafe_usage,
        "auth_automation_pct": auth_auto,
        "auth_shared_pct": auth_shared,
        "auth_human_pct": auth_human,
    }
