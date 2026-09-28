"""Matplotlib helpers for the result figures (Agg backend, file output)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .kalman import CHI2_2DOF

_COL = {"naive": "#d1495b", "handover": "#edae49", "proposed": "#2e7d32"}


def plot_time_histories(log, title: str, path: str):
    """Figure 3: e_y, ensemble sigma, NIS and integrity over time (one method)."""
    t = log["t"]
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(t, log["e_y"], color="#1f77b4", label=r"$e_y$ (true)")
    ax[0].plot(t, log["y_ey"], color="#999", lw=0.8, alpha=0.7, label=r"$e_y$ (perceived)")
    ax[0].axhline(0.7, ls=":", color="r", lw=0.8); ax[0].axhline(-0.7, ls=":", color="r", lw=0.8)
    ax[0].set_ylabel("lateral error [m]"); ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, np.sqrt(np.maximum(log["cov_trace"], 0)), color="#8e44ad")
    ax[1].set_ylabel(r"$\sqrt{\mathrm{tr}\,\Sigma^{AI}}$")

    ax[2].plot(t, log["nis"], color="#e67e22")
    ax[2].axhline(CHI2_2DOF[0.99], ls="--", color="k", lw=0.9,
                  label=r"$\chi^2_{2,0.99}$")
    ax[2].set_ylabel("NIS"); ax[2].legend(loc="upper right", fontsize=8)

    ax[3].plot(t, log["integrity"], color=_COL["proposed"])
    ax[3].set_ylabel("integrity $I$"); ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_xlabel("time [s]")

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_authority(log, title: str, path: str):
    """Figure 4: authority lambda(t) with the human vs automation steering."""
    t = log["t"]
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    ax[0].plot(t, log["lambda"], color=_COL["proposed"], label=r"$\lambda$ (automation authority)")
    ax[0].fill_between(t, 0, log["lambda"], color=_COL["proposed"], alpha=0.12)
    ax[0].plot(t, log["v_ref"] / max(log["v_ref"].max(), 1e-6), ls="--",
               color="#555", lw=1.0, label=r"$v_{ref}$ (normalised)")
    ax[0].set_ylabel("authority / speed"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(loc="lower left", fontsize=8)

    ax[1].plot(t, log["delta_auto"], color="#2e7d32", lw=1.0, label=r"$\delta_A$ automation")
    ax[1].plot(t, log["delta_human"], color="#d1495b", lw=1.0, alpha=0.8, label=r"$\delta_H$ human")
    ax[1].plot(t, log["delta"], color="k", lw=1.2, label=r"$\delta$ applied")
    ax[1].set_ylabel("steering [rad]"); ax[1].set_xlabel("time [s]")
    ax[1].legend(loc="upper right", fontsize=8)

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example_images(renderer, path: str):
    """Figure 2: example clear / fog / out-of-distribution frames."""
    from .road import RoadConditions
    rng = np.random.default_rng(0)
    panels = [
        ("clear (in ODD)", 0.30, 0.02, 0.012, RoadConditions()),
        ("fog + blur", 0.30, 0.02, 0.012, RoadConditions(brightness=0.6, blur=2.6, noise=0.03)),
        ("partial occlusion", 0.30, 0.02, 0.012, RoadConditions(occlusion=0.45)),
        ("OOD curvature", 0.30, 0.02, 0.05, RoadConditions()),
    ]
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for a, (name, ey, epsi, kap, cond) in zip(ax, panels):
        img = renderer.render(ey, epsi, kap, cond, rng)
        a.imshow(img, cmap="gray", vmin=0, vmax=1)
        a.set_title(name, fontsize=10); a.axis("off")
    fig.suptitle("Synthetic camera frames used to train / stress the perception")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary_bars(metrics_by_method, path: str):
    """Compact bar comparison across the three methods (aggregated scenarios)."""
    methods = list(metrics_by_method.keys())
    keys = [("RMSE_ey", "RMSE $e_y$ [m]"), ("lane_exits", "lane exits"),
            ("steering_jerk", "steering jerk"), ("unsafe_ai_usage_pct", "unsafe AI use [%]")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a, (k, label) in zip(ax, keys):
        vals = [metrics_by_method[m][k] for m in methods]
        a.bar(methods, vals, color=[_COL[m] for m in methods])
        a.set_title(label, fontsize=10); a.tick_params(axis="x", labelsize=8)
        a.grid(axis="y", alpha=0.25)
    fig.suptitle("Aggregated performance across all scenarios")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
