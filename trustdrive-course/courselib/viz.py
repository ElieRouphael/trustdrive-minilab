"""Small plotting helpers shared by the notebooks (matplotlib only)."""
from __future__ import annotations
import matplotlib.pyplot as plt

METHOD_COLORS = {"naive": "#d1495b", "handover": "#edae49", "proposed": "#2e7d32"}


def use_course_style():
    """A clean, consistent look for every notebook figure."""
    plt.rcParams.update({
        "figure.figsize": (8, 3.2),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "font.size": 10,
    })


def trust_panel(log, title=""):
    """Four stacked axes: e_y, sqrt(tr Sigma), NIS, integrity -- the core view."""
    import numpy as np
    from .monitor import CHI2_2DOF
    t = log["t"]
    fig, ax = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t, log["e_y"], color="#1f77b4", label="true $e_y$")
    ax[0].plot(t, log["y_ey"], color="#999", lw=0.8, alpha=.7, label="perceived")
    for s in (0.7, -0.7):
        ax[0].axhline(s, ls=":", color="r", lw=.8)
    ax[0].set_ylabel("$e_y$ [m]"); ax[0].legend(loc="upper right")
    ax[1].plot(t, np.sqrt(np.maximum(log["cov_trace"], 0)), color="#8e44ad")
    ax[1].set_ylabel(r"$\sqrt{tr\,\Sigma}$")
    ax[2].plot(t, log["nis"], color="#e67e22")
    ax[2].axhline(CHI2_2DOF[0.99], ls="--", color="k", lw=.9, label=r"$\chi^2_{2,0.99}$")
    ax[2].set_ylabel("NIS"); ax[2].legend(loc="upper right")
    ax[3].plot(t, log["integrity"], color=METHOD_COLORS["proposed"])
    ax[3].set_ylabel("integrity $I$"); ax[3].set_ylim(-.05, 1.05); ax[3].set_xlabel("time [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax
