"""Comparative benchmark: three architectures x four scenarios.

Runs the naive / hard-handover / proposed controllers on every scenario, prints
a results table and writes the figures into ``figures/``.

Usage::

    python -m experiments.benchmark            # synthetic perception (default)
    python -m experiments.benchmark --cnn models/perception.pt   # CNN ensemble
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# allow running both as a module and as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustdrive import (SCENARIOS, METHODS, build_scenario, run_simulation,  # noqa: E402
                        compute_metrics)
from trustdrive.perception import SyntheticPerception, CNNPerception  # noqa: E402
from trustdrive.road import RoadRenderer  # noqa: E402
from trustdrive import plotting  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

_METRIC_COLS = [
    ("RMSE_ey", "RMSE e_y [m]", "{:.3f}"),
    ("max_ey", "max|e_y| [m]", "{:.3f}"),
    ("lane_exits", "lane exits", "{:d}"),
    ("steering_jerk", "steer jerk", "{:.2f}"),
    ("unsafe_ai_usage_pct", "unsafe AI [%]", "{:.1f}"),
]


def _make_perception(args, seed):
    if args.cnn:
        renderer = RoadRenderer()
        return CNNPerception.load(args.cnn, renderer=renderer)
    return SyntheticPerception(kappa_train_max=0.02, seed=seed, base_std=(0.02, 0.01))


def _fmt(code, val):
    """Format a metric, coercing to int when the code expects one."""
    if code.endswith("d}"):
        return code.format(int(round(val)))
    return code.format(val)


def markdown_table(rows, header):
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows])
              for i, h in enumerate(header)]
    def fmt(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(header), sep] + [fmt(r) for r in rows])


def run(args):
    os.makedirs(FIG_DIR, exist_ok=True)
    results = {}          # results[scenario][method] = metrics
    logs = {}             # logs[scenario][method] = log
    agg = {m: [] for m in METHODS}

    for sc in SCENARIOS:
        frames = build_scenario(sc, T=args.T)
        results[sc], logs[sc] = {}, {}
        for m in METHODS:
            perception = _make_perception(args, seed=args.seed)
            log = run_simulation(m, frames, perception=perception, seed=args.seed)
            met = compute_metrics(log)
            results[sc][m] = met
            logs[sc][m] = log
            agg[m].append(met)

    # ---- per-scenario tables ------------------------------------------
    print(f"\nTrustDrive MiniLab benchmark  (perception: {'CNN' if args.cnn else 'synthetic'})")
    header = ["method"] + [c[1] for c in _METRIC_COLS] + ["auth A/S/H [%]"]
    md_sections = []
    for sc in SCENARIOS:
        rows = []
        for m in METHODS:
            met = results[sc][m]
            cells = [m] + [_fmt(c[2], met[c[0]]) for c in _METRIC_COLS]
            cells.append(f"{met['auth_automation_pct']:.0f}/{met['auth_shared_pct']:.0f}/{met['auth_human_pct']:.0f}")
            rows.append(cells)
        table = markdown_table(rows, header)
        print(f"\n### Scenario: {sc}\n{table}")
        md_sections.append(f"### Scenario: {sc}\n\n{table}\n")

    # ---- aggregated table ---------------------------------------------
    agg_metrics = {}
    for m in METHODS:
        keys = agg[m][0].keys()
        agg_metrics[m] = {k: float(np.mean([d[k] for d in agg[m]])) for k in keys}
    rows = []
    for m in METHODS:
        met = agg_metrics[m]
        cells = [m] + [_fmt(c[2], met[c[0]]) for c in _METRIC_COLS]
        cells.append(f"{met['auth_automation_pct']:.0f}/{met['auth_shared_pct']:.0f}/{met['auth_human_pct']:.0f}")
        rows.append(cells)
    agg_table = markdown_table(rows, header)
    print(f"\n### Aggregated over all scenarios\n{agg_table}")
    md_sections.insert(0, f"### Aggregated over all scenarios\n\n{agg_table}\n")

    # ---- persist results ----------------------------------------------
    with open(os.path.join(FIG_DIR, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("# TrustDrive MiniLab - benchmark results\n\n")
        f.write(f"Perception backend: **{'CNN ensemble' if args.cnn else 'synthetic'}**\n\n")
        f.write("\n".join(md_sections))
    with open(os.path.join(FIG_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"per_scenario": results, "aggregated": agg_metrics}, f, indent=2)

    # ---- figures -------------------------------------------------------
    if not args.no_figures:
        renderer = RoadRenderer()
        plotting.plot_example_images(renderer, os.path.join(FIG_DIR, "fig2_example_images.png"))
        for sc in ("fog", "ood_curve"):
            plotting.plot_time_histories(
                logs[sc]["proposed"], f"Proposed system - {sc}",
                os.path.join(FIG_DIR, f"fig3_timehistories_{sc}.png"))
            plotting.plot_authority(
                logs[sc]["proposed"], f"Authority arbitration - {sc}",
                os.path.join(FIG_DIR, f"fig4_authority_{sc}.png"))
        plotting.plot_summary_bars(agg_metrics, os.path.join(FIG_DIR, "fig5_summary_bars.png"))
        print(f"\nFigures and tables written to: {FIG_DIR}")

    return results, agg_metrics


def main():
    ap = argparse.ArgumentParser(description="TrustDrive MiniLab benchmark")
    ap.add_argument("--cnn", type=str, default=None,
                    help="path to a trained CNN ensemble (.pt); default synthetic perception")
    ap.add_argument("--T", type=float, default=30.0, help="episode length [s]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
