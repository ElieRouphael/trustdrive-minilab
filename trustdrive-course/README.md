# TrustDrive Course

**Trustworthy AI perception in a control loop - a hands-on, build-it-yourself course.**

Ten Jupyter notebooks that build, one concept at a time, a small autonomous
lane-keeping system that stays safe when its AI perception becomes unreliable. Each
notebook isolates a single idea, **derives the mathematics**, implements it from
scratch, demonstrates it with a runnable experiment, ends with exercises, and cites
the primary literature. A capstone composes everything into a closed loop and
evaluates it as a controlled experiment; a final notebook develops advanced methods.

The whole course runs on **numpy + matplotlib only** - no GPU, no deep-learning
framework, no simulator to install. It is self-contained and independent of any other
project.

> **Central question:** what should an autonomous vehicle do when its AI perception
> becomes uncertain, and how should control authority be shared between automation and
> the driver?

---

## Who it is for

Anyone with basic linear algebra, probability, and Python who wants to understand how
estimation, optimal control, uncertainty quantification, run-time monitoring, and
human-automation sharing fit together in a safety-critical loop. No prior
control-theory or deep-learning background is assumed - each idea is built up from
first principles.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook            # then open notebooks/00_overview.ipynb
```

Work through the notebooks in order. Each is self-contained; the capstone and the
advanced notebook additionally import `courselib`, the consolidated reference library.

## Curriculum

Every chapter pairs a derivation with a runnable result and a short reference list.

| # | Notebook | What you derive and build | A concrete result you reproduce |
|---|----------|---------------------------|---------------------------------|
| 00 | [`00_overview`](notebooks/00_overview.ipynb) | The closed-loop problem statement; why self-reported confidence is not enough; notation | Teaser: reported uncertainty flat while NIS jumps 1.3 -> 13.6 out of distribution |
| 01 | [`01_vehicle_model`](notebooks/01_vehicle_model.ipynb) | Kinematic bicycle from the no-slip constraints; Frenet error dynamics; discretisation; linearisation; controllability/observability | Euler convergence order fits **1.00**; observable from `e_y` alone; ODD limit = 0.5 g |
| 02 | [`02_lqr_control`](notebooks/02_lqr_control.ipynb) | LQR by dynamic programming; the DARE and its solvability; Bryson's rule; 2-DOF feed-forward; gain scheduling | DARE residual **7e-15**; the tracking-vs-effort cost frontier |
| 03 | [`03_perception_and_rendering`](notebooks/03_perception_and_rendering.ipynb) | Pinhole projection from similar triangles; ground-plane homography; PSF blur and noise models | An out-of-distribution curve is **invisible** in the pixel statistics |
| 04 | [`04_uncertainty_ensembles`](notebooks/04_uncertainty_ensembles.ipynb) | Aleatoric vs epistemic via the law of total variance; ensembles as a Bayesian approximation; calibration; proper scoring | Ensemble NLL **-0.37** vs single-model 11.6; the "uncertainty != error" failure |
| 05 | [`05_kalman_and_nis`](notebooks/05_kalman_and_nis.ipynb) | Kalman filter as recursive Bayes / BLUE; innovation whiteness; NIS ~ chi-squared; consistency and fault tests | Innovations white; mean NIS **2.08** ~ m; honest-R sweep |
| 06 | [`06_odd_and_integrity`](notebooks/06_odd_and_integrity.ipynb) | ODD as set membership; the three detectors; product (naive-Bayes veto) fusion; ROC and detection delay | Fused **AUC 0.96** vs ~0.70 for any single detector |
| 07 | [`07_shared_control`](notebooks/07_shared_control.ipynb) | Haptic shared control; the McRuer crossover driver model; Yerkes-Dodson workload; the authority law | Delayed-loop stability; integrity-aware conflict arbitration |
| 08 | [`08_capstone_closed_loop`](notebooks/08_capstone_closed_loop.ipynb) | The whole system as a controlled ablation; multi-seed statistics; a component ablation; limitations | Proposed vs naive: **-0.170 m RMSE, 95% CI [-0.185, -0.157]** |
| 09 | [`09_advanced_ideas`](notebooks/09_advanced_ideas.ipynb) | CUSUM change detection (SPRT, ARL); conformal prediction (coverage, ACI); a research roadmap | CUSUM beats the single-sample gate on the ARL trade-off |

## The idea you prove to yourself

Three ways of using the *same* AI perception are compared throughout:

- **Naive AI** - perception drives the controller directly; no monitoring.
- **Hard handover** - switch to the driver when reported uncertainty crosses a threshold.
- **Proposed** - fuse epistemic uncertainty, a model-based consistency test (NIS), and
  operational-domain membership into an *integrity score*, and use it to share authority
  smoothly and contract speed.

On an out-of-distribution curve the network stays *confident* - its reported
uncertainty barely moves - so the threshold method never triggers. Only the
model-based checks catch it. The capstone shows, with a controlled and
statistically-tested experiment, that using perception *through* a monitoring layer is
safer than trusting it or switching away from it.

## What makes it rigorous

- **Derivations, not assertions** - the Riccati recursion, the NIS chi-squared
  distribution, the total-variance decomposition, and the conformal coverage bound are
  each worked out.
- **Every claim is run** - orders of convergence, filter consistency, ROC/AUC, bootstrap
  confidence intervals and ablations are computed, not stated.
- **Primary sources** - roughly 90 references across the ten notebooks (Kalman,
  Bar-Shalom, Anderson-Moore, McRuer, Abbink, Lakshminarayanan, Vovk, Page, Efron, and
  the driving-specific monitoring literature).
- **Honest limitations** - the capstone states where the demonstrator is simplified
  (synthetic perception, kinematic plant, heuristic thresholds), and notebook 09 turns
  those into research directions.

## Repository layout

```
trustdrive-course/
|- README.md
|- LICENSE
|- requirements.txt
|- test_courselib.py            # smoke tests for the reference library
|- courselib/                   # clean, consolidated reference implementation
|  |- vehicle.py  control.py  perception.py
|  |- monitor.py  driver.py    sim.py   viz.py
|- notebooks/                   # 00 .. 09, the course itself
|- figures/                     # (generated)
```

The notebooks build each idea from scratch; `courselib` is the tidy version they
culminate in, used by the capstone. Run `python test_courselib.py` to check it.

## License

MIT - see [LICENSE](LICENSE).
