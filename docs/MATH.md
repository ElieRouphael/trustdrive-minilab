# TrustDrive MiniLab - the mathematics behind the code

This document derives every equation used in the demonstrator and points to the
exact function that implements it. It is meant to be read next to the source:
each section names the file and line so you can jump straight to the code.

> GitHub renders the `$…$` / `$$…$$` LaTeX below. If you read this in a plain
> editor, the formulas are still legible as source.

**Contents**

1. [Notation and sign conventions](#1-notation-and-sign-conventions)
2. [Vehicle model](#2-vehicle-model) - `vehicle.py`
3. [Linearisation](#3-linearisation-of-the-error-dynamics) - `vehicle.py`
4. [LQR design](#4-lqr-lane-keeping) - `controllers.py`
5. [Curvature feed-forward](#5-curvature-feed-forward) - `vehicle.py` / `controllers.py`
6. [Gain scheduling](#6-speed-gain-scheduling-lpv) - `controllers.py`
7. [Camera projection & rendering](#7-camera-projection-and-image-rendering) - `road.py`
8. [Perception & ensemble uncertainty](#8-perception-and-epistemic-uncertainty) - `perception.py`, `uncertainty.py`
9. [Kalman predictor & NIS](#9-kalman-predictor-and-nis) - `kalman.py`
10. [ODD monitor](#10-odd-monitor) - `integrity.py`
11. [Integrity score](#11-integrity-score-fusion) - `integrity.py`
12. [Shared control](#12-shared-control-and-authority-arbitration) - `controllers.py`
13. [Driver model](#13-simulated-driver) - `driver.py`
14. [Closed-loop ordering](#14-closed-loop-ordering-per-step) - `simulation.py`
15. [Metrics](#15-metrics) - `simulation.py`

---

## 1. Notation and sign conventions

| symbol | meaning | units |
| --- | --- | --- |
| $e_y$ | lateral displacement of the vehicle from the lane centre | m |
| $e_\psi$ | heading (yaw) error relative to the lane tangent | rad |
| $x = [e_y,\ e_\psi]^\top$ | error state | - |
| $\delta$ | front steering angle (the control) | rad |
| $\kappa$ | road curvature (measured disturbance) | 1/m |
| $v$ | longitudinal speed | m/s |
| $L$ | wheelbase | m |
| $T_s$ | sample time | s |
| $\Sigma^{AI}$ | perception (epistemic) covariance | - |
| $\nu,\ S$ | Kalman innovation and its covariance | - |
| $I$ | integrity score $\in[0,1]$ | - |
| $\lambda$ | automation authority $\in[0,1]$ | - |

Positive $e_y$ = vehicle to the right of the lane centre. The subscript $k$ is
the discrete time index; $\hat{\cdot}$ denotes an estimate, $\bar{\cdot}$ an
ensemble mean.

---

## 2. Vehicle model

**Code:** [`Vehicle.dynamics`](../trustdrive/vehicle.py#L58), [`Vehicle.step`](../trustdrive/vehicle.py#L69).

The plant is a kinematic bicycle written in **road-error coordinates**. Two
things move the error state: the vehicle's heading relative to the lane, and the
mismatch between the steering-induced yaw rate and the yaw rate the road demands.

The lateral error changes with the component of velocity across the lane:

$$
\dot e_y = v \sin(e_\psi).
$$

The heading error changes with the difference between the vehicle yaw rate
$\dot\psi = \tfrac{v}{L}\tan\delta$ (kinematic bicycle) and the lane tangent's
rate of rotation $\dot\psi_\text{road} = v\,\kappa$:

$$
\dot e_\psi = \frac{v}{L}\tan\delta - v\,\kappa.
$$

A forward-Euler discretisation with step $T_s$ gives the update used in code:

$$
\boxed{\;
\begin{aligned}
e_{y,k+1}   &= e_{y,k}   + v\,\sin(e_{\psi,k})\,T_s,\\
e_{\psi,k+1}&= e_{\psi,k} + \Big(\tfrac{v}{L}\tan\delta_k - v\,\kappa_k\Big)T_s.
\end{aligned}\;}
$$

In `dynamics` the steering is first saturated, $\delta \leftarrow \operatorname{clip}(\delta,-\delta_{\max},\delta_{\max})$, matching the physical actuator limit `VehicleParams.delta_max`.

---

## 3. Linearisation of the error dynamics

**Code:** [`Vehicle.linear_model`](../trustdrive/vehicle.py#L77).

For controller and observer design we linearise about the centred, aligned
operating point $x=0,\ \delta=0$. Using $\sin(e_\psi)\approx e_\psi$ and
$\tan\delta\approx\delta$:

$$
x_{k+1} = A\,x_k + B\,\delta_k + E\,\kappa_k,
$$

$$
A = \begin{bmatrix} 1 & vT_s \\ 0 & 1 \end{bmatrix},\qquad
B = \begin{bmatrix} 0 \\ \dfrac{vT_s}{L} \end{bmatrix},\qquad
E = \begin{bmatrix} 0 \\ -vT_s \end{bmatrix}.
$$

This is a double-integrator-like structure: heading error integrates into
lateral error ($A_{12}=vT_s$), steering drives heading ($B_2$), and curvature is
a pure heading disturbance ($E_2$).

**Controllability.** The controllability matrix is

$$
\mathcal C = [\,B\ \ AB\,] =
\begin{bmatrix} 0 & \frac{v^2T_s^2}{L} \\[2pt] \frac{vT_s}{L} & \frac{vT_s}{L}\end{bmatrix},
\qquad \det\mathcal C = -\frac{v^3T_s^3}{L^2}\neq 0
$$

for $v\neq 0$, so the pair $(A,B)$ is controllable and the LQR below is
guaranteed to stabilise it. This is asserted in
[`tests/test_smoke.py::test_dlqr_stabilises`](../tests/test_smoke.py).

---

## 4. LQR lane keeping

**Code:** [`dlqr`](../trustdrive/controllers.py#L16), [`LQRController`](../trustdrive/controllers.py#L50).

We minimise the infinite-horizon quadratic cost

$$
J = \sum_{k=0}^{\infty} \big(x_k^\top Q\,x_k + \delta_k^\top R\,\delta_k\big),
\qquad Q = \operatorname{diag}(1,5),\ \ R = [\,8\,],
$$

(the weights live in [`LQRGains`](../trustdrive/controllers.py#L39)). The
optimal control is the static state feedback $\delta_k = -K x_k$ with

$$
K = (R + B^\top P B)^{-1} B^\top P A,
$$

where $P=P^\top\succeq 0$ solves the **discrete algebraic Riccati equation (DARE)**

$$
P = Q + A^\top P A - A^\top P B\,(R + B^\top P B)^{-1} B^\top P A.
$$

**How the code solves it.** Rather than pull in `scipy`, `dlqr` iterates the
Riccati recursion to its fixed point:

$$
P \leftarrow Q + A^\top P A - A^\top P B\,(R + B^\top P B)^{-1} B^\top P A,
\qquad P_0 = Q,
$$

stopping when $\max_{ij}|P^{+}_{ij}-P_{ij}| < \texttt{tol}$. For a stabilisable,
detectable system this iteration converges to the unique stabilising $P$; the
final $K$ is then formed from the converged $P$. The weights encode the
trade-off: $Q_{22}=5$ penalises heading error (ride comfort / anticipation),
$R=8$ penalises aggressive steering (smoothness).

---

## 5. Curvature feed-forward

**Code:** [`Vehicle.curvature_feedforward`](../trustdrive/vehicle.py#L90), used in [`LQRController.feedforward`](../trustdrive/controllers.py#L77).

Feedback alone leaves a steady-state error on a curve, because holding the lane
on a bend requires a nonzero steering angle. We cancel the curvature term
exactly. Setting the disturbance contribution to zero in $\dot e_\psi$:

$$
\frac{v}{L}\tan\delta_{\text{ff}} - v\,\kappa = 0
\;\Longrightarrow\;
\tan\delta_{\text{ff}} = L\,\kappa
\;\Longrightarrow\;
\boxed{\;\delta_{\text{ff}} = \arctan(L\,\kappa).\;}
$$

Note this holds in the **nonlinear** model (it uses $\tan$, not the small-angle
version), so it centres the vehicle on curves of any curvature the actuator can
reach. The total command is

$$
\delta = \delta_{\text{ff}} + \underbrace{(-K\hat x)}_{\text{feedback}}.
$$

A key design decision (see §12): $\delta_{\text{ff}}$ is treated as coming from
the **map / navigation**, not the camera, so it is applied at all times even
when perception is untrusted and authority shifts to the human.

---

## 6. Speed gain-scheduling (LPV)

**Code:** [`LQRController.gain_at`](../trustdrive/controllers.py#L68).

Because $A$ and $B$ depend on $v$, a gain designed at the nominal speed becomes
mistuned when the integrity monitor contracts the speed (§12). We therefore
schedule the gain on the current speed:

$$
K(v) = \big(R + B(v)^\top P(v) B(v)\big)^{-1} B(v)^\top P(v) A(v),
$$

re-solving the DARE for the linear model at speed $v$. Results are cached per
rounded $v$ so the online cost is negligible. This is a lightweight
**linear-parameter-varying (LPV)** scheme with $v$ as the scheduling parameter.

---

## 7. Camera projection and image rendering

**Code:** [`RoadRenderer._project`](../trustdrive/road.py#L87), [`RoadRenderer.render`](../trustdrive/road.py#L96), [`gaussian_blur`](../trustdrive/road.py#L61).

**Pinhole projection.** A ground point at longitudinal distance $s$ ahead,
lateral offset $y$, seen by a camera at height $h$ and focal length $f$ (pixels),
projects to image column/row

$$
u = c_x + f\,\frac{y}{s},\qquad
v = c_y + f\,\frac{h}{s},
$$

with $c_x = W/2$ and $c_y$ the horizon row. As $s\to\infty$, $u\to c_x$ and
$v\to c_y$: the two lane lines converge to the vanishing point, giving the
correct perspective foreshortening.

**Lane geometry.** Expressed in the vehicle frame, the lane centreline at
look-ahead $s$ is approximated to second order:

$$
y_\text{road}(s) = -e_y - e_\psi\,s + \tfrac12\,\kappa\,s^2.
$$

The constant term is the current offset, the linear term is the drift caused by
heading error, and the quadratic term is the road bend. Left/right markings sit
at $y_\text{road}(s)\pm \tfrac{w}{2}$ (lane width $w$); the renderer sweeps $s$
and paints the projected pixels.

**Degradations** (the ODD-relevant knobs, [`RoadConditions`](../trustdrive/road.py#L30)):

- brightness scales the image, $I \leftarrow b\,I$;
- occlusion zeroes a band of the far road;
- blur convolves with a **separable Gaussian** kernel
  $g(n)\propto e^{-n^2/2\sigma^2}$ applied along rows then columns (implemented
  in `gaussian_blur` so there is no `scipy` dependency);
- noise adds $\mathcal N(0,\sigma_\text{noise}^2)$ per pixel.

---

## 8. Perception and epistemic uncertainty

**Code:** [`SyntheticPerception`](../trustdrive/perception.py#L39), [`CNNPerception`](../trustdrive/perception.py#L82), [`ensemble_stats`](../trustdrive/uncertainty.py#L14).

The perception block returns an estimate $\hat x$ **and** a covariance. With an
ensemble of $M$ models producing predictions $\{\hat x^{(i)}\}_{i=1}^M$, the
mean and unbiased empirical covariance are

$$
\bar x = \frac1M\sum_{i=1}^M \hat x^{(i)},\qquad
\Sigma^{AI} = \frac{1}{M-1}\sum_{i=1}^M
(\hat x^{(i)}-\bar x)(\hat x^{(i)}-\bar x)^\top,
$$

computed in `ensemble_stats`. $\Sigma^{AI}$ is the **epistemic** uncertainty -
the disagreement between independently-initialised models - and is used as the
measurement noise $R^{AI}$ for the Kalman monitor (§9). `CNNPerception` adds a
small covariance floor so a confidently-wrong ensemble still yields a usable $R$.

**The synthetic stand-in and why it matters.** `SyntheticPerception` reproduces
the same interface analytically. Its *reported* per-axis std grows with the
degradations:

$$
\sigma_\text{rep} = \sigma_0
\underbrace{(1+2.5\,\text{blur})}_{\text{blur}}
\underbrace{(1+1.5(1-b)^+)}_{\text{darkness}}
\underbrace{(1+3\,\text{occ})}_{\text{occlusion}}
\underbrace{(1+4(|\kappa|-\kappa_\text{train})^+)}_{\text{curvature}} .
$$

Crucially, out of distribution the *true* error is larger than reported, and a
**systematic bias** appears that the reported covariance does not model. With
$\text{ood} = (|\kappa|-\kappa_\text{train})^+/\kappa_\text{train}$:

$$
\sigma_\text{true} = \sigma_\text{rep}(1+1.2\,\text{ood}),\qquad
b_\text{bias} = \big[\,0.18\,\text{ood}\,\operatorname{sign}\kappa,\ \ 0.05\,\text{ood}\,\big]^\top,
$$

$$
\hat x = x_\text{true} + b_\text{bias} + \mathcal N(0,\sigma_\text{true}^2),
\qquad R^{AI}=\operatorname{diag}(\sigma_\text{rep}^2).
$$

This over-confidence is intentional: it reproduces the regime where **uncertainty
alone is insufficient** and the model-based NIS provides the detection signal.

---

## 9. Kalman predictor and NIS

**Code:** [`KalmanMonitor.predict`](../trustdrive/kalman.py#L48), [`innovation`](../trustdrive/kalman.py#L54), [`update`](../trustdrive/kalman.py#L63).

The monitor runs the **same** linear model as the controller and treats
perception as a measurement of the full state, $C=I$.

**Time update (predict).** Using the last applied control and the measured
curvature:

$$
\hat x_{k|k-1} = A\,\hat x_{k-1} + B\,\delta_{k-1} + E\,\kappa_{k-1},\qquad
P_{k|k-1} = A\,P_{k-1}\,A^\top + Q.
$$

**Innovation.** The perception output $y_k^{AI}$ is compared with the model
prediction:

$$
\nu_k = y_k^{AI} - C\,\hat x_{k|k-1},\qquad
S_k = C\,P_{k|k-1}\,C^\top + R_k^{AI},
$$

where $R^{AI}$ comes from the perception covariance (§8).

**Measurement update.**

$$
K_k = P_{k|k-1} C^\top S_k^{-1},\quad
\hat x_k = \hat x_{k|k-1} + K_k\nu_k,\quad
P_k = (I - K_k C)\,P_{k|k-1}.
$$

**Normalised innovation squared (NIS).**

$$
\boxed{\;\mathrm{NIS}_k = \nu_k^\top S_k^{-1}\,\nu_k.\;}
$$

**Why the $\chi^2$ threshold.** If model and measurement are mutually consistent,
the innovation is zero-mean Gaussian with covariance $S_k$, i.e.
$S_k^{-1/2}\nu_k \sim \mathcal N(0,I)$. The sum of squares of $m=\dim(y)=2$
independent standard normals is chi-squared with 2 degrees of freedom, so under
the consistency hypothesis

$$
\mathrm{NIS}_k \sim \chi^2_2.
$$

A sustained $\mathrm{NIS}_k > \chi^2_{2,\,1-\alpha}$ (e.g. $\chi^2_{2,0.99}=9.21$,
[`CHI2_2DOF`](../trustdrive/kalman.py#L23)) rejects consistency - the perception
disagrees with the physics more than its own reported covariance allows. This
test fires **even when the network is confident**, so it detects the
out-of-distribution curvature where $\Sigma^{AI}$ stays flat.

---

## 10. ODD monitor

**Code:** [`ODD.margins`](../trustdrive/integrity.py#L36), [`ODD.score`](../trustdrive/integrity.py#L48).

The operational design domain is the box the perception was validated on:

$$
\mathcal D = \Big\{\,|\kappa|<\kappa_{\max},\ b>b_{\min},\
\sigma_\text{blur}<\sigma_{\max},\ o<o_{\max}\,\Big\}.
$$

Each guard is turned into a **signed normalised margin** (positive = inside):

$$
m_\kappa = 1-\frac{|\kappa|}{\kappa_{\max}},\quad
m_b = \frac{b-b_{\min}}{1-b_{\min}},\quad
m_\text{blur} = 1-\frac{\sigma_\text{blur}}{\sigma_{\max}},\quad
m_o = 1-\frac{o}{o_{\max}}.
$$

The scalar ODD sub-score uses the **worst-case** guard passed through a logistic
so the boundary is soft rather than a hard cliff:

$$
s_\text{ODD} = \sigma_L\!\big(6\,\min_j m_j\big),\qquad
\sigma_L(z)=\frac{1}{1+e^{-z}}.
$$

$s_\text{ODD}=0.5$ exactly on the nearest boundary, $\to 1$ well inside, $\to 0$
well outside.

---

## 11. Integrity score fusion

**Code:** [`IntegrityMonitor.update`](../trustdrive/integrity.py#L87) and the three sub-scores above it.

Three trust signals become three sub-scores in $[0,1]$ (1 = trustworthy):

$$
s_\text{unc} = \exp\!\Big(-\frac{\operatorname{tr}\Sigma^{AI}}{\sigma_\text{ref}}\Big),
\qquad
s_\text{NIS} = \exp\!\Big(-\frac{(\mathrm{NIS}-\chi^2_{2,0.99})^+}{c_\text{NIS}}\Big),
\qquad
s_\text{ODD}\ \text{(§10)}.
$$

- $s_\text{unc}$ decays as the ensemble variance grows.
- $s_\text{NIS}=1$ while NIS is below threshold, then decays for the *excess*
  above it ($(\cdot)^+=\max(0,\cdot)$).
- $s_\text{ODD}$ penalises leaving the validated box.

They are fused **multiplicatively** - any one signal can veto trust - and
smoothed with a first-order (EMA) filter to reject single-sample spikes:

$$
I_\text{raw} = s_\text{unc}\,s_\text{NIS}\,s_\text{ODD},\qquad
I_k = (1-\beta)\,I_{k-1} + \beta\,I_\text{raw},\quad \beta=0.4.
$$

The product is preferred over a tuned weighted sum because it keeps the score
interpretable: one can *read off which factor collapsed* $I$ rather than relying
on an optimal but opaque combination.

---

## 12. Shared control and authority arbitration

**Code:** [`SharedController.authority`](../trustdrive/controllers.py#L118), [`reference_speed`](../trustdrive/controllers.py#L132), [`arbitrate`](../trustdrive/controllers.py#L136).

**Authority law.** The automation authority $\lambda$ rises with integrity, and
is modulated by driver workload $W$:

$$
\lambda = \operatorname{clip}\Big(\,I + \rho\,W\,(I-0.5),\ \ \lambda_{\min},\ \lambda_{\max}\Big),
\qquad \rho=0.35.
$$

The workload term is signed around $I=0.5$: when integrity is decent ($I>0.5$)
and the human is loaded, keep a little *more* authority with automation; when
integrity is already poor ($I<0.5$), a loaded human should not be dumped into
control abruptly. Clipping to $[\lambda_{\min},\lambda_{\max}]=[0.05,0.95]$ means
neither agent is *ever* fully locked out. If the driver is unavailable,
$\lambda$ is floored at $0.85$ (no human to fall back on).

**Blended command.** Only the **feedback** authority is arbitrated; the
map-based feed-forward is always applied:

$$
\boxed{\;\delta = \delta_{\text{ff}} + \lambda\,\mathrm{fb}_A + (1-\lambda)\,\delta_H,\;}
\qquad \mathrm{fb}_A = -K(v)\,\hat x.
$$

At $\lambda=1$ this is pure automation $\delta_{\text{ff}}+\mathrm{fb}_A$; at
$\lambda=0$ it is feed-forward plus the human correction $\delta_{\text{ff}}+\delta_H$.
Keeping $\delta_{\text{ff}}$ always on is what prevents the vehicle from drifting
to the outside of a curve when a reactive human takes over - the failure mode a
naive hard handover suffers.

**Envelope contraction.** As integrity drops, the reference speed shrinks
linearly, and the LQR is re-scheduled to that speed (§6):

$$
v_\text{ref} = v_\text{nom}\big(0.4 + 0.6\,I\big)\in[0.4\,v_\text{nom},\ v_\text{nom}].
$$

So when neither perception nor driver is ideal, the system reduces its operating
envelope instead of blindly trusting one agent.

**Discrete mode** (for logging / display, [`_mode`](../trustdrive/controllers.py#L158)) is a
readout of the same state: `automation` ($I>0.75$), `shared` ($0.4<I\le0.75$),
`human` (low $I$, driver ok), `degrade+shared` (low $I$ **and** driver
loaded/unavailable).

---

## 13. Simulated driver

**Code:** [`SimulatedDriver.command`](../trustdrive/driver.py#L46).

The human is a **delayed, noisy proportional** lane-keeper acting on their own
(true-state) perception:

$$
\delta_H = -\,(1-\tfrac12 W)\,K_h^\top x_{k-\tau} + w_k
- \underbrace{c\,\cdot 0.4\,\operatorname{sign}(\cdot)}_{\text{optional conflict}},
\qquad w_k\sim\mathcal N\big(0,(\sigma_h(1+2W))^2\big).
$$

- $\tau$ = reaction delay in steps ($\tau=4\approx0.2$ s), applied via a ring
  buffer.
- Workload $W$ both attenuates the correction ($1-\tfrac12 W$) and inflates the
  noise ($1+2W$).
- $K_h=(0.12,0.9)$ and $\tau$ were chosen so the human is **stabilising but
  imperfect** (RMSE $\approx0.21$ m alone vs the automation's $0.035$ m) - a
  viable fallback, not a perfect one and not an unstable one.
- The conflict term lets a scenario make the human steer against the automation
  to test arbitration.

---

## 14. Closed-loop ordering (per step)

**Code:** [`run_simulation`](../trustdrive/simulation.py#L29).

Each sample $k$ executes in this order (the ordering matters - predict uses the
*previous* control, update uses the *current* measurement):

1. read scenario frame → $\kappa_k$, conditions, driver state;
2. **perceive**: $y_k, R_k, \Sigma^{AI}_k \leftarrow$ `perception.measure(x_\text{true},\kappa,\text{cond})`;
3. **predict** with $\delta_{k-1}$: $\hat x_{k|k-1},P_{k|k-1}$;
4. **update** with $y_k$: get $\hat x_k$ and $\mathrm{NIS}_k$;
5. **integrity**: $I_k \leftarrow f(\operatorname{tr}\Sigma^{AI}_k,\mathrm{NIS}_k,\text{ODD})$;
6. **state estimate for control**: filtered $\hat x_k$ (proposed) or raw $y_k$ (baselines);
7. compute $\delta_{\text{ff}}$, $\mathrm{fb}_A=-K(v_\text{ref})\hat x$, and $\delta_H$;
8. **arbitrate** → $\delta_k$, $\lambda_k$, $v_{\text{ref},k}$ (method-dependent, see below);
9. **advance the plant** at the (possibly contracted) speed: $x_{k+1}=f(x_k,\delta_k,\kappa_k;v_\text{ref})$.

The three methods differ **only** at steps 6-8:

| method | state used | authority | speed |
| --- | --- | --- | --- |
| `naive` | raw $y_k$ | $\lambda=1$ | $v_\text{nom}$ |
| `handover` | raw $y_k$ | $\lambda\in\{0,1\}$, switch on $\operatorname{tr}\Sigma^{AI}>\theta$ | $v_\text{nom}$ |
| `proposed` | filtered $\hat x_k$ | $\lambda=\Phi(I,W)$ smooth | $v_\text{ref}(I)$ |

That the hard-handover switch is on $\operatorname{tr}\Sigma^{AI}$ (reported
uncertainty) is exactly why it **fails to trigger** on the OOD-curvature
scenario, while the proposed method's NIS-driven integrity does.

---

## 15. Metrics

**Code:** [`compute_metrics`](../trustdrive/simulation.py#L155).

Given a logged episode of length $N$:

$$
\mathrm{RMSE}(e_y)=\sqrt{\frac1N\sum_k e_{y,k}^2},\qquad
\max|e_y| = \max_k |e_{y,k}|.
$$

**Lane exits** count *rising edges* over the threshold $e_\text{thr}=0.7$ m
(number of crossings, not samples):

$$
\#\text{exits} = \sum_{k\ge1}\mathbb 1[\,|e_{y,k}|>e_\text{thr}\ \wedge\ |e_{y,k-1}|\le e_\text{thr}\,].
$$

**Steering smoothness** (jerk proxy) is the total variation of the command:

$$
\text{jerk} = \sum_k |\delta_{k}-\delta_{k-1}|.
$$

**Unsafe-AI usage** - the safety-relevant headline - is the fraction of time the
automation is trusted ($\lambda>0.5$) while the vehicle is *outside* the ODD:

$$
\text{unsafe} = 100\cdot\frac1N\sum_k \mathbb 1[\lambda_k>0.5]\,\mathbb 1[x_k\notin\mathcal D]\ \ [\%].
$$

**Authority split** reports the percentage of time in automation-dominant
($\lambda>0.75$), shared ($0.25\le\lambda\le0.75$) and human-dominant
($\lambda<0.25$) regimes.

---

### Cross-reference summary

| equation | file | symbol |
| --- | --- | --- |
| nonlinear update | `vehicle.py` | $f(x,\delta,\kappa)$ |
| $A,B,E$ | `vehicle.py` | `linear_model` |
| DARE / $K$ | `controllers.py` | `dlqr` |
| $\delta_\text{ff}=\arctan(L\kappa)$ | `vehicle.py` | `curvature_feedforward` |
| $K(v)$ | `controllers.py` | `gain_at` |
| projection $u,v$ | `road.py` | `_project` |
| $\bar x,\Sigma^{AI}$ | `uncertainty.py` | `ensemble_stats` |
| $\nu,S,\mathrm{NIS}$ | `kalman.py` | `innovation` |
| ODD margins/score | `integrity.py` | `ODD` |
| $I=s_\text{unc}s_\text{NIS}s_\text{ODD}$ | `integrity.py` | `IntegrityMonitor.update` |
| $\lambda,\ v_\text{ref},\ \delta$ | `controllers.py` | `SharedController` |
| $\delta_H$ | `driver.py` | `SimulatedDriver.command` |
| metrics | `simulation.py` | `compute_metrics` |
