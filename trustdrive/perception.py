"""Perception backends: a synthetic uncertain sensor and an optional CNN ensemble.

Two interchangeable backends expose the same interface::

    measure(true_state, kappa, cond) -> PerceptionOutput(mean, cov, members)

``SyntheticPerception`` is a fast, dependency-free backend for the closed loop;
``CNNPerception`` is a drop-in replacement backed by an ensemble of small CNNs
and requires :mod:`torch`.

As the operating condition leaves the ODD, the synthetic sensor's true error
grows and it becomes mildly over-confident: the reported covariance grows more
slowly than the real error and a systematic bias appears. In that regime
epistemic uncertainty is insufficient on its own and the model-based NIS
provides the additional detection signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .road import RoadConditions, RoadRenderer, image_features
from .uncertainty import ensemble_stats, total_variance


@dataclass
class PerceptionOutput:
    mean: np.ndarray            # [e_y, e_psi] estimate
    cov: np.ndarray             # 2x2 reported covariance (measurement noise R_AI)
    members: np.ndarray | None = None  # (M, 2) ensemble members, if available

    @property
    def trace(self) -> float:
        return total_variance(self.cov)


class SyntheticPerception:
    """Analytic stand-in for a learned perception front-end.

    The reported covariance degrades with blur, darkness, occlusion and, less
    steeply, curvature.  Out-of-distribution curvature additionally injects a
    systematic bias that the reported covariance does *not* account for.
    """

    def __init__(self, kappa_train_max: float = 0.02, seed: int | None = None,
                 base_std=(0.02, 0.01)):
        self.kappa_train_max = kappa_train_max
        self.base_std = np.asarray(base_std, float)
        self.rng = np.random.default_rng(seed)

    def _reported_std(self, cond: RoadConditions, kappa: float) -> np.ndarray:
        blur_term = 1.0 + 2.5 * cond.blur
        dark_term = 1.0 + 1.5 * max(0.0, 1.0 - cond.brightness)
        occ_term = 1.0 + 3.0 * cond.occlusion
        # in-distribution curvature only mildly increases reported uncertainty
        kap_term = 1.0 + 4.0 * max(0.0, abs(kappa) - self.kappa_train_max)
        return self.base_std * blur_term * dark_term * occ_term * kap_term

    def measure(self, true_state, kappa, cond: RoadConditions | None = None):
        cond = cond or RoadConditions()
        std_rep = self._reported_std(cond, kappa)

        # the *actual* error is larger than reported when out of distribution
        ood = max(0.0, abs(kappa) - self.kappa_train_max) / max(self.kappa_train_max, 1e-6)
        std_true = std_rep * (1.0 + 1.2 * ood)
        bias = np.array([0.18 * ood * np.sign(kappa if kappa != 0 else 1.0), 0.05 * ood])

        noise = self.rng.normal(0.0, std_true)
        mean = np.asarray(true_state, float) + bias + noise
        cov = np.diag(std_rep ** 2)

        # emulate a 3-member ensemble consistent with the reported covariance
        members = mean + self.rng.normal(0.0, std_rep, size=(3, 2))
        return PerceptionOutput(mean=mean, cov=cov, members=members)


# --------------------------------------------------------------------------- #
# Optional CNN ensemble backend (torch)
# --------------------------------------------------------------------------- #
class CNNPerception:
    """Ensemble of small CNNs regressing ``[e_y, e_psi]`` from a road image.

    Requires :mod:`torch`.  Renders an image for the true state/condition, runs
    every ensemble member, and returns the ensemble mean and covariance as the
    perception estimate and (epistemic) measurement covariance.

    A small floor is added to the covariance so a confidently-wrong ensemble
    still yields a usable ``R`` for the Kalman monitor.
    """

    def __init__(self, models, renderer: RoadRenderer | None = None,
                 cov_floor=(0.015, 0.008), device: str = "cpu",
                 seed: int | None = None):
        import torch  # noqa: F401  (import-time dependency check)

        self.models = models
        self.renderer = renderer or RoadRenderer()
        self.cov_floor = np.diag(np.asarray(cov_floor, float) ** 2)
        self.device = device
        self.rng = np.random.default_rng(seed)

    @classmethod
    def load(cls, path: str, renderer: RoadRenderer | None = None,
             device: str = "cpu"):
        import torch
        from .cnn import LaneCNN

        blob = torch.load(path, map_location=device, weights_only=False)
        models = []
        for sd in blob["state_dicts"]:
            m = LaneCNN(**blob.get("arch", {}))
            m.load_state_dict(sd)
            m.eval().to(device)
            models.append(m)
        norm = blob.get("target_norm")
        obj = cls(models, renderer=renderer, device=device)
        obj.target_norm = norm  # (mean, std) used to de-normalise outputs
        return obj

    def measure(self, true_state, kappa, cond: RoadConditions | None = None):
        import torch

        cond = cond or RoadConditions()
        img = self.renderer.render(true_state[0], true_state[1], kappa, cond, self.rng)
        t = torch.from_numpy(img[None, None]).float().to(self.device)
        preds = []
        with torch.no_grad():
            for m in self.models:
                out = m(t).cpu().numpy().reshape(-1)
                preds.append(out)
        preds = np.asarray(preds)
        norm = getattr(self, "target_norm", None)
        if norm is not None:
            preds = preds * norm[1] + norm[0]
        mean, cov = ensemble_stats(preds)
        cov = cov + self.cov_floor
        return PerceptionOutput(mean=mean, cov=cov, members=preds)
