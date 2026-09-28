"""ODD monitoring and the integrity score.

Three independent signals are fused into a single, *transparent* integrity
score ``I in [0, 1]`` (1 = fully trustworthy automation, 0 = do not trust):

1. **Epistemic uncertainty** of the perception (ensemble covariance trace).
2. **Model consistency** via the Kalman NIS versus a chi-squared threshold.
3. **ODD membership** - whether the current operating condition lies inside the
   domain the perception was validated on.

The combination is a simple, interpretable product of three sub-scores rather
than a tuned black box, so that it is possible to read off which factor caused
the integrity to drop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kalman import CHI2_2DOF


@dataclass
class ODD:
    """Miniature operational design domain.

    A condition is inside the ODD when every guarded quantity is within range.
    """

    kappa_max: float = 0.022      # trained curvature envelope [1/m]
    brightness_min: float = 0.45  # minimum acceptable illumination
    blur_max: float = 1.6         # maximum acceptable blur sigma [px]
    occlusion_max: float = 0.35   # maximum acceptable occlusion fraction

    def margins(self, kappa, brightness, blur, occlusion) -> dict:
        """Signed, normalised slack for each guard (>=0 means inside)."""
        return {
            "kappa": 1.0 - abs(kappa) / self.kappa_max,
            "brightness": (brightness - self.brightness_min) / (1.0 - self.brightness_min),
            "blur": 1.0 - blur / self.blur_max,
            "occlusion": 1.0 - occlusion / self.occlusion_max,
        }

    def inside(self, kappa, brightness, blur, occlusion) -> bool:
        return all(m > 0.0 for m in self.margins(kappa, brightness, blur, occlusion).values())

    def score(self, kappa, brightness, blur, occlusion) -> float:
        """Smooth ODD sub-score in ``[0, 1]`` (soft margin, worst-case guard)."""
        m = self.margins(kappa, brightness, blur, occlusion)
        worst = min(m.values())
        # logistic soft threshold centred on the boundary
        return float(1.0 / (1.0 + np.exp(-6.0 * worst)))


@dataclass
class IntegrityConfig:
    sigma_ref: float = 0.02       # uncertainty trace giving ~0.5 sub-score
    nis_confidence: float = 0.99  # chi-squared confidence for the NIS guard
    nis_scale: float = 6.0        # softness of the NIS penalty above threshold
    ema: float = 0.4              # smoothing of the final score (0..1, higher=faster)


class IntegrityMonitor:
    """Fuse uncertainty, NIS and ODD membership into an integrity score."""

    def __init__(self, odd: ODD | None = None, cfg: IntegrityConfig | None = None):
        self.odd = odd or ODD()
        self.cfg = cfg or IntegrityConfig()
        self._I = 1.0

    # - individual sub-scores ---------------------------------------- #
    def uncertainty_score(self, cov_trace: float) -> float:
        """Maps ensemble-variance trace to ``[0, 1]`` (1 = confident)."""
        return float(np.exp(-cov_trace / self.cfg.sigma_ref))

    def nis_score(self, nis: float) -> float:
        """Maps NIS to ``[0, 1]`` (1 = consistent with the model)."""
        thr = CHI2_2DOF.get(self.cfg.nis_confidence, 9.210)
        excess = max(0.0, nis - thr)
        return float(np.exp(-excess / self.cfg.nis_scale))

    def odd_score(self, kappa, brightness, blur, occlusion) -> float:
        return self.odd.score(kappa, brightness, blur, occlusion)

    # - fused score --------------------------------------------------- #
    def update(self, cov_trace, nis, kappa, brightness, blur, occlusion):
        """Return ``(I, components)`` and update the internal EMA state."""
        s_unc = self.uncertainty_score(cov_trace)
        s_nis = self.nis_score(nis)
        s_odd = self.odd_score(kappa, brightness, blur, occlusion)
        raw = s_unc * s_nis * s_odd
        self._I = (1 - self.cfg.ema) * self._I + self.cfg.ema * raw
        comps = {
            "uncertainty": s_unc,
            "nis": s_nis,
            "odd": s_odd,
            "raw": raw,
            "integrity": self._I,
        }
        return float(self._I), comps

    @property
    def integrity(self) -> float:
        return float(self._I)
