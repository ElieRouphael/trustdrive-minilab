"""Small CNN for image -> lane-error regression (optional, requires torch).

This module is only imported when the CNN perception path is used, so the core
closed loop keeps running with numpy alone. ``LaneCNN`` is intentionally small;
the focus is the trustworthy use of a learned perception block in a
safety-critical loop rather than maximal vision accuracy.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - torch is optional
    raise ImportError(
        "trustdrive.cnn requires torch. Install it with `pip install torch`."
    ) from exc


class LaneCNN(nn.Module):
    """Compact conv net regressing ``[e_y, e_psi]`` from a 1xHxW image."""

    def __init__(self, channels=(16, 32, 64), n_out: int = 2):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c3, 64), nn.ReLU(), nn.Linear(64, n_out)
        )

    def forward(self, x):
        return self.head(self.features(x))
