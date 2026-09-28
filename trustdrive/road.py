"""Procedural forward-camera road renderer and image-degradation utilities.

We synthesise small grayscale "camera" images of a lane so that a CNN can be
trained to regress the lane error ``[e_y, e_psi]`` directly from pixels, and so
that we can *deliberately degrade* the images (blur, darkness, occlusion, noise)
to leave the perception system's operational design domain (ODD).

A simple pinhole projection is used. A ground point at longitudinal distance
``s`` ahead of the camera and lateral offset ``y`` projects to::

    u = cx + f * y / s        (image column)
    v = cy + f * h / s        (image row, h = camera height)

The lane centreline, expressed in the vehicle frame, is approximated by a second
order expansion in the look-ahead distance ``s``::

    y_road(s) = -e_y - e_psi * s + 0.5 * kappa * s^2

Only numpy is required; the Gaussian blur is implemented with a separable
kernel so the package has no scipy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoadConditions:
    """Environmental / sensor condition for one rendered frame.

    These are exactly the quantities the ODD monitor watches.
    """

    brightness: float = 1.0     # global illumination gain in (0, 1]
    blur: float = 0.0           # Gaussian blur sigma in pixels
    occlusion: float = 0.0      # fraction of image height occluded from the top of the road [0, 1]
    noise: float = 0.02         # additive Gaussian pixel-noise std
    lane_width: float = 3.5     # metres between the two lane markings


@dataclass
class CameraParams:
    H: int = 64
    W: int = 64
    f: float = 55.0             # focal length [px]
    cam_height: float = 1.25    # camera height above the road [m]
    horizon_frac: float = 0.34  # image row of the horizon as a fraction of H
    s_near: float = 3.0         # nearest rendered look-ahead distance [m]
    s_far: float = 45.0         # farthest rendered look-ahead distance [m]


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (reflect padding), numpy only."""
    if sigma <= 1e-3:
        return img
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2
    # horizontal pass
    padded = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * padded[:, i:i + img.shape[1]]
    # vertical pass
    padded = np.pad(out, ((r, r), (0, 0)), mode="reflect")
    res = np.zeros_like(img)
    for i, w in enumerate(k):
        res += w * padded[i:i + img.shape[0], :]
    return res


class RoadRenderer:
    """Render a lane image for a given error state and road/sensor condition."""

    def __init__(self, cam: CameraParams | None = None):
        self.cam = cam or CameraParams()

    # ------------------------------------------------------------------ #
    def _project(self, s: np.ndarray, y: np.ndarray):
        """Project ground points ``(s, y)`` to image coordinates ``(u, v)``."""
        cam = self.cam
        cx = cam.W / 2.0
        cy = cam.horizon_frac * cam.H
        u = cx + cam.f * y / s
        v = cy + cam.f * cam.cam_height / s
        return u, v

    def render(self, e_y: float, e_psi: float, kappa: float,
               cond: RoadConditions | None = None,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """Return an ``H x W`` float32 grayscale image in ``[0, 1]``."""
        cam = self.cam
        cond = cond or RoadConditions()
        rng = rng or np.random.default_rng()

        img = np.full((cam.H, cam.W), 0.15, dtype=np.float32)  # dark asphalt

        s = np.linspace(cam.s_near, cam.s_far, 400)
        y_center = -e_y - e_psi * s + 0.5 * kappa * s ** 2
        half = cond.lane_width / 2.0

        # line brightness fades with distance (far markings are dimmer)
        fade = np.clip(1.0 - (s - cam.s_near) / (cam.s_far - cam.s_near), 0.2, 1.0)

        for lane_off, dashed in ((-half, False), (half, False), (0.0, True)):
            y = y_center + lane_off
            u, v = self._project(s, y)
            col = np.round(u).astype(int)
            row = np.round(v).astype(int)
            for i in range(len(s)):
                if dashed and (int(s[i]) % 3 == 0):
                    continue  # centre line is dashed
                c, r = col[i], row[i]
                if 0 <= r < cam.H and 0 <= c < cam.W:
                    # draw a small vertical smear so the line has width
                    intensity = 0.9 * fade[i]
                    for dc in (-1, 0, 1):
                        cc = c + dc
                        if 0 <= cc < cam.W:
                            img[r, cc] = max(img[r, cc], intensity)

        # ---- apply the environmental / sensor degradations -------------
        img = img * cond.brightness

        if cond.occlusion > 0.0:
            # occlude the far part of the road (top band below the horizon)
            top = int(cam.horizon_frac * cam.H)
            band = int(cond.occlusion * (cam.H - top))
            img[top:top + band, :] = 0.05

        if cond.blur > 0.0:
            img = gaussian_blur(img, cond.blur)

        if cond.noise > 0.0:
            img = img + rng.normal(0.0, cond.noise, size=img.shape)

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def render_batch(self, states, kappas, conds, rng=None):
        """Vectorised convenience wrapper returning ``(N, H, W)``."""
        rng = rng or np.random.default_rng()
        out = np.empty((len(states), self.cam.H, self.cam.W), dtype=np.float32)
        for i, (x, k, c) in enumerate(zip(states, kappas, conds)):
            out[i] = self.render(x[0], x[1], k, c, rng)
        return out


def image_features(img: np.ndarray) -> dict:
    """Cheap image statistics used by the ODD monitor when only pixels are known.

    Returns estimated brightness and a blur proxy (Tenengrad-style sharpness:
    low gradient energy => blurred / low information image).
    """
    brightness = float(img.mean())
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)
    sharpness = float(np.mean(gx ** 2) + np.mean(gy ** 2))
    return {"brightness": brightness, "sharpness": sharpness}
