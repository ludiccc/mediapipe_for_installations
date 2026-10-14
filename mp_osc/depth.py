"""Replace MediaPipe's z by a measured distance (Kinect depth).

MediaPipe's ``z`` is *relative*: it says how far a landmark is in front of or
behind the reference point of its own model (the hips for a body, the wrist
for a hand), in roughly the same units as x, and nothing about the distance
to the camera. A Kinect measures that distance for every pixel. With the
*Z from Kinect* toggle the app looks up, for each landmark, the depth at its
(x, y) pixel and writes it into the landmark's z - in **metres** from the
sensor, 0 when the sensor has no reading there.

Why a small window and a median: a landmark sits on the edge of the body
half of the time (fingertips, the silhouette seen from above), where the
depth image mixes person and background; the median of the valid pixels in a
5x5 window is robust to that and to the holes (0) of depth images.
"""

from __future__ import annotations

import numpy as np

from .detectors import Detections

WINDOW = 5          # landmarks: side of the square sampled around the pixel
BLOB_WINDOW = 15    # blobs: their centroid is inside a big shape, a wider window is fine


def sample_depth(depth_m: np.ndarray, xs: np.ndarray, ys: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Median depth (metres) in a ``window`` x ``window`` square around each normalized point.

    ``xs``, ``ys`` are normalized image coordinates (0..1). Points outside the
    image, or whose window holds no valid (> 0) depth, get 0.
    """
    h, w = depth_m.shape[:2]
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    n = len(xs)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    px = np.floor(xs * w).astype(np.int64)
    py = np.floor(ys * h).astype(np.int64)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not inside.any():
        return out
    r = window // 2
    offsets = np.arange(-r, r + 1)
    # (n, window, window) indices, clipped at the image border
    gx = np.clip(px[:, None, None] + offsets[None, None, :], 0, w - 1)
    gy = np.clip(py[:, None, None] + offsets[None, :, None], 0, h - 1)
    values = depth_m[gy, gx].reshape(n, -1).astype(np.float32)
    values[values <= 0] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(values, axis=1)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    out[inside] = med[inside]
    return out


def apply_depth(det: Detections, depth_m: np.ndarray) -> None:
    """Overwrite the z of every landmark / blob centre of ``det`` with the Kinect distance, in place."""
    for group in (det.poses, det.hands, det.faces):
        for d in group:
            lms = d.landmarks
            lms[:, 2] = sample_depth(depth_m, lms[:, 0], lms[:, 1])
    for b in det.blobs:
        b.center[2] = sample_depth(depth_m, b.center[:1], b.center[1:2], BLOB_WINDOW)[0]
