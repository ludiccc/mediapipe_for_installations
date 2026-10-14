"""Blob detection with plain OpenCV - shapes instead of bodies.

For a camera on the ceiling (or any view where MediaPipe's body models do not
apply: top views, silhouettes, objects on a table, a Kinect depth map) the
``Blobs`` detector finds the connected shapes that stand out from the
background and hands them to the same tracker as the other detectors, so every
blob gets a stable id, ``enter`` / ``leave`` events and a velocity.

It is the OpenCV version of the Processing + Kinect sketch in
``Sonido_Cuadrafonico/ejemplo_instalacion``: capture the empty scene as the
background, subtract it from every frame, threshold the difference, keep the
connected components that are big enough, follow them by nearest centroid.

Modes (``--blob-mode``):

* ``background`` (default): difference against a learned static background.
  The background is the average of the first ``learn_frames`` frames (the
  scene should be empty then); ``learn()`` / the *Learn bg* button captures
  it again. ``adapt`` > 0 lets the background slowly follow lighting changes
  (only where nothing is detected).
* ``motion``: OpenCV's MOG2 adaptive background model. Needs no calibration but
  things that stop moving fade into the background after some seconds.
* ``bright`` / ``dark``: fixed threshold on the grey level - light objects on a
  dark floor, dark objects on a light floor, or a Kinect depth image where
  near = bright.

The pipeline is: grey -> blur -> mask (per mode) -> morphological open/close
-> contours -> filter by area -> sort by area -> ``max_blobs`` largest.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BLOB_MODES = ("background", "motion", "bright", "dark")
CONTOUR_POINTS = 16      # every blob's contour is resampled to this many points


@dataclass
class BlobDetection:
    """One connected shape. All coordinates are normalized to the image (0..1)."""

    center: np.ndarray         # (x, y, 0.0) centroid (image moments)
    landmarks: np.ndarray      # (CONTOUR_POINTS, 2) contour resampled, starts at the topmost point
    box: np.ndarray            # (xmin, ymin, xmax, ymax)
    area: float                # fraction of the image covered by the blob (0..1)
    angle: float               # orientation of the main axis, degrees (-90..90, clockwise on screen)
    size: np.ndarray           # (length, width) of the equivalent ellipse, normalized, length >= width
    pixels: int                # raw pixel count of the contour area

    def bbox(self) -> np.ndarray:
        return self.box


def resample_contour(contour: np.ndarray, n: int = CONTOUR_POINTS) -> np.ndarray:
    """``n`` points evenly spaced along the closed polyline ``contour`` ((k, 2) floats).

    The result starts at the topmost point (smallest y, then smallest x) so
    consecutive frames produce comparable point lists.
    """
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    start = int(np.lexsort((pts[:, 0], pts[:, 1]))[0])
    pts = np.roll(pts, -start, axis=0)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(targets, cum, closed[:, 0])
    ys = np.interp(targets, cum, closed[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _orientation(m: dict) -> tuple[float, float, float]:
    """Main-axis angle (degrees) and the axes of the equivalent ellipse from image moments."""
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))
    common = 0.5 * (mu20 + mu02)
    diff = 0.5 * np.sqrt(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2)
    long_axis = 4.0 * np.sqrt(max(common + diff, 0.0))
    short_axis = 4.0 * np.sqrt(max(common - diff, 0.0))
    return float(angle), float(long_axis), float(short_axis)


class BlobDetector:
    def __init__(
        self,
        mode: str = "background",
        threshold: int = 25,
        min_area: float = 0.004,
        max_area: float = 0.5,
        max_blobs: int = 8,
        blur: int = 5,
        morph: int = 5,
        learn_frames: int = 30,
        adapt: float = 0.0,
        roi: tuple[float, float, float, float] | None = None,
        contour_points: int = CONTOUR_POINTS,
    ):
        if mode not in BLOB_MODES:
            raise ValueError(f"unknown blob mode {mode!r}; choose from {BLOB_MODES}")
        self.mode = mode
        self.threshold = int(threshold)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.max_blobs = int(max_blobs)
        self.blur = int(blur)
        self.morph = int(morph)
        self.learn_frames = max(1, int(learn_frames))
        self.adapt = float(adapt)
        self.roi = roi
        self.contour_points = int(contour_points)

        self._background: np.ndarray | None = None     # float32 grey
        self._last_grey: np.ndarray | None = None
        self._accum: np.ndarray | None = None
        self._accum_n = 0
        self._learning = mode == "background"
        self._mog: cv2.BackgroundSubtractor | None = None
        self.mask: np.ndarray | None = None             # last binary mask (uint8, for the preview)
        self.frames = 0

    # ------------------------------------------------------------------
    @property
    def learning(self) -> bool:
        """True while the background is being captured (detections are empty meanwhile)."""
        return self._learning

    @property
    def learn_progress(self) -> float:
        return min(1.0, self._accum_n / self.learn_frames) if self._learning else 1.0

    @property
    def background(self) -> np.ndarray | None:
        return None if self._background is None else self._background.astype(np.uint8)

    def learn(self) -> None:
        """Capture the background again from the next ``learn_frames`` frames (mode 'background')."""
        self._background = None
        self._accum = None
        self._accum_n = 0
        self._learning = self.mode == "background"
        self._mog = None

    def set_mode(self, mode: str) -> None:
        if mode not in BLOB_MODES:
            raise ValueError(f"unknown blob mode {mode!r}; choose from {BLOB_MODES}")
        self.mode = mode
        self.learn()

    # ------------------------------------------------------------------
    def _grey(self, frame_bgr: np.ndarray) -> np.ndarray:
        grey = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = self.blur | 1 if self.blur > 0 else 0
        if k >= 3:
            grey = cv2.GaussianBlur(grey, (k, k), 0)
        return grey

    def _mask(self, grey: np.ndarray) -> np.ndarray | None:
        thr = max(1, self.threshold)
        if self.mode == "background":
            g = grey.astype(np.float32)
            if self._learning:
                if self._accum is None or self._accum.shape != g.shape:
                    self._accum = np.zeros_like(g)
                    self._accum_n = 0
                self._accum += g
                self._accum_n += 1
                if self._accum_n >= self.learn_frames:
                    self._background = self._accum / self._accum_n
                    self._learning = False
                    self._accum = None
                return None
            if self._background is None or self._background.shape != g.shape:
                self.learn()
                return None
            diff = cv2.absdiff(g, self._background)
            self._last_grey = g
            return (diff > thr).astype(np.uint8) * 255
        if self.mode == "motion":
            if self._mog is None:
                self._mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=float(thr),
                                                               detectShadows=False)
            return self._mog.apply(grey)
        if self.mode == "bright":
            return cv2.threshold(grey, thr, 255, cv2.THRESH_BINARY)[1]
        return cv2.threshold(grey, thr, 255, cv2.THRESH_BINARY_INV)[1]   # dark

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        k = self.morph | 1 if self.morph > 0 else 0
        if k >= 3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _apply_roi(self, mask: np.ndarray) -> np.ndarray:
        if not self.roi:
            return mask
        h, w = mask.shape[:2]
        x0, y0, x1, y1 = self.roi
        x0, x1 = sorted((int(x0 * w), int(x1 * w)))
        y0, y1 = sorted((int(y0 * h), int(y1 * h)))
        out = np.zeros_like(mask)
        out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        return out

    def process(self, frame_bgr: np.ndarray) -> list[BlobDetection]:
        """Detect the blobs of one frame; biggest first, at most ``max_blobs``."""
        self.frames += 1
        h, w = frame_bgr.shape[:2]
        grey = self._grey(frame_bgr)
        mask = self._mask(grey)
        if mask is None:
            self.mask = np.zeros((h, w), dtype=np.uint8)
            return []
        mask = self._apply_roi(self._clean(mask))
        self.mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(w * h)
        min_px = self.min_area * image_area
        max_px = self.max_area * image_area
        found: list[tuple[float, np.ndarray]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_px or area > max_px:
                continue
            found.append((area, c))
        found.sort(key=lambda t: -t[0])

        kept = found[: self.max_blobs]
        if self.mode == "background" and self.adapt > 0.0 and self._background is not None:
            # Let the background follow slow changes (light, a moved chair)
            # everywhere except inside the blobs themselves, so a person
            # standing still is not absorbed.
            inside = np.zeros((h, w), dtype=np.uint8)
            if kept:
                cv2.drawContours(inside, [c for _, c in kept], -1, 255, -1)
            free = inside == 0
            g = self._last_grey
            self._background[free] += self.adapt * (g[free] - self._background[free])

        out: list[BlobDetection] = []
        scale = np.array([w, h], dtype=np.float32)
        for area, c in kept:
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            x, y, bw, bh = cv2.boundingRect(c)
            angle, rw, rh = _orientation(m)
            out.append(BlobDetection(
                center=np.array([cx / w, cy / h, 0.0], dtype=np.float32),
                landmarks=resample_contour(c, self.contour_points) / scale,
                box=np.array([x / w, y / h, (x + bw) / w, (y + bh) / h], dtype=np.float32),
                area=float(area / image_area),
                angle=float(angle),
                size=np.array([rw / w, rh / h], dtype=np.float32),
                pixels=int(area),
            ))
        return out
