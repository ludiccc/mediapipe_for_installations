"""On-screen preview: draws landmarks, blobs, IDs, velocities and status text.

This is the drawing used by the default theme; templates can reuse the
helpers (``draw_blob``, ``label``, ``mask_inset``) or replace everything in
``Theme.draw_preview``.
"""

from __future__ import annotations

import cv2
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from .tracking import Track

_drawing = solutions.drawing_utils
_styles = solutions.drawing_styles

COLORS = [
    (0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200),
    (120, 120, 255), (255, 255, 0), (0, 255, 255), (255, 160, 160),
]

POSE_CONNECTIONS = solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = solutions.hands.HAND_CONNECTIONS
FACE_CONTOURS = solutions.face_mesh.FACEMESH_CONTOURS


def _proto(landmarks: np.ndarray) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend(
        [landmark_pb2.NormalizedLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in landmarks]
    )
    return proto


def label(image, track: Track, text: str, color, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6) -> None:
    """Id text, center dot and velocity arrow for one track."""
    h, w = image.shape[:2]
    x, y = int(track.center[0] * w), int(track.center[1] * h)
    cv2.putText(image, text, (x + 8, y - 8), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x + 8, y - 8), font, scale, color, 1, cv2.LINE_AA)
    cv2.circle(image, (x, y), 5, color, -1)
    # velocity arrow (normalized units/s -> pixels, scaled for readability)
    vx, vy = track.velocity[:2]
    if abs(vx) + abs(vy) > 1e-3:
        cv2.arrowedLine(image, (x, y), (int(x + vx * w * 0.25), int(y + vy * h * 0.25)), color, 2)


_label = label  # backwards compatibility


def blob_polygon(track: Track, width: int, height: int) -> np.ndarray:
    pts = np.asarray(track.landmarks)[:, :2] * np.array([width, height], dtype=np.float32)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def draw_blob(image, track: Track, color, thickness: int = 2, fill_alpha: float = 0.0) -> None:
    """Outline (and optionally a translucent fill) plus the bounding box of a blob."""
    h, w = image.shape[:2]
    poly = blob_polygon(track, w, h)
    if fill_alpha > 0.0:
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0, image)
    cv2.polylines(image, [poly], True, color, thickness, cv2.LINE_AA)
    x0, y0, x1, y1 = track.detection.bbox()
    cv2.rectangle(image, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), color, 1)


def mask_inset(image, mask: np.ndarray, color=(0, 200, 255), scale: float = 0.25) -> None:
    """Small copy of the blob mask in the bottom-left corner (to tune the threshold)."""
    if mask is None:
        return
    h, w = image.shape[:2]
    iw, ih = max(32, int(w * scale)), max(24, int(h * scale))
    small = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_AREA)
    tinted = np.zeros((ih, iw, 3), dtype=np.uint8)
    tinted[:] = (20, 20, 20)
    tinted[small > 127] = color
    y0, x0 = h - ih - 6, 6
    image[y0:y0 + ih, x0:x0 + iw] = tinted
    cv2.rectangle(image, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), color, 1)


def draw(
    image_bgr: np.ndarray,
    poses: list[Track],
    hands: list[Track],
    faces: list[Track],
    blobs: list[Track] = (),
    show_ids: bool = True,
    draw_landmarks: bool = True,
    status: str = "",
    theme=None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    img = image_bgr  # drawn in place
    colors = list(theme.track_colors) if theme is not None else COLORS
    font = theme.preview_font if theme is not None else cv2.FONT_HERSHEY_SIMPLEX
    scale = theme.preview_scale if theme is not None else 0.6

    def color_of(track_id: int):
        return colors[track_id % len(colors)]

    if blobs and draw_landmarks and mask is not None:
        mask_inset(img, mask, theme.mask_color if theme is not None else (0, 200, 255))

    for t in blobs:
        color = color_of(t.id)
        if draw_landmarks:
            draw_blob(img, t, color)
        if show_ids:
            label(img, t, f"B{t.id}", color, font, scale)

    for t in poses:
        color = color_of(t.id)
        if draw_landmarks:
            _drawing.draw_landmarks(img, _proto(t.landmarks), POSE_CONNECTIONS,
                                    _styles.get_default_pose_landmarks_style())
        if show_ids:
            label(img, t, f"P{t.id}", color, font, scale)

    for t in hands:
        color = color_of(t.id)
        if draw_landmarks:
            _drawing.draw_landmarks(img, _proto(t.landmarks), HAND_CONNECTIONS,
                                    _styles.get_default_hand_landmarks_style(),
                                    _styles.get_default_hand_connections_style())
        if show_ids:
            d = t.detection
            text = f"H{t.id} {d.handedness[:1]}"
            if d.gesture and d.gesture != "None":
                text += f" {d.gesture}"
            label(img, t, text, color, font, scale)

    for t in faces:
        color = color_of(t.id)
        if draw_landmarks:
            _drawing.draw_landmarks(img, _proto(t.landmarks), FACE_CONTOURS,
                                    None, _styles.get_default_face_mesh_contours_style())
        if show_ids:
            label(img, t, f"F{t.id}", color, font, scale)

    if status:
        cv2.putText(img, status, (10, 22), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, status, (10, 22), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img
