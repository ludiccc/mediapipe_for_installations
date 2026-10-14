"""Crop & mapping of one camera: an *input* quad and an *output* quad.

The same idea as the ``mapping_in_mapping_out`` tool of tpf-tools (Processing)
and ``multicamara.py`` (OpenCV), built into the app for every camera of the
list:

* the **input** quad says which part of the camera image is used - four
  corners on the raw picture (top-left, top-right, bottom-right,
  bottom-left), normalized 0..1. A surface seen in perspective (a floor, a
  table, a wall) can be "straightened" by putting the corners on its corners;
* the **output** quad says where that region is drawn inside the camera's
  slot of the analyzed image - four corners, normalized to the slot. What is
  outside the output quad is black.

Both together are one perspective transform (``cv2.getPerspectiveTransform``
+ ``cv2.warpPerspective``), applied to the frame *before* MediaPipe / Blobs
see it - so the OSC coordinates follow the mapping: with the floor mapped to
the whole slot, ``x``/``y`` are positions on the floor. The depth map of a
Kinect is warped the same way.

The JSON format is the one of tpf-tools, so their files can be loaded::

    {"input":  [{"x": 0.15, "y": 0.15}, {"x": 0.85, "y": 0.15}, {"x": 0.85, "y": 0.85}, {"x": 0.15, "y": 0.85}],
     "output": [{"x": 0.08, "y": 0.08}, ...]}

:class:`MappingEditor` holds the editing state (which camera, input or output
mode, selected corner, dragging) and draws the handles; it knows nothing
about windows - the app feeds it mouse positions and keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

UNIT_QUAD = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")
HANDLE_RADIUS = 14
NUDGE = 0.0025            # arrow keys, in normalized units (the tpf value)
INPUT_COLOR = (0, 170, 255)     # orange, like the tpf tool
OUTPUT_COLOR = (255, 220, 0)    # cyan


def _quad(points) -> np.ndarray:
    q = np.asarray(points, dtype=np.float32).reshape(4, 2)
    return np.clip(q, 0.0, 1.0)


@dataclass
class CameraMapping:
    """Input and output quads of one camera, normalized (0..1)."""

    input: np.ndarray = field(default_factory=lambda: UNIT_QUAD.copy())
    output: np.ndarray = field(default_factory=lambda: UNIT_QUAD.copy())

    # ---- state -------------------------------------------------------------
    @property
    def identity(self) -> bool:
        """True when nothing is cropped or moved (the frame passes untouched)."""
        return np.allclose(self.input, UNIT_QUAD) and np.allclose(self.output, UNIT_QUAD)

    def reset(self, which: str = "both") -> None:
        if which in ("both", "input"):
            self.input = UNIT_QUAD.copy()
        if which in ("both", "output"):
            self.output = UNIT_QUAD.copy()

    def copy(self) -> "CameraMapping":
        return CameraMapping(self.input.copy(), self.output.copy())

    # ---- the transform -----------------------------------------------------
    def matrix(self, width: int, height: int) -> np.ndarray:
        """3x3 homography taking input-quad pixels to output-quad pixels of a width x height frame."""
        size = np.array([width, height], dtype=np.float32)
        src = self.input * size
        dst = self.output * size
        return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))

    def apply(self, frame: np.ndarray, depth: np.ndarray | None = None):
        """Warp ``frame`` (and ``depth``) so the input quad lands on the output quad; black elsewhere."""
        if self.identity:
            return frame, depth
        h, w = frame.shape[:2]
        try:
            m = self.matrix(w, h)
        except cv2.error:
            return frame, depth
        if not np.isfinite(m).all() or abs(np.linalg.det(m)) < 1e-9:
            return frame, depth        # degenerate quad (corners on a line): leave the picture alone
        warped = cv2.warpPerspective(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        if depth is not None:
            depth = cv2.warpPerspective(depth, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return warped, depth

    # ---- JSON (tpf-tools format) -------------------------------------------
    def to_dict(self) -> dict:
        return {"input": [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in self.input],
                "output": [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in self.output]}

    @classmethod
    def from_dict(cls, data: dict) -> "CameraMapping":
        def points(key):
            pts = data.get(key)
            if pts is None:
                return UNIT_QUAD.copy()
            if len(pts) != 4:
                raise ValueError(f"{key!r} must have 4 points")
            return _quad([(p["x"], p["y"]) if isinstance(p, dict) else p for p in pts])
        return cls(points("input"), points("output"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "CameraMapping":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def apply_mappings(frame: np.ndarray, depth: np.ndarray | None, slots: Iterable[tuple[str, int, int]],
                   mappings: dict[str, CameraMapping]):
    """Apply each camera's mapping to its slot (``(spec, x0, x1)`` columns of the composed frame).

    Works in place on ``frame`` / ``depth`` and returns them.
    """
    for spec, x0, x1 in slots:
        mapping = mappings.get(spec)
        if mapping is None or mapping.identity or x1 <= x0:
            continue
        region = frame[:, x0:x1]
        dregion = depth[:, x0:x1] if depth is not None else None
        warped, dwarped = mapping.apply(region, dregion)
        frame[:, x0:x1] = warped
        if depth is not None and dwarped is not None:
            depth[:, x0:x1] = dwarped
    return frame, depth


# ============================================================================
# the editor
# ============================================================================
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 81, 82, 83, 84      # cv2.waitKey() & 0xFF on Qt/GTK (Linux)
KEY_LEFT_WIN, KEY_UP_WIN, KEY_RIGHT_WIN, KEY_DOWN_WIN = 37, 38, 39, 40   # Windows virtual keys
ARROWS = {KEY_LEFT: (-1, 0), KEY_RIGHT: (1, 0), KEY_UP: (0, -1), KEY_DOWN: (0, 1),
          KEY_LEFT_WIN: (-1, 0), KEY_RIGHT_WIN: (1, 0), KEY_UP_WIN: (0, -1), KEY_DOWN_WIN: (0, 1)}


class MappingEditor:
    """Editing state of one camera's mapping + drawing of the handles.

    ``mode`` is ``"input"`` (the raw camera picture is shown, orange quad) or
    ``"output"`` (the mapped composition is shown, cyan quad in the camera's
    slot). The app converts mouse positions to normalized coordinates of the
    picture being edited and calls :meth:`press` / :meth:`drag` /
    :meth:`release`; keys go to :meth:`key`.

    Keys (the tpf-tools ones): ``i`` / ``o`` switch mode, ``1``..``4`` select
    a corner, arrows nudge it, ``r`` reset the current quad, ``Enter`` /
    ``Esc`` close the editor.
    """

    def __init__(self, spec: str, mapping: CameraMapping, mode: str = "input"):
        self.spec = spec
        self.mapping = mapping
        self.mode = mode
        self.selected = 0
        self.dragging = False
        self.done = False

    @property
    def quad(self) -> np.ndarray:
        return self.mapping.input if self.mode == "input" else self.mapping.output

    @property
    def color(self):
        return INPUT_COLOR if self.mode == "input" else OUTPUT_COLOR

    # ---- input -------------------------------------------------------------
    def corner_at(self, nx: float, ny: float, width: int, height: int) -> int:
        """Index of the handle under the normalized point (nx, ny), or -1."""
        pts = self.quad * np.array([width, height], dtype=np.float32)
        d = np.linalg.norm(pts - np.array([nx * width, ny * height], dtype=np.float32), axis=1)
        i = int(np.argmin(d))
        return i if d[i] <= HANDLE_RADIUS * 1.5 else -1

    def press(self, nx: float, ny: float, width: int, height: int) -> bool:
        """Mouse button down at a normalized position; True when a handle was grabbed."""
        i = self.corner_at(nx, ny, width, height)
        if i < 0:
            return False
        self.selected = i
        self.dragging = True
        self.move_to(nx, ny)
        return True

    def drag(self, nx: float, ny: float) -> None:
        if self.dragging:
            self.move_to(nx, ny)

    def release(self) -> None:
        self.dragging = False

    def move_to(self, nx: float, ny: float) -> None:
        self.quad[self.selected] = (min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0))

    def nudge(self, dx: int, dy: int) -> None:
        x, y = self.quad[self.selected]
        self.move_to(x + dx * NUDGE, y + dy * NUDGE)

    def key(self, code: int) -> bool:
        """Handle a key code (``cv2.waitKey() & 0xFF``). True when consumed."""
        if code in (ord("i"), ord("I")):
            self.mode = "input"
        elif code in (ord("o"), ord("O")):
            self.mode = "output"
        elif ord("1") <= code <= ord("4"):
            self.selected = code - ord("1")
        elif code in ARROWS:
            self.nudge(*ARROWS[code])
        elif code in (ord("r"), ord("R")):
            self.mapping.reset(self.mode)
        elif code in (10, 13, 27):          # Enter / Esc
            self.done = True
        else:
            return False
        return True

    # ---- drawing -----------------------------------------------------------
    def draw_quad(self, img: np.ndarray, x0: int = 0, y0: int = 0, width: int | None = None,
                  height: int | None = None) -> None:
        """Polygon + numbered handles of the current quad on ``img`` (inside the rectangle given)."""
        h, w = img.shape[:2]
        width = w if width is None else width
        height = h if height is None else height
        pts = (self.quad * np.array([width, height], dtype=np.float32) + np.array([x0, y0])).astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, self.color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            active = i == self.selected
            cv2.circle(img, (int(x), int(y)), HANDLE_RADIUS, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), HANDLE_RADIUS, (255, 255, 255) if active else self.color,
                       3 if active else 2, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (int(x) - 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def draw_info(self, img: np.ndarray, name: str) -> None:
        title = f"EDITING {self.mode.upper()} of {name}"
        cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 560), 64), (0, 0, 0), -1)
        cv2.putText(img, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "i/o: input or output   1-4: corner   mouse/arrows: move   r: reset   Enter/Esc: done",
                    (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
