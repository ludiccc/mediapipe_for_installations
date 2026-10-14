"""Look & feel templates for the preview window.

Everything that is *drawn* - the control panel, its widgets, the camera
image and the landmarks on it - goes through a :class:`Theme`. The base class
is the default ("classic") look; a *template* is a small Python file that
subclasses it and changes colours, fonts, geometry or whole drawing methods.
Three examples live in the ``templates/`` folder of the repository:
``minimal.py``, ``ascii.py`` and ``baroque.py``. Start a new one by copying
the closest example; the app loads it with ``--template my_file.py`` (or by
name when it is in ``templates/``), the *Look* button cycles through all of
them at run time and ``r`` reloads the current file while you edit it.

A template file must define a class ``Template(Theme)`` (or a ``THEME``
instance). The smallest possible one::

    from mp_osc.themes import Theme

    class Template(Theme):
        name = "pink"
        bg = (200, 120, 255)          # colours are (B, G, R)
        on = (120, 60, 220)

The drawing hooks receive the widget objects of ``mp_osc.ui`` (they only
carry geometry ``x, y, w, h`` and a ``label``) and the numpy image of the
panel to draw on. Nothing here touches OSC or the detectors: a template can
only change how things look.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import Sequence

import cv2
import numpy as np

from .tracking import Track

BGR = tuple[int, int, int]

TRACK_COLORS: list[BGR] = [
    (0, 200, 255), (0, 255, 120), (255, 120, 0), (255, 0, 200),
    (120, 120, 255), (255, 255, 0), (0, 255, 255), (255, 160, 160),
]


class Theme:
    """The default look. Subclass and override attributes or methods."""

    name = "classic"
    description = "Dark grey panel, green toggles - the original look."

    # ---- geometry -------------------------------------------------------
    panel_width = 270
    margin = 10
    row = 26              # height of a row of buttons (including the gap below)
    list_row = 17         # height of a row in the camera list
    section_gap = 18      # space taken by a section title
    status_line = 15      # line spacing of the status text
    panel_side = "right"  # "right" or "left" of the camera image

    # ---- colours (B, G, R) ---------------------------------------------
    bg: BGR = (38, 38, 38)
    text: BGR = (230, 230, 230)
    title: BGR = (230, 230, 230)
    dim: BGR = (150, 150, 150)
    on: BGR = (70, 170, 90)
    off: BGR = (72, 72, 72)
    button: BGR = (95, 95, 95)
    field: BGR = (22, 22, 22)
    field_border: BGR = (90, 90, 90)
    focus: BGR = (60, 200, 255)
    error: BGR = (70, 70, 230)
    accent: BGR = (60, 200, 255)
    selected: BGR = (70, 170, 90)
    letterbox: BGR = (38, 38, 38)      # around the image when the panel is taller
    track_colors: Sequence[BGR] = TRACK_COLORS
    mask_color: BGR = (0, 200, 255)    # Blobs mask inset

    # ---- fonts ----------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    title_scale = 0.46
    small_scale = 0.40
    thickness = 1
    preview_font = cv2.FONT_HERSHEY_SIMPLEX
    preview_scale = 0.6

    # ==================================================================
    # text helpers
    # ==================================================================
    def text_(self, img: np.ndarray, s: str, x: float, y: float, color: BGR | None = None,
              scale: float | None = None, thick: int | None = None) -> None:
        """Draw ``s`` with its baseline at ``y``."""
        cv2.putText(img, s, (int(x), int(y)), self.font, self.scale if scale is None else scale,
                    self.text if color is None else color, self.thickness if thick is None else thick,
                    cv2.LINE_AA)

    def text_width(self, s: str, scale: float | None = None, thick: int | None = None) -> int:
        return cv2.getTextSize(s, self.font, self.scale if scale is None else scale,
                               self.thickness if thick is None else thick)[0][0]

    def fit_text(self, s: str, width: int, scale: float | None = None) -> str:
        """Truncate ``s`` with an ellipsis so it fits in ``width`` pixels."""
        if self.text_width(s, scale) <= width:
            return s
        while s and self.text_width(s + "..", scale) > width:
            s = s[:-1]
        return s + ".."

    # ==================================================================
    # panel
    # ==================================================================
    def panel_background(self, width: int, height: int) -> np.ndarray:
        return np.full((height, width, 3), self.bg, dtype=np.uint8)

    def compose(self, frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
        """Put the camera image and the panel side by side."""
        fh, fw = frame.shape[:2]
        ph, pw = panel.shape[:2]
        h = max(fh, ph)
        out = np.full((h, fw + pw, 3), self.letterbox, dtype=np.uint8)
        if self.panel_side == "left":
            out[:ph, :pw] = panel
            out[:fh, pw:pw + fw] = frame
        else:
            out[:fh, :fw] = frame
            out[:ph, fw:fw + pw] = panel
        return out

    def panel_origin(self, frame_w: int, frame_h: int) -> tuple[int, int]:
        """Where the panel's (0, 0) lands in the composed window (for the mouse)."""
        return (0, 0) if self.panel_side == "left" else (frame_w, 0)

    def draw_section(self, img: np.ndarray, w) -> None:
        """Section title (``w.text``)."""
        self.text_(img, w.text, w.x, w.y, self.title, self.title_scale)

    def draw_label(self, img: np.ndarray, w) -> None:
        self.text_(img, w.text, w.x, w.y, w.color or self.dim, w.scale or self.small_scale)

    def draw_note(self, img: np.ndarray, w, text: str, is_error: bool) -> None:
        """One line of small dynamic text (camera scan status, errors...)."""
        self.text_(img, self.fit_text(text, w.w, 0.38), w.x, w.y, self.error if is_error else self.dim, 0.38)

    def draw_button(self, img: np.ndarray, w, pressed: bool, text: str | None = None) -> None:
        text = self.fit_text(w.label if text is None else text, w.w - 8)
        color = tuple(min(255, c + 50) for c in self.button) if pressed else self.button
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), color, -1)
        self.text_(img, text, w.x + (w.w - self.text_width(text)) / 2, w.y + w.h * 0.68)

    def draw_toggle(self, img: np.ndarray, w, on: bool, enabled: bool = True) -> None:
        """On/off button. ``enabled=False``: greyed out (the option does not apply right now)."""
        fill = (self.on if on else self.off) if enabled else self.off
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), fill, -1)
        box = (w.x + 7, w.y + w.h // 2 - 6)
        ink = self.text if enabled else self.dim
        cv2.rectangle(img, box, (box[0] + 12, box[1] + 12), ink, -1 if on else 1)
        label = w.label if enabled else self.fit_text(w.label + "  (n/a)", w.w - 30)
        self.text_(img, label, w.x + 26, w.y + w.h * 0.68, self.text if (on and enabled) else self.dim)

    def draw_radio(self, img: np.ndarray, w, selected: bool) -> None:
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), self.selected if selected else self.off, -1)
        c = (w.x + 11, w.y + w.h // 2)
        cv2.circle(img, c, 6, self.text, 1, cv2.LINE_AA)
        if selected:
            cv2.circle(img, c, 3, self.text, -1, cv2.LINE_AA)
        self.text_(img, self.fit_text(w.label, w.w - 24), w.x + 22, w.y + w.h * 0.68,
                   self.text if selected else self.dim)

    def draw_field(self, img: np.ndarray, w, shown: str, focused: bool, cursor: bool, error: str) -> None:
        self.text_(img, w.label, w.x, w.y - 5, self.dim, self.small_scale)
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), self.field, -1)
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), self.focus if focused else self.field_border, 1)
        self.text_(img, shown, w.x + 6, w.y + w.h * 0.68)
        if cursor:
            cx = w.x + 8 + self.text_width(shown)
            cv2.line(img, (cx, w.y + 5), (cx, w.y + w.h - 5), self.text, 1)
        if error:
            self.text_(img, error, w.x, w.y + w.h + 13, self.error, 0.38)

    def draw_list(self, img: np.ndarray, w, items: Sequence[str], selected: Sequence[int], offset: int,
                  visible: int, more_above: bool, more_below: bool) -> None:
        """The camera list. ``selected``: checked rows; checked rows get < > buttons when ``w.can_move``."""
        rh = w.row_h
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), self.field, -1)
        cv2.rectangle(img, (w.x, w.y), (w.x + w.w, w.y + w.h), self.field_border, 1)
        scroll = w.SCROLL_W if (more_above or more_below) else 0
        right = w.x + w.w - scroll
        gear_w = w.GEAR_W if w.has_gear else 0
        text_w = w.w - 10 - scroll - gear_w - (w.MOVE_W if w.can_move else 0)
        if not items:
            self.text_(img, w.empty_text, w.x + 6, w.y + rh * 0.75, self.dim, self.small_scale)
        for i, item in enumerate(items[offset:offset + visible]):
            y = w.y + i * rh
            row = offset + i
            is_sel = row in selected
            if is_sel:
                cv2.rectangle(img, (w.x + 1, y + 1), (right - 1, y + rh - 1), self.selected, -1)
            self.text_(img, self.fit_text(item, text_w, self.small_scale), w.x + 6, y + rh * 0.75,
                       self.text if is_sel else self.dim, self.small_scale)
            if w.has_gear:
                self.draw_gear(img, right - gear_w // 2, y + rh // 2, min(rh // 2 - 2, 7), w.gear_state(row), is_sel)
            if is_sel and w.can_move:
                self.draw_move_buttons(img, right - gear_w - w.MOVE_W, y, w.MOVE_W, rh)
        ax = w.x + w.w - 8
        if more_above:
            cv2.fillPoly(img, [np.array([(ax - 4, w.y + 8), (ax + 4, w.y + 8), (ax, w.y + 3)])], self.dim)
        if more_below:
            yb = w.y + w.h
            cv2.fillPoly(img, [np.array([(ax - 4, yb - 8), (ax + 4, yb - 8), (ax, yb - 3)])], self.dim)

    def draw_gear(self, img: np.ndarray, cx: int, cy: int, r: int, state: str, on_row_selected: bool) -> None:
        """The gear icon of a list row (opens the crop & mapping editor).

        ``state``: "" (no mapping), "mapped" (a mapping is active) or "editing".
        """
        color = self.focus if state == "editing" else self.accent if state == "mapped" else \
            (self.text if on_row_selected else self.dim)
        cv2.circle(img, (cx, cy), max(2, r - 3), color, 2, cv2.LINE_AA)
        for k in range(8):
            a = k * np.pi / 4
            x0, y0 = cx + (r - 2) * np.cos(a), cy + (r - 2) * np.sin(a)
            x1, y1 = cx + (r + 1) * np.cos(a), cy + (r + 1) * np.sin(a)
            cv2.line(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), color, 2, cv2.LINE_AA)

    def draw_move_buttons(self, img: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """The ``<`` ``>`` pair at the right of a checked list row (move left / right)."""
        cy = y + h // 2
        left = np.array([(x + 4, cy), (x + 10, cy - 4), (x + 10, cy + 4)])
        right = np.array([(x + w - 4, cy), (x + w - 10, cy - 4), (x + w - 10, cy + 4)])
        cv2.rectangle(img, (x, y + 2), (x + w, y + h - 2), self.button, -1)
        cv2.fillPoly(img, [left], self.text)
        cv2.fillPoly(img, [right], self.text)

    def draw_stepper(self, img: np.ndarray, w, value: str) -> None:
        """``[-] label value [+]``; ``w.button_w`` is the width of the - and + parts."""
        bw = w.button_w
        for x0, sign in ((w.x, "-"), (w.x + w.w - bw, "+")):
            cv2.rectangle(img, (x0, w.y), (x0 + bw, w.y + w.h), self.button, -1)
            self.text_(img, sign, x0 + (bw - self.text_width(sign)) / 2, w.y + w.h * 0.7)
        self.text_(img, w.label, w.x + bw + 6, w.y + w.h * 0.68, self.dim, self.small_scale)
        self.text_(img, value, w.x + w.w - bw - 6 - self.text_width(value), w.y + w.h * 0.68)

    def draw_status(self, img: np.ndarray, lines: Sequence[str], x: int, y: int) -> None:
        for i, line in enumerate(lines):
            self.text_(img, line, x, y + 12 + i * self.status_line, self.dim, self.small_scale)

    # ==================================================================
    # preview (camera image)
    # ==================================================================
    def track_color(self, track_id: int) -> BGR:
        return self.track_colors[track_id % len(self.track_colors)]

    def draw_preview(self, frame: np.ndarray, poses: list[Track], hands: list[Track], faces: list[Track],
                     blobs: list[Track], show_ids: bool, draw_landmarks: bool,
                     mask: np.ndarray | None = None, status: str = "") -> np.ndarray:
        """Draw the detections on the camera image (in place or on a copy) and return it."""
        from . import preview
        return preview.draw(frame, poses, hands, faces, blobs, show_ids, draw_landmarks, status,
                            theme=self, mask=mask)

    def draw_banner(self, frame: np.ndarray, text: str) -> None:
        """A line of text at the top of the image (e.g. 'learning background 40%')."""
        cv2.putText(frame, text, (10, 22), self.preview_font, self.preview_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 22), self.preview_font, self.preview_scale, (255, 255, 255), 1, cv2.LINE_AA)


# ======================================================================
# helpers for template authors
# ======================================================================
def rounded_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, color: BGR,
                 thickness: int = -1) -> None:
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                            ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)


def vertical_gradient(width: int, height: int, top: BGR, bottom: BGR) -> np.ndarray:
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    img = (1 - t) * np.array(top, np.float32) + t * np.array(bottom, np.float32)
    return np.repeat(img, width, axis=1).astype(np.uint8)


def vignette(frame: np.ndarray, strength: float = 0.6, _cache: dict = {}) -> np.ndarray:
    """Darken the corners of ``frame`` (in place)."""
    h, w = frame.shape[:2]
    key = (h, w, strength)
    mask = _cache.get(key)
    if mask is None:
        ys = (np.arange(h, dtype=np.float32) - h / 2) / (h / 2)
        xs = (np.arange(w, dtype=np.float32) - w / 2) / (w / 2)
        d = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) / np.sqrt(2)
        mask = np.clip(1.0 - strength * d ** 2, 0.0, 1.0)[:, :, None]
        _cache[key] = mask
    frame[:] = (frame.astype(np.float32) * mask).astype(np.uint8)
    return frame


SEPIA = np.array([[0.131, 0.534, 0.272],       # BGR in, BGR out
                  [0.168, 0.686, 0.349],
                  [0.189, 0.769, 0.393]], dtype=np.float32)


def sepia(frame: np.ndarray) -> np.ndarray:
    return np.clip(cv2.transform(frame, SEPIA), 0, 255).astype(np.uint8)


# ======================================================================
# loading templates
# ======================================================================
BUILTIN = "classic"


def available_templates(templates_dir: str) -> list[str]:
    """``["classic"]`` + the ``*.py`` files of ``templates_dir`` (without extension)."""
    names = [BUILTIN]
    if templates_dir and os.path.isdir(templates_dir):
        for fn in sorted(os.listdir(templates_dir)):
            if fn.endswith(".py") and not fn.startswith("_"):
                names.append(fn[:-3])
    return names


def template_path(name: str, templates_dir: str) -> str | None:
    """Resolve a template name or path to a file; None for the built-in look."""
    if name in ("", BUILTIN, None):
        return None
    if os.path.isfile(name):
        return os.path.abspath(name)
    candidate = os.path.join(templates_dir or "", name + ".py")
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    candidate = os.path.join(templates_dir or "", name)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"template {name!r} not found: not a file and not in {templates_dir!r} "
                            f"(available: {', '.join(available_templates(templates_dir))})")


_counter = [0]


def load_theme(name: str = BUILTIN, templates_dir: str = "") -> Theme:
    """Instantiate the template ``name`` (a name in ``templates_dir`` or a path).

    The file is executed fresh every time, so calling this again picks up
    edits (that is what the ``r`` key does).
    """
    path = template_path(name, templates_dir)
    if path is None:
        return Theme()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:          # so that `from mp_osc.themes import Theme` works in the file
        sys.path.insert(0, root)
    _counter[0] += 1
    module_name = f"mp_osc_template_{os.path.splitext(os.path.basename(path))[0]}_{_counter[0]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load template {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    theme = getattr(module, "THEME", None)
    if theme is None:
        cls = getattr(module, "Template", None)
        if cls is None:
            for value in vars(module).values():
                if isinstance(value, type) and issubclass(value, Theme) and value is not Theme:
                    cls = value
                    break
        if cls is None:
            raise ImportError(f"template {path!r} defines neither a Template class nor THEME")
        theme = cls()
    if not isinstance(theme, Theme):
        raise ImportError(f"template {path!r}: THEME must be a Theme instance")
    if not getattr(theme, "name", ""):
        theme.name = os.path.splitext(os.path.basename(path))[0]
    theme.path = path  # type: ignore[attr-defined]
    theme.loaded_at = time.time()  # type: ignore[attr-defined]
    return theme
