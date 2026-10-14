"""ASCII-art template: the whole window is a grid of characters.

    python mediapipe_osc.py --template ascii

The camera image is converted to text (one character per 8x14 cell, picked
by brightness), landmarks and blob outlines are typed over it, and the panel
uses ``[x]`` toggles, ``(*)`` radios and ``+---+`` frames. Everything is
rendered through a small glyph atlas (each character drawn once with OpenCV,
then copied around with numpy), so it stays fast.

Knobs at the top of the file: ``CELL`` (character size), ``RAMP`` (dark ->
bright characters), ``PHOSPHOR`` (text colour; set ``COLOR_MODE = "color"``
to tint each character with the camera colour instead).
"""

from __future__ import annotations

import cv2
import numpy as np

from mp_osc import preview
from mp_osc.themes import Theme

CELL = (8, 14)                     # width, height of one character in pixels
RAMP = " .:-=+*#%@"                # characters from dark to bright
COLOR_MODE = "green"               # "green", "amber", "white" or "color"
PHOSPHOR = {"green": (80, 255, 120), "amber": (60, 190, 255), "white": (230, 230, 230)}
BLACK = (0, 0, 0)


class Glyphs:
    """Monospace glyph atlas: 95 printable ASCII characters as alpha masks."""

    def __init__(self, cw: int = CELL[0], ch: int = CELL[1]):
        self.cw, self.ch = cw, ch
        atlas = np.zeros((95, ch, cw), dtype=np.float32)
        baseline = 16                      # row of the text baseline in the scratch canvas
        top = baseline - (ch - 4)          # the cell shows ch-4 rows above the baseline, 4 below
        for i in range(95):
            c = chr(32 + i)
            if c == " ":
                continue
            canvas = np.zeros((baseline + 12, cw + 24), dtype=np.uint8)
            cv2.putText(canvas, c, (8, baseline), cv2.FONT_HERSHEY_PLAIN, 0.9, 255, 1, cv2.LINE_AA)
            ys, xs = np.nonzero(canvas)
            if len(xs) == 0:
                continue
            glyph = canvas[top:top + ch, xs.min():xs.max() + 1]    # descenders may lose a pixel
            gh, gw = glyph.shape
            if gw > cw - 1:                                         # condense wide glyphs
                glyph = cv2.resize(glyph, (cw - 1, gh), interpolation=cv2.INTER_AREA)
                gw = cw - 1
            left = (cw - gw) // 2
            atlas[i, :gh, left:left + gw] = glyph / 255.0
        self.atlas = atlas
        self.masks = (atlas * 255).astype(np.uint8)

    def index(self, text: str) -> np.ndarray:
        codes = np.frombuffer(text.encode("ascii", "replace"), dtype=np.uint8).astype(np.int32) - 32
        return np.clip(codes, 0, 94)

    def render(self, idx: np.ndarray, colors: np.ndarray, bg=BLACK) -> np.ndarray:
        """Grid of char indices (rows, cols) + colours (rows, cols, 3) -> image."""
        rows, cols = idx.shape
        masks = self.masks[idx]                                   # (rows, cols, ch, cw) uint8
        col = colors.astype(np.uint16)[:, :, None, None, :]       # (rows, cols, 1, 1, 3)
        img = (masks[..., None].astype(np.uint16) * col) >> 8     # integer blend on black
        if any(bg):
            inv = (255 - masks)[..., None].astype(np.uint16)
            img += (inv * np.array(bg, dtype=np.uint16)) >> 8
        img = img.transpose(0, 2, 1, 3, 4).reshape(rows * self.ch, cols * self.cw, 3)
        return img.astype(np.uint8)

    def draw_text(self, img: np.ndarray, text: str, x: int, y: int, color) -> None:
        """Blit ``text`` with the top-left corner of the first cell at (x, y)."""
        h, w = img.shape[:2]
        color = np.array(color, dtype=np.float32)
        for i, ci in enumerate(self.index(text)):
            x0 = x + i * self.cw
            if x0 >= w or x0 + self.cw <= 0 or y >= h or y + self.ch <= 0:
                continue
            if ci == 0:
                continue
            x1, y1 = min(x0 + self.cw, w), min(y + self.ch, h)
            cx0, cy0 = max(x0, 0), max(y, 0)
            a = self.atlas[ci][cy0 - y:y1 - y, cx0 - x0:x1 - x0][..., None]
            region = img[cy0:y1, cx0:x1]
            region[:] = (a * color + (1 - a) * region).astype(np.uint8)


GLYPHS = Glyphs()


class Template(Theme):
    name = "ascii"
    description = "Everything typed in characters, green on black."

    cw, ch = CELL
    panel_width = 40 * cw           # 320 px = 40 columns
    margin = 2 * cw
    row = 2 * ch                    # two text rows per button row
    list_row = ch
    section_gap = 2 * ch
    status_line = ch

    fg = PHOSPHOR.get(COLOR_MODE, PHOSPHOR["green"])
    dim_fg = tuple(int(c * 0.45) for c in fg)
    bg = BLACK
    letterbox = BLACK
    text = fg
    title = fg
    dim = dim_fg
    on = fg
    off = dim_fg
    button = fg
    field = BLACK
    field_border = dim_fg
    focus = fg
    error = (60, 60, 255)
    accent = fg
    selected = fg
    mask_color = fg
    track_colors = [(255, 255, 255)]

    # ---- text through the atlas --------------------------------------------
    def text_(self, img, s, x, y, color=None, scale=None, thick=None):
        GLYPHS.draw_text(img, s, int(x), int(y) - self.ch + 3, self.text if color is None else color)

    def text_width(self, s, scale=None, thick=None):
        return len(s) * self.cw

    def fit_text(self, s, width, scale=None):
        n = max(0, width // self.cw)
        return s if len(s) <= n else s[: max(0, n - 1)] + "~"

    # ---- panel -------------------------------------------------------------
    def panel_background(self, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cols, rows = width // self.cw, height // self.ch
        self.text_(img, "+" + "-" * (cols - 2) + "+", 0, self.ch, self.dim)
        for r in range(1, rows - 1):
            self.text_(img, "|", 0, (r + 1) * self.ch, self.dim)
            self.text_(img, "|", (cols - 1) * self.cw, (r + 1) * self.ch, self.dim)
        self.text_(img, "+" + "-" * (cols - 2) + "+", 0, rows * self.ch, self.dim)
        return img

    def draw_section(self, img, w):
        cols = (self.panel_width - 2 * self.margin) // self.cw
        title = f"== {w.text.upper()} "
        self.text_(img, title + "=" * max(0, cols - len(title)), w.x, w.y, self.fg)

    def draw_label(self, img, w):
        self.text_(img, w.text, w.x, w.y, w.color or self.dim)

    def draw_note(self, img, w, text, is_error):
        self.text_(img, self.fit_text(text, w.w), w.x, w.y, self.error if is_error else self.dim)

    def _baseline(self, w):
        return w.y + (w.h + self.ch) // 2 - 3

    def draw_button(self, img, w, pressed, text=None):
        text = w.label if text is None else text
        label = f"[ {self.fit_text(text, w.w - 4 * self.cw)} ]"
        x = w.x + (w.w - len(label) * self.cw) // 2
        self.text_(img, label, x, self._baseline(w), self.dim if pressed else self.fg)

    def draw_toggle(self, img, w, on, enabled=True):
        mark = ("[x] " if on else "[ ] ") if enabled else "[-] "
        self.text_(img, mark + self.fit_text(w.label, w.w - 4 * self.cw),
                   w.x, self._baseline(w), self.fg if (on and enabled) else self.dim)

    def draw_radio(self, img, w, selected):
        self.text_(img, ("(*) " if selected else "( ) ") + self.fit_text(w.label, w.w - 4 * self.cw),
                   w.x, self._baseline(w), self.fg if selected else self.dim)

    def draw_field(self, img, w, shown, focused, cursor, error):
        cols = w.w // self.cw
        self.text_(img, w.label + ":", w.x, w.y - 2, self.dim)
        body = shown + ("_" if cursor else "")
        pad = "_" * max(0, cols - len(body)) if focused else " " * max(0, cols - len(body))
        self.text_(img, (body + pad)[:cols], w.x, self._baseline(w), self.fg if focused else self.text)
        if not focused:
            self.text_(img, "-" * cols, w.x, self._baseline(w) + self.ch - 6, self.dim)
        if error:
            self.text_(img, "! " + self.fit_text(error, w.w - 2 * self.cw), w.x, w.y + w.h + self.ch - 2, self.error)

    def draw_list(self, img, w, items, selected, offset, visible, more_above, more_below):
        rh = w.row_h
        cols = w.w // self.cw
        move = " < >" if w.can_move else ""
        gear = " @" if w.has_gear else ""
        if not items:
            self.text_(img, "  " + w.empty_text, w.x, w.y + rh - 3, self.dim)
        for i, item in enumerate(items[offset:offset + visible]):
            row = offset + i
            is_sel = row in selected
            text = self.fit_text(item, (cols - 4 - len(move) - len(gear)) * self.cw)
            line = ("> " if is_sel else "  ") + text
            tail = (move if is_sel else " " * len(move)) + gear
            line = line.ljust(cols - 2 - len(tail)) + tail
            self.text_(img, line, w.x, w.y + (i + 1) * rh - 3, self.fg if is_sel else self.dim)
            if gear:
                state = w.gear_state(row)
                color = self.error if state == "editing" else self.fg if state == "mapped" else self.dim
                self.text_(img, "@", w.x + (cols - 3) * self.cw, w.y + (i + 1) * rh - 3, color)
        if more_above:
            self.text_(img, "^", w.x + (cols - 1) * self.cw, w.y + rh - 3, self.dim)
        if more_below:
            self.text_(img, "v", w.x + (cols - 1) * self.cw, w.y + visible * rh - 3, self.dim)

    def draw_stepper(self, img, w, value):
        cols = w.w // self.cw
        middle = cols - 8
        label = self.fit_text(w.label, (middle - len(value) - 1) * self.cw)
        line = "[-] " + label + " " * max(1, middle - len(label) - len(value)) + value + " [+]"
        self.text_(img, line[:cols], w.x, self._baseline(w), self.fg)

    def draw_status(self, img, lines, x, y):
        for i, line in enumerate(lines):
            self.text_(img, self.fit_text(line, self.panel_width - 2 * self.margin), x,
                       y + self.ch + i * self.status_line, self.dim)

    # ---- preview: the camera image as characters -----------------------------
    def draw_preview(self, frame, poses, hands, faces, blobs, show_ids, draw_landmarks, mask=None, status=""):
        h, w = frame.shape[:2]
        cw, ch = self.cw, self.ch
        cols, rows = max(1, w // cw), max(1, h // ch)
        small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(grey, (3, 97))         # auto contrast: use the whole ramp
        grey = np.clip((grey - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        ramp = GLYPHS.index(RAMP)
        idx = ramp[np.clip((grey * (len(RAMP) - 1) + 0.5).astype(np.int32), 0, len(RAMP) - 1)]
        if COLOR_MODE == "color":
            colors = np.clip(small.astype(np.float32) * 1.4 + 40, 0, 255).astype(np.uint8)
        else:
            shade = (0.35 + 0.65 * grey)[..., None]
            colors = (np.array(self.fg, dtype=np.float32) * shade).astype(np.uint8)

        white = np.array((255, 255, 255), dtype=np.uint8)
        yellow = np.array((0, 230, 255), dtype=np.uint8)

        def cell(p):
            return int(p[0] * w / cw), int(p[1] * h / ch)

        def put(cx, cy, c, color):
            if 0 <= cx < cols and 0 <= cy < rows:
                idx[cy, cx] = ord(c) - 32
                colors[cy, cx] = color

        def line(a, b, color):
            (x0, y0), (x1, y1) = cell(a), cell(b)
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            dx, dy = x1 - x0, y1 - y0
            c = "-" if abs(dx) > 2 * abs(dy) else "|" if abs(dy) > 2 * abs(dx) else ("\\" if dx * dy > 0 else "/")
            t = np.arange(n + 1) / n
            xs = np.rint(x0 + dx * t).astype(np.int32)
            ys = np.rint(y0 + dy * t).astype(np.int32)
            keep = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
            idx[ys[keep], xs[keep]] = ord(c) - 32
            colors[ys[keep], xs[keep]] = color

        def skeleton(track, connections):
            for a, b in connections:
                line(track.landmarks[a], track.landmarks[b], white)
            for p in track.landmarks:
                cx, cy = cell(p)
                put(cx, cy, "o", yellow)

        def tag(track, text):
            cx, cy = cell(track.center)
            for i, c in enumerate(text):
                put(cx + 1 + i, cy - 1, c, yellow)

        for t in blobs:
            if draw_landmarks:
                pts = t.landmarks
                for i in range(len(pts)):
                    line(pts[i], pts[(i + 1) % len(pts)], white)
                for p in pts:
                    put(*cell(p), "#", white)
            if show_ids:
                tag(t, f"B{t.id}")
        for t in poses:
            if draw_landmarks:
                skeleton(t, preview.POSE_CONNECTIONS)
            if show_ids:
                tag(t, f"P{t.id}")
        for t in hands:
            if draw_landmarks:
                skeleton(t, preview.HAND_CONNECTIONS)
            if show_ids:
                tag(t, f"H{t.id}")
        for t in faces:
            if draw_landmarks:
                x0, y0, x1, y1 = t.detection.bbox()
                line((x0, y0), (x1, y0), white); line((x1, y0), (x1, y1), white)
                line((x1, y1), (x0, y1), white); line((x0, y1), (x0, y0), white)
            if show_ids:
                tag(t, f"F{t.id}")
        if status:
            for i, c in enumerate(status[:cols]):
                put(i, 0, c, white)

        out = GLYPHS.render(idx, colors)
        if out.shape[0] != h or out.shape[1] != w:       # pad to the original size
            full = np.zeros((h, w, 3), dtype=np.uint8)
            full[:out.shape[0], :out.shape[1]] = out
            out = full
        return out

    def draw_banner(self, frame, text):
        GLYPHS.draw_text(frame, text[: frame.shape[1] // self.cw], 0, 0, (255, 255, 255))
