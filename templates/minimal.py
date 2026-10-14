"""Minimalist template: white panel, thin black type, no boxes.

    python mediapipe_osc.py --template minimal

Everything is text. A toggle that is on gets a filled square, a radio option
that is selected a filled dot; buttons are just underlined words; fields a
single hairline. The camera image is left untouched except for thin white
landmarks and tiny ids.

Copy this file, rename the class attributes you want to change and load it
with ``--template yourfile.py``. See mp_osc/themes.py for every hook.
"""

from __future__ import annotations

import cv2

from mp_osc import preview
from mp_osc.themes import Theme

INK = (30, 30, 30)
PAPER = (250, 250, 250)
GREY = (150, 150, 150)
LIGHT = (215, 215, 215)
RED = (60, 60, 220)


class Template(Theme):
    name = "minimal"
    description = "White panel, hairlines, nothing but type."

    panel_width = 264
    margin = 16
    row = 24
    list_row = 16
    section_gap = 24
    status_line = 14

    bg = PAPER
    letterbox = PAPER
    text = INK
    title = INK
    dim = GREY
    on = INK
    off = PAPER
    button = PAPER
    field = PAPER
    field_border = LIGHT
    focus = INK
    error = RED
    accent = INK
    selected = INK
    mask_color = (90, 90, 90)
    track_colors = [(255, 255, 255)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    title_scale = 0.42
    small_scale = 0.36
    thickness = 1
    preview_font = cv2.FONT_HERSHEY_SIMPLEX
    preview_scale = 0.45

    # ---- panel -----------------------------------------------------------
    def draw_section(self, img, w):
        # a thin rule above the title, the title in capitals
        cv2.line(img, (w.x, w.y - 14), (self.panel_width - self.margin, w.y - 14), LIGHT, 1)
        self.text_(img, w.text.upper(), w.x, w.y, GREY, 0.34)

    def draw_button(self, img, w, pressed, text=None):
        text = self.fit_text(w.label if text is None else text, w.w - 6)
        color = GREY if pressed else INK
        tw = self.text_width(text)
        x = w.x + (w.w - tw) // 2
        self.text_(img, text, x, w.y + w.h * 0.66, color)
        cv2.line(img, (x, w.y + w.h - 3), (x + tw, w.y + w.h - 3), color, 1)

    def draw_toggle(self, img, w, on, enabled=True):
        box = (w.x + 2, w.y + w.h // 2 - 5)
        ink = INK if enabled else LIGHT
        cv2.rectangle(img, box, (box[0] + 9, box[1] + 9), ink if on else GREY, -1 if on else 1)
        label = w.label if enabled else w.label + "  (n/a)"
        self.text_(img, self.fit_text(label, w.w - 20), w.x + 18, w.y + w.h * 0.66,
                   INK if (on and enabled) else GREY)

    def draw_radio(self, img, w, selected):
        c = (w.x + 7, w.y + w.h // 2)
        cv2.circle(img, c, 5, INK if selected else GREY, -1 if selected else 1, cv2.LINE_AA)
        self.text_(img, self.fit_text(w.label, w.w - 18), w.x + 18, w.y + w.h * 0.66,
                   INK if selected else GREY)

    def draw_field(self, img, w, shown, focused, cursor, error):
        self.text_(img, w.label.upper(), w.x, w.y - 4, GREY, 0.32)
        self.text_(img, shown, w.x, w.y + w.h * 0.66, INK)
        cv2.line(img, (w.x, w.y + w.h - 2), (w.x + w.w, w.y + w.h - 2), INK if focused else LIGHT, 1)
        if cursor:
            cx = w.x + 2 + self.text_width(shown)
            cv2.line(img, (cx, w.y + 4), (cx, w.y + w.h - 5), INK, 1)
        if error:
            self.text_(img, error, w.x, w.y + w.h + 11, RED, 0.34)

    def draw_list(self, img, w, items, selected, offset, visible, more_above, more_below):
        rh = w.row_h
        move_w = w.MOVE_W if w.can_move else 0
        gear_w = w.GEAR_W if w.has_gear else 0
        scroll = w.SCROLL_W if (more_above or more_below) else 0
        right = w.x + w.w - scroll
        if not items:
            self.text_(img, w.empty_text, w.x, w.y + rh * 0.75, GREY, self.small_scale)
        for i, item in enumerate(items[offset:offset + visible]):
            y = w.y + i * rh
            row = offset + i
            is_sel = row in selected
            if is_sel:
                cv2.circle(img, (w.x + 3, int(y + rh * 0.55)), 2, INK, -1, cv2.LINE_AA)
            self.text_(img, self.fit_text(item, w.w - 16 - move_w - gear_w - scroll, self.small_scale),
                       w.x + 12, y + rh * 0.75, INK if is_sel else GREY, self.small_scale)
            if w.has_gear:
                self.draw_gear(img, right - gear_w // 2, y + rh // 2, min(rh // 2 - 2, 6), w.gear_state(row), is_sel)
            if is_sel and w.can_move:
                x0 = right - gear_w - move_w
                self.text_(img, "<", x0 + 4, y + rh * 0.75, INK, self.small_scale)
                self.text_(img, ">", x0 + move_w // 2 + 4, y + rh * 0.75, INK, self.small_scale)
        if more_above:
            self.text_(img, "^", w.x + w.w - 10, w.y + 10, GREY, 0.34)
        if more_below:
            self.text_(img, "v", w.x + w.w - 10, w.y + w.h - 3, GREY, 0.34)

    def draw_stepper(self, img, w, value):
        bw = w.button_w
        self.text_(img, "-", w.x + 4, w.y + w.h * 0.66, INK, 0.5)
        self.text_(img, "+", w.x + w.w - bw + 8, w.y + w.h * 0.66, INK, 0.5)
        self.text_(img, w.label, w.x + bw, w.y + w.h * 0.66, GREY, self.small_scale)
        self.text_(img, value, w.x + w.w - bw - 4 - self.text_width(value), w.y + w.h * 0.66, INK)

    def draw_note(self, img, w, text, is_error):
        self.text_(img, self.fit_text(text, w.w, 0.34), w.x, w.y, RED if is_error else GREY, 0.34)

    def draw_status(self, img, lines, x, y):
        cv2.line(img, (x, y - 2), (self.panel_width - self.margin, y - 2), LIGHT, 1)
        for i, line in enumerate(lines):
            self.text_(img, line, x, y + 12 + i * self.status_line, GREY, 0.34)

    # ---- preview ---------------------------------------------------------
    def draw_preview(self, frame, poses, hands, faces, blobs, show_ids, draw_landmarks, mask=None, status=""):
        h, w = frame.shape[:2]
        white = (255, 255, 255)

        def pt(p):
            return int(p[0] * w), int(p[1] * h)

        def skeleton(track, connections):
            for a, b in connections:
                cv2.line(frame, pt(track.landmarks[a]), pt(track.landmarks[b]), white, 1, cv2.LINE_AA)
            for p in track.landmarks:
                cv2.circle(frame, pt(p), 2, INK, -1, cv2.LINE_AA)

        def tag(track, text):
            x, y = pt(track.center)
            cv2.putText(frame, text, (x + 6, y - 6), self.preview_font, self.preview_scale, INK, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x + 6, y - 6), self.preview_font, self.preview_scale, white, 1, cv2.LINE_AA)

        if blobs and draw_landmarks and mask is not None:
            preview.mask_inset(frame, mask, self.mask_color, 0.2)
        for t in blobs:
            if draw_landmarks:
                preview.draw_blob(frame, t, white, 1)
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
                cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), white, 1)
            if show_ids:
                tag(t, f"F{t.id}")
        if status:
            cv2.putText(frame, status, (10, 18), self.preview_font, 0.4, white, 1, cv2.LINE_AA)
        return frame

    def draw_banner(self, frame, text):
        cv2.putText(frame, text, (10, 18), self.preview_font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
