"""Baroque template: gold on burgundy, serif and script type, gilded frames.

    python mediapipe_osc.py --template baroque

The panel is a wine-red velvet with a double gold border and corner
flourishes; toggles and buttons are gilded plaques, fields are parchment.
The camera image gets a sepia tone, a vignette and a gold frame; landmarks are
drawn in gold with bronze shadows.

All colours are (B, G, R). Change GOLD / WINE / PARCHMENT below for a
different palette, or override any ``draw_*`` method (see mp_osc/themes.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from mp_osc import preview
from mp_osc.themes import Theme, rounded_rect, sepia, vertical_gradient, vignette

GOLD = (60, 190, 235)
PALE_GOLD = (140, 225, 250)
DARK_GOLD = (20, 120, 170)
BRONZE = (30, 80, 120)
WINE = (40, 22, 70)
WINE_DARK = (22, 12, 40)
WINE_LIGHT = (70, 40, 110)
PARCHMENT = (190, 225, 245)
INK = (30, 25, 60)
CRIMSON = (40, 30, 190)


def ornament(img, x, y, size, color, flip_x=False, flip_y=False):
    """A little spiral flourish for a corner."""
    sx, sy = (-1 if flip_x else 1), (-1 if flip_y else 1)
    for r, a0 in ((size, 0), (size * 0.6, 90), (size * 0.3, 180)):
        cx, cy = int(x + sx * size), int(y + sy * size)
        cv2.ellipse(img, (cx, cy), (int(r), int(r)), 0, a0, a0 + 180 if not flip_x ^ flip_y else a0 - 180,
                    color, 1, cv2.LINE_AA)
    cv2.circle(img, (int(x + sx * size * 1.3), int(y + sy * size * 1.3)), 2, color, -1, cv2.LINE_AA)


def flourish_line(img, x0, x1, y, color):
    """Thin line with a diamond in the middle and dots at the ends."""
    cv2.line(img, (x0, y), (x1, y), color, 1, cv2.LINE_AA)
    cx = (x0 + x1) // 2
    cv2.fillPoly(img, [np.array([(cx, y - 4), (cx + 4, y), (cx, y + 4), (cx - 4, y)])], color, cv2.LINE_AA)
    cv2.circle(img, (x0, y), 2, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1, y), 2, color, -1, cv2.LINE_AA)


def plaque(img, x0, y0, x1, y1, fill, border, double=True):
    rounded_rect(img, x0, y0, x1, y1, 6, fill)
    rounded_rect(img, x0, y0, x1, y1, 6, border, 1)
    if double:
        rounded_rect(img, x0 + 3, y0 + 3, x1 - 3, y1 - 3, 4, border, 1)


def gilded_frame(img, inset=6, color=GOLD, inner=DARK_GOLD):
    h, w = img.shape[:2]
    cv2.rectangle(img, (inset, inset), (w - inset - 1, h - inset - 1), color, 2)
    cv2.rectangle(img, (inset + 5, inset + 5), (w - inset - 6, h - inset - 6), inner, 1)
    s = 10
    ornament(img, inset + 8, inset + 8, s, color)
    ornament(img, w - inset - 9, inset + 8, s, color, flip_x=True)
    ornament(img, inset + 8, h - inset - 9, s, color, flip_y=True)
    ornament(img, w - inset - 9, h - inset - 9, s, color, flip_x=True, flip_y=True)


class Template(Theme):
    name = "baroque"
    description = "Gold on burgundy, serifs, gilded frames and a sepia picture."

    panel_width = 290
    margin = 18
    row = 28
    list_row = 18
    section_gap = 26
    status_line = 16

    bg = WINE
    letterbox = WINE_DARK
    text = PALE_GOLD
    title = GOLD
    dim = DARK_GOLD
    on = GOLD
    off = WINE_DARK
    button = GOLD
    field = PARCHMENT
    field_border = GOLD
    focus = PALE_GOLD
    error = CRIMSON
    accent = GOLD
    selected = GOLD
    mask_color = GOLD
    track_colors = [GOLD, PALE_GOLD, (80, 200, 255), (120, 220, 240)]

    font = cv2.FONT_HERSHEY_COMPLEX
    scale = 0.42
    title_scale = 0.62
    small_scale = 0.36
    thickness = 1
    preview_font = cv2.FONT_HERSHEY_TRIPLEX
    preview_scale = 0.6

    # ---- panel -----------------------------------------------------------
    def panel_background(self, width, height):
        img = vertical_gradient(width, height, WINE_LIGHT, WINE_DARK)
        cv2.rectangle(img, (3, 3), (width - 4, height - 4), GOLD, 2)
        cv2.rectangle(img, (8, 8), (width - 9, height - 9), DARK_GOLD, 1)
        for (x, y, fx, fy) in ((11, 11, False, False), (width - 12, 11, True, False),
                               (11, height - 12, False, True), (width - 12, height - 12, True, True)):
            ornament(img, x, y, 7, GOLD, fx, fy)
        return img

    def draw_section(self, img, w):
        cv2.putText(img, w.text, (w.x, w.y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, self.title_scale,
                    GOLD, 1, cv2.LINE_AA)
        tw = cv2.getTextSize(w.text, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, self.title_scale, 1)[0][0]
        flourish_line(img, w.x + tw + 10, self.panel_width - self.margin - 2, w.y - 5, DARK_GOLD)

    def draw_button(self, img, w, pressed, text=None):
        text = w.label if text is None else text
        while len(text) > 3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 0.42, 1)[0][0] > w.w - 10:
            text = text[:-3] + ".."
        plaque(img, w.x, w.y, w.x + w.w, w.y + w.h, PALE_GOLD if pressed else GOLD, DARK_GOLD)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 0.42, 1)[0][0]
        cv2.putText(img, text, (int(w.x + (w.w - tw) / 2), int(w.y + w.h * 0.68)), cv2.FONT_HERSHEY_TRIPLEX,
                    0.42, INK, 1, cv2.LINE_AA)

    def draw_toggle(self, img, w, on, enabled=True):
        lit = on and enabled
        plaque(img, w.x, w.y, w.x + w.w, w.y + w.h, GOLD if lit else WINE_DARK, DARK_GOLD if lit else BRONZE,
               double=lit)
        cx, cy = w.x + 13, w.y + w.h // 2
        d = np.array([(cx, cy - 5), (cx + 5, cy), (cx, cy + 5), (cx - 5, cy)])
        if lit:
            cv2.fillPoly(img, [d], INK, cv2.LINE_AA)
        else:
            cv2.polylines(img, [d], True, DARK_GOLD if enabled else BRONZE, 1, cv2.LINE_AA)
        label = w.label if enabled else w.label + "  (n/a)"
        self.text_(img, self.fit_text(label, w.w - 28), w.x + 24, w.y + w.h * 0.68,
                   INK if lit else (DARK_GOLD if enabled else BRONZE))

    def draw_radio(self, img, w, selected):
        cx, cy = w.x + 10, w.y + w.h // 2
        cv2.circle(img, (cx, cy), 7, GOLD if selected else DARK_GOLD, 1, cv2.LINE_AA)
        if selected:
            cv2.circle(img, (cx, cy), 4, GOLD, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 9, DARK_GOLD, 1, cv2.LINE_AA)
        self.text_(img, self.fit_text(w.label, w.w - 22), w.x + 21, w.y + w.h * 0.68,
                   PALE_GOLD if selected else DARK_GOLD)

    def draw_field(self, img, w, shown, focused, cursor, error):
        cv2.putText(img, w.label, (w.x, w.y - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, GOLD, 1, cv2.LINE_AA)
        rounded_rect(img, w.x, w.y, w.x + w.w, w.y + w.h, 4, PARCHMENT)
        rounded_rect(img, w.x, w.y, w.x + w.w, w.y + w.h, 4, PALE_GOLD if focused else DARK_GOLD, 1)
        self.text_(img, shown, w.x + 8, w.y + w.h * 0.68, INK)
        if cursor:
            cx = w.x + 10 + self.text_width(shown)
            cv2.line(img, (cx, w.y + 5), (cx, w.y + w.h - 5), INK, 1)
        if error:
            self.text_(img, error, w.x, w.y + w.h + 13, CRIMSON, 0.36)

    def draw_list(self, img, w, items, selected, offset, visible, more_above, more_below):
        rh = w.row_h
        rounded_rect(img, w.x, w.y, w.x + w.w, w.y + w.h, 4, PARCHMENT)
        rounded_rect(img, w.x, w.y, w.x + w.w, w.y + w.h, 4, DARK_GOLD, 1)
        scroll = w.SCROLL_W if (more_above or more_below) else 0
        move_w = w.MOVE_W if w.can_move else 0
        gear_w = w.GEAR_W if w.has_gear else 0
        text_w = w.w - 12 - scroll - move_w - gear_w
        if not items:
            self.text_(img, w.empty_text, w.x + 8, w.y + rh * 0.75, BRONZE, self.small_scale)
        for i, item in enumerate(items[offset:offset + visible]):
            y = w.y + i * rh
            row = offset + i
            is_sel = row in selected
            if is_sel:
                rounded_rect(img, w.x + 2, y + 1, w.x + w.w - 2 - scroll, y + rh - 1, 3, GOLD)
            self.text_(img, self.fit_text(item, text_w, self.small_scale), w.x + 8, y + rh * 0.75,
                       INK if is_sel else BRONZE, self.small_scale)
            if w.has_gear:
                self.draw_gear(img, w.x + w.w - scroll - gear_w // 2, y + rh // 2, min(rh // 2 - 2, 6),
                               w.gear_state(row), is_sel)
            if is_sel and w.can_move:
                x0 = w.x + w.w - scroll - gear_w - move_w
                cy = y + rh // 2
                cv2.fillPoly(img, [np.array([(x0 + 4, cy), (x0 + 10, cy - 4), (x0 + 10, cy + 4)])], INK, cv2.LINE_AA)
                cv2.fillPoly(img, [np.array([(x0 + move_w - 4, cy), (x0 + move_w - 10, cy - 4),
                                             (x0 + move_w - 10, cy + 4)])], INK, cv2.LINE_AA)
        ax = w.x + w.w - 8
        if more_above:
            cv2.fillPoly(img, [np.array([(ax - 4, w.y + 8), (ax + 4, w.y + 8), (ax, w.y + 3)])], BRONZE)
        if more_below:
            yb = w.y + w.h
            cv2.fillPoly(img, [np.array([(ax - 4, yb - 8), (ax + 4, yb - 8), (ax, yb - 3)])], BRONZE)

    def draw_stepper(self, img, w, value):
        bw = w.button_w
        for x0, sign in ((w.x, "-"), (w.x + w.w - bw, "+")):
            c = (x0 + bw // 2, w.y + w.h // 2)
            cv2.circle(img, c, w.h // 2 - 1, GOLD, -1, cv2.LINE_AA)
            cv2.circle(img, c, w.h // 2 - 1, DARK_GOLD, 1, cv2.LINE_AA)
            cv2.putText(img, sign, (c[0] - 5, c[1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, INK, 1, cv2.LINE_AA)
        self.text_(img, w.label, w.x + bw + 8, w.y + w.h * 0.68, DARK_GOLD, self.small_scale)
        self.text_(img, value, w.x + w.w - bw - 8 - self.text_width(value), w.y + w.h * 0.68, PALE_GOLD)

    def draw_note(self, img, w, text, is_error):
        self.text_(img, self.fit_text(text, w.w, 0.36), w.x, w.y, CRIMSON if is_error else DARK_GOLD, 0.36)

    def draw_gear(self, img, cx, cy, r, state, on_row_selected):
        color = CRIMSON if state == "editing" else (INK if on_row_selected else GOLD)
        if state == "mapped" and not on_row_selected:
            color = PALE_GOLD
        cv2.circle(img, (cx, cy), max(2, r - 3), color, 2, cv2.LINE_AA)
        for k in range(8):
            a = k * np.pi / 4
            cv2.line(img, (int(round(cx + (r - 2) * np.cos(a))), int(round(cy + (r - 2) * np.sin(a)))),
                     (int(round(cx + (r + 1) * np.cos(a))), int(round(cy + (r + 1) * np.sin(a)))), color, 2, cv2.LINE_AA)

    def draw_status(self, img, lines, x, y):
        flourish_line(img, x, self.panel_width - self.margin, y - 2, DARK_GOLD)
        for i, line in enumerate(lines):
            self.text_(img, line, x, y + 14 + i * self.status_line, DARK_GOLD, self.small_scale)

    # ---- preview -----------------------------------------------------------
    def draw_preview(self, frame, poses, hands, faces, blobs, show_ids, draw_landmarks, mask=None, status=""):
        h, w = frame.shape[:2]
        frame[:] = sepia(frame)
        vignette(frame, 0.7)

        def pt(p):
            return int(p[0] * w), int(p[1] * h)

        def skeleton(track, connections, color):
            for a, b in connections:
                cv2.line(frame, pt(track.landmarks[a]), pt(track.landmarks[b]), BRONZE, 3, cv2.LINE_AA)
                cv2.line(frame, pt(track.landmarks[a]), pt(track.landmarks[b]), color, 1, cv2.LINE_AA)
            for p in track.landmarks:
                cv2.circle(frame, pt(p), 3, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt(p), 3, BRONZE, 1, cv2.LINE_AA)

        def tag(track, text, color):
            x, y = pt(track.center)
            cv2.putText(frame, text, (x + 9, y - 9), self.preview_font, self.preview_scale, WINE_DARK, 4, cv2.LINE_AA)
            cv2.putText(frame, text, (x + 9, y - 9), self.preview_font, self.preview_scale, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 8, BRONZE, 1, cv2.LINE_AA)
            vx, vy = track.velocity[:2]
            if abs(vx) + abs(vy) > 1e-3:
                cv2.arrowedLine(frame, (x, y), (int(x + vx * w * 0.25), int(y + vy * h * 0.25)), color, 2,
                                cv2.LINE_AA)

        if blobs and draw_landmarks and mask is not None:
            preview.mask_inset(frame, mask, GOLD, 0.2)
        for t in blobs:
            color = self.track_color(t.id)
            if draw_landmarks:
                preview.draw_blob(frame, t, color, 2, fill_alpha=0.18)
            if show_ids:
                tag(t, f"B{t.id}", color)
        for t in poses:
            color = self.track_color(t.id)
            if draw_landmarks:
                skeleton(t, preview.POSE_CONNECTIONS, color)
            if show_ids:
                tag(t, f"P{t.id}", color)
        for t in hands:
            color = self.track_color(t.id)
            if draw_landmarks:
                skeleton(t, preview.HAND_CONNECTIONS, color)
            if show_ids:
                tag(t, f"H{t.id}", color)
        for t in faces:
            color = self.track_color(t.id)
            if draw_landmarks:
                x0, y0, x1, y1 = t.detection.bbox()
                rounded_rect(frame, int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h), 8, color, 1)
            if show_ids:
                tag(t, f"F{t.id}", color)
        gilded_frame(frame)
        if status:
            self.draw_banner(frame, status)
        return frame

    def draw_banner(self, frame, text):
        cv2.putText(frame, text, (22, 34), self.preview_font, 0.55, WINE_DARK, 4, cv2.LINE_AA)
        cv2.putText(frame, text, (22, 34), self.preview_font, 0.55, PALE_GOLD, 1, cv2.LINE_AA)
