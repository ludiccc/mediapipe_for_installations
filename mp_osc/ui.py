"""Control panel drawn inside the OpenCV preview window.

OpenCV has no widgets, so this module keeps its own: toggle buttons, radio
buttons (one-of-many), push buttons, text fields, a list and +/- steppers,
laid out on a panel next to the camera image. Clicks arrive through
``cv2.setMouseCallback`` and key presses through the ``cv2.waitKey`` value
that the main loop already polls - no extra toolkit, no second thread, and
the preview keeps running while you edit settings.

The widgets hold state and geometry only; *how* they look is decided by the
active :class:`mp_osc.themes.Theme` (see the ``templates/`` folder).

Usage from the main loop::

    panel = ControlPanel(app)             # app exposes the settings (see app.py)
    cv2.namedWindow(name); cv2.setMouseCallback(name, panel.on_mouse)
    ...
    cv2.imshow(name, panel.render(frame, status_lines))
    if panel.on_key(cv2.waitKey(1)) == "quit": ...
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from .themes import Theme

PANEL_WIDTH = Theme.panel_width      # of the default theme (kept for older imports / tests)
MARGIN = Theme.margin
ROW = Theme.row

KEY_ENTER = (10, 13)
KEY_BACKSPACE = (8, 127)
KEY_ESC = 27


def wheel_delta(flags: int) -> int:
    """The signed wheel movement packed in the high 16 bits of a mouse event's flags."""
    delta = (int(flags) >> 16) & 0xFFFF
    return delta - 0x10000 if delta >= 0x8000 else delta


class Widget:
    label: str = ""

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)

    def hit(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, img: np.ndarray, theme: Theme) -> None:
        raise NotImplementedError

    def click(self, px: int = 0, py: int = 0) -> None:
        pass

    def scroll(self, direction: int) -> bool:
        """Mouse wheel over the widget; True if consumed."""
        return False


class Button(Widget):
    """Push button; ``on_click`` is called on every click."""

    def __init__(self, x, y, w, h, label: str, on_click: Callable[[], None]):
        super().__init__(x, y, w, h)
        self.label = label
        self.on_click = on_click
        self._pressed_at = 0.0

    def draw(self, img, theme):
        theme.draw_button(img, self, time.monotonic() - self._pressed_at < 0.15)

    def click(self, px=0, py=0):
        self._pressed_at = time.monotonic()
        self.on_click()


class DynamicButton(Button):
    """Push button whose text comes from a function (``label`` stays fixed for ``find``)."""

    def __init__(self, x, y, w, h, label: str, get_text: Callable[[], str], on_click: Callable[[], None]):
        super().__init__(x, y, w, h, label, on_click)
        self.get_text = get_text

    def draw(self, img, theme):
        theme.draw_button(img, self, time.monotonic() - self._pressed_at < 0.15, self.get_text())


class Toggle(Widget):
    """On/off button bound to a getter/setter pair. Any number can be on.

    ``enabled`` (optional) says whether the toggle makes sense right now
    (e.g. *Z from Kinect* without a Kinect): it is drawn dimmed and clicks
    are ignored while it returns False.
    """

    def __init__(self, x, y, w, h, label: str, get: Callable[[], bool], set_: Callable[[bool], None],
                 enabled: Callable[[], bool] | None = None):
        super().__init__(x, y, w, h)
        self.label = label
        self.get = get
        self.set = set_
        self.enabled = enabled or (lambda: True)

    def draw(self, img, theme):
        theme.draw_toggle(img, self, bool(self.get()), bool(self.enabled()))

    def click(self, px=0, py=0):
        if self.enabled():
            self.set(not self.get())


class RadioGroup:
    """Shared state of a set of :class:`RadioButton`: exactly one value is selected."""

    def __init__(self, get: Callable[[], Any], set_: Callable[[Any], None]):
        self.get = get
        self.set = set_
        self.buttons: list[RadioButton] = []

    @property
    def value(self):
        return self.get()

    def select(self, value) -> None:
        self.set(value)


class RadioButton(Widget):
    """One option of a RadioGroup; clicking it selects its value (and deselects the others)."""

    def __init__(self, x, y, w, h, label: str, value, group: RadioGroup):
        super().__init__(x, y, w, h)
        self.label = label
        self.value = value
        self.group = group
        group.buttons.append(self)

    @property
    def selected(self) -> bool:
        return self.group.get() == self.value

    def draw(self, img, theme):
        theme.draw_radio(img, self, self.selected)

    def click(self, px=0, py=0):
        self.group.select(self.value)


class TextField(Widget):
    """Single-line text field. Click to focus, type, Enter to submit, Esc to cancel."""

    def __init__(self, x, y, w, h, label: str, get: Callable[[], str], submit: Callable[[str], str | None],
                 max_len: int = 40):
        super().__init__(x, y, w, h)
        self.label = label
        self.get = get
        self.submit = submit        # returns an error message or None
        self.max_len = max_len
        self.focused = False
        self.text = ""
        self.error = ""

    def draw(self, img, theme):
        shown = self.text if self.focused else self.get()
        cursor = self.focused and int(time.monotonic() * 2) % 2 == 0
        theme.draw_field(img, self, shown, self.focused, cursor, self.error)

    def focus(self):
        self.focused = True
        self.text = self.get()
        self.error = ""

    def blur(self):
        self.focused = False

    def click(self, px=0, py=0):
        self.focus()

    def key(self, code: int) -> bool:
        """Handle a key while focused. Returns True if consumed."""
        if code in KEY_ENTER:
            self.error = self.submit(self.text.strip()) or ""
            if not self.error:
                self.blur()
            return True
        if code == KEY_ESC:
            self.blur()
            return True
        if code in KEY_BACKSPACE:
            self.text = self.text[:-1]
            return True
        if 32 <= code <= 126 and len(self.text) < self.max_len:
            self.text += chr(code)
            return True
        return True  # swallow everything else while editing


class ListBox(Widget):
    """Scrollable list of strings with one or several checked rows.

    * clicking a row calls ``on_select(index)`` (the app decides what that
      means: select it, or add/remove it from a multi-selection);
    * with ``on_move`` set, every checked row gets ``<`` ``>`` buttons at its
      right end that call ``on_move(index, -1 / +1)`` (the app uses them to
      order the cameras left to right);
    * the mouse wheel scrolls; when the list is longer than the box, the
      narrow strip at the far right scrolls too (top half up, bottom half down).

    ``get_selected`` returns the checked row indices (an int is accepted for
    a single selection, -1 = none).
    """

    MOVE_W = 34          # width of the "< >" area of a checked row
    GEAR_W = 20          # width of the gear (crop & mapping) button of a row
    SCROLL_W = 12        # width of the scroll strip

    def __init__(self, x, y, w, row_h: int, visible: int, label: str,
                 get_items: Callable[[], Sequence[str]], get_selected: Callable[[], Any],
                 on_select: Callable[[int], None], empty_text: str = "(empty)",
                 on_move: Callable[[int, int], None] | None = None,
                 on_gear: Callable[[int], None] | None = None,
                 gear_state: Callable[[int], str] | None = None):
        super().__init__(x, y, w, row_h * visible)
        self.label = label
        self.row_h = int(row_h)
        self.visible = int(visible)
        self.get_items = get_items
        self._get_selected = get_selected
        self.on_select = on_select
        self.on_move = on_move
        self.on_gear = on_gear                  # gear clicked on row i
        self.gear_state = gear_state or (lambda i: "")   # "" / "mapped" / "editing" per row
        self.empty_text = empty_text
        self.offset = 0

    def get_selected(self) -> list[int]:
        sel = self._get_selected()
        if isinstance(sel, int):
            return [sel] if sel >= 0 else []
        return [int(i) for i in sel]

    @property
    def can_move(self) -> bool:
        return self.on_move is not None

    @property
    def has_gear(self) -> bool:
        return self.on_gear is not None

    def right_edge(self, n_items: int) -> int:
        """x where the row text area ends (before the scroll strip)."""
        return self.x + self.w - (self.SCROLL_W if self._overflow(n_items) else 0)

    def _clamp(self, n: int) -> None:
        self.offset = max(0, min(self.offset, max(0, n - self.visible)))

    def _overflow(self, n: int) -> bool:
        return n > self.visible

    def draw(self, img, theme):
        items = list(self.get_items())
        self._clamp(len(items))
        theme.draw_list(img, self, items, self.get_selected(), self.offset, self.visible,
                        self.offset > 0, self.offset + self.visible < len(items))

    def click(self, px=0, py=0):
        items = list(self.get_items())
        n = len(items)
        self._clamp(n)
        if self._overflow(n) and px >= self.x + self.w - self.SCROLL_W:
            self.scroll(-1 if py < self.y + self.h // 2 else 1)
            return
        row = (py - self.y) // self.row_h
        index = self.offset + row
        if not (0 <= row < self.visible and 0 <= index < n):
            return
        right = self.right_edge(n)
        if self.has_gear and px >= right - self.GEAR_W:
            self.on_gear(index)
            return
        if self.has_gear:
            right -= self.GEAR_W
        if self.can_move and index in self.get_selected() and px >= right - self.MOVE_W:
            self.on_move(index, -1 if px < right - self.MOVE_W // 2 else +1)
            return
        self.on_select(index)

    def scroll(self, direction: int) -> bool:
        n = len(self.get_items())
        if n <= self.visible:
            return False
        before = self.offset
        self.offset += direction
        self._clamp(n)
        return self.offset != before

    def select_label(self, label: str) -> None:
        """Click the row whose text equals / ends with / contains ``label`` (tests, scripting)."""
        items = list(self.get_items())
        for i, item in enumerate(items):
            if item == label or item.strip() == label or item.endswith(label):
                self.on_select(i)
                return
        for i, item in enumerate(items):
            if label in item:
                self.on_select(i)
                return
        raise KeyError(label)


class Stepper(Widget):
    """``[-] label value [+]``: clicking the ends calls ``on_step(-1)`` / ``on_step(+1)``."""

    def __init__(self, x, y, w, h, label: str, get_value: Callable[[], str], on_step: Callable[[int], None],
                 button_w: int = 26):
        super().__init__(x, y, w, h)
        self.label = label
        self.get_value = get_value
        self.on_step = on_step
        self.button_w = int(button_w)

    def draw(self, img, theme):
        theme.draw_stepper(img, self, str(self.get_value()))

    def click(self, px=0, py=0):
        if px < self.x + self.button_w:
            self.on_step(-1)
        elif px >= self.x + self.w - self.button_w:
            self.on_step(+1)

    def scroll(self, direction: int) -> bool:
        self.on_step(-direction)
        return True


class Label(Widget):
    """Static text; ``section=True`` for a section title."""

    def __init__(self, x, y, text: str, color=None, scale=None, section: bool = False):
        super().__init__(x, y, 0, 0)
        self.text, self.color, self.scale, self.section = text, color, scale, section

    def draw(self, img, theme):
        if self.section:
            theme.draw_section(img, self)
        else:
            theme.draw_label(img, self)

    def hit(self, px, py):
        return False


class Note(Widget):
    """One line of dynamic text (``get`` returns (text, is_error)); nothing when empty."""

    def __init__(self, x, y, w, get: Callable[[], tuple[str, bool]]):
        super().__init__(x, y, w, 0)
        self.get = get

    def draw(self, img, theme):
        text, is_error = self.get()
        if text:
            theme.draw_note(img, self, text, is_error)

    def hit(self, px, py):
        return False


class ControlPanel:
    """The settings panel. ``app`` must provide the attributes used in ``_build``."""

    def __init__(self, app, theme: Theme | None = None):
        self.app = app
        self.theme = theme or getattr(app, "theme", None) or Theme()
        self.widgets: list[Widget] = []
        self.fields: list[TextField] = []
        self._frame_width = 0
        self.origin = (0, 0)          # panel position inside the window (mouse offset)
        self.height = 0
        self.dirty = True             # something changed since the last render
        self._build()

    @property
    def width(self) -> int:
        return self.theme.panel_width

    @property
    def frame_width(self) -> int:
        """Width of the camera image last rendered (the panel sits next to it)."""
        return self._frame_width

    @frame_width.setter
    def frame_width(self, value: int) -> None:
        self._frame_width = int(value)
        self.origin = self.theme.panel_origin(self._frame_width, 0)

    def set_theme(self, theme: Theme) -> None:
        """Switch the look; the layout is rebuilt with the theme's geometry."""
        self.theme = theme
        self.widgets, self.fields = [], []
        self._build()
        self.dirty = True

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        app = self.app
        th = self.theme
        W, M, ROWH = th.panel_width, th.margin, th.row
        inner = W - 2 * M
        half = (inner - 6) // 2
        third = (inner - 12) // 3
        y = M + 8

        def section(title):
            nonlocal y
            self.widgets.append(Label(M, y + 8, title, section=True))
            y += th.section_gap

        def grid(items, cols=2):
            """items: list of Widget factories taking (x, y, w, h); ``cols`` per row."""
            nonlocal y
            w = half if cols == 2 else third
            for i, make in enumerate(items):
                col = i % cols
                self.widgets.append(make(M + col * (w + 6), y, w, ROWH - 4))
                if col == cols - 1 or i == len(items) - 1:
                    y += ROWH
            y += 4

        def toggle(label, get, set_):
            return lambda x, yy, w, h: Toggle(x, yy, w, h, label, get, set_)

        def button(label, fn):
            return lambda x, yy, w, h: Button(x, yy, w, h, label, fn)

        def radio(label, value, group):
            return lambda x, yy, w, h: RadioButton(x, yy, w, h, label, value, group)

        def field(label, get, submit, max_len=40, x=None, w=None, advance=True):
            """A text field; ``x``/``w`` place several on one row, ``advance`` moves to the next row."""
            nonlocal y
            f = TextField(M if x is None else x, y + 12, inner if w is None else w, ROWH - 4,
                          label, get, submit, max_len)
            self.widgets.append(f)
            self.fields.append(f)
            if advance:
                y += 12 + ROWH + 4

        def note(get):
            nonlocal y
            self.widgets.append(Note(M, y + 10, inner, get))
            y += 16

        # ---- detectors ----------------------------------------------------
        section("Detectors")
        grid([
            toggle("Body", lambda: app.cfg.pose, lambda v: app.set_detectors(pose=v)),
            toggle("Hands", lambda: app.cfg.hands, lambda v: app.set_detectors(hands=v)),
            toggle("Gestures", lambda: app.cfg.gesture, lambda v: app.set_detectors(gesture=v)),
            toggle("Face", lambda: app.cfg.face, lambda v: app.set_detectors(face=v)),
            toggle("Blobs", lambda: app.cfg.blobs, lambda v: app.set_detectors(blobs=v)),
            button("Learn bg  (b)", app.learn_background),
        ])
        self.widgets.append(Stepper(M, y, inner, ROWH - 4, "Blob sensitivity",
                                    lambda: str(app.cfg.blob_threshold), app.step_blob_threshold))
        y += ROWH
        self.widgets.append(Stepper(M, y, inner, ROWH - 4, "Blob min area %",
                                    lambda: f"{app.cfg.blob_min_area * 100:.2f}", app.step_blob_min_area))
        y += ROWH + 4

        # ---- source (one of) ----------------------------------------------
        section("Source")
        # Cameras = whatever is checked in the list (webcams and Kinects alike).
        self.source_group = RadioGroup(lambda: app.mode, app.set_mode)
        options = [radio("Cameras", "camera", self.source_group),
                   radio("Pause", "pause", self.source_group)]
        if app.file_source:
            options.append(radio("File", "file", self.source_group))
        grid(options, cols=2)
        # The list: a click checks/unchecks a camera (checked ones are shown
        # side by side, numbered left to right); < > on a checked row moves it.
        # The gear at the right of a row opens the crop & mapping editor of that camera.
        self.camera_list = ListBox(M, y, inner, th.list_row, 4, "Cameras",
                                   app.camera_labels, app.selected_camera_rows, app.toggle_camera_row,
                                   empty_text="(no cameras found)", on_move=app.move_camera_row,
                                   on_gear=app.toggle_mapping_editor_row, gear_state=app.gear_state_row)
        self.widgets.append(self.camera_list)
        y += self.camera_list.h + 4
        note(app.source_note)
        self.widgets.append(Button(M, y, inner, ROWH - 4, "Rescan cameras", app.scan_cameras))
        y += ROWH + 4

        # ---- image --------------------------------------------------------
        section("Image")
        grid([
            toggle("Mirror", lambda: app.mirror, lambda v: setattr(app, "mirror", v)),
            toggle("Draw", lambda: app.draw_landmarks, lambda v: setattr(app, "draw_landmarks", v)),
        ])
        # z of every landmark = distance measured by the Kinect (metres); only
        # meaningful when the running source has depth (a Kinect, or Kinects only).
        self.widgets.append(Toggle(M, y, inner, ROWH - 4, "Z from Kinect  (z)",
                                   lambda: app.cfg.z_from_kinect, app.set_z_from_kinect,
                                   enabled=lambda: app.depth_available))
        y += ROWH + 4

        # ---- OSC ----------------------------------------------------------
        section("OSC output")
        ip_w = int(inner * 0.64)
        field("IP / host", lambda: app.sender.ip, lambda v: app.set_osc(ip=v), x=M, w=ip_w, advance=False)
        field("Port", lambda: str(app.sender.port), lambda v: app.set_osc(port=v), 5,
              x=M + ip_w + 6, w=inner - ip_w - 6)
        grid([
            toggle("mp", lambda: "mp" in app.cfg.formats, lambda v: app.set_format("mp", v)),
            toggle("legacy", lambda: "legacy" in app.cfg.formats, lambda v: app.set_format("legacy", v)),
            toggle("kinect", lambda: "kinect" in app.cfg.formats, lambda v: app.set_format("kinect", v)),
            toggle("Landmarks", lambda: app.cfg.send_landmarks, lambda v: setattr(app.cfg, "send_landmarks", v)),
        ])

        # ---- look / configuration files ------------------------------------
        grid([
            lambda x, yy, w, h: DynamicButton(x, yy, w, h, "Look  (t)", lambda: f"Look: {app.theme.name}",
                                              app.next_template),
            button("Quit  (q)", app.request_quit),
            button("Save config", lambda: app.save_config()),
            button("Load config...", lambda: app.load_config()),
        ])
        note(app.config_status)

        self.status_y = y + 4
        y += 4 + 5 * th.status_line + 6
        self.height = y

    # ------------------------------------------------------------------ events
    def find(self, label: str) -> Widget:
        """Widget by label (Toggle/Button/Radio/TextField/...), e.g. for tests or scripting."""
        for w in self.widgets:
            if getattr(w, "label", None) == label:
                return w
        raise KeyError(label)

    def click(self, label: str, where: str = "center") -> None:
        """Simulate a click on the widget with that label (``where``: center/left/right)."""
        w = self.find(label)
        px = {"left": w.x + 4, "right": w.x + w.w - 4}.get(where, w.x + w.w // 2)
        self.on_mouse(cv2.EVENT_LBUTTONDOWN, self.origin[0] + px, self.origin[1] + w.y + w.h // 2)

    def type_text(self, text: str) -> None:
        """Feed characters to the panel as key presses (for tests / scripting)."""
        for ch in text:
            self.on_key(ord(ch))

    @property
    def focused_field(self) -> TextField | None:
        return next((f for f in self.fields if f.focused), None)

    def on_mouse(self, event, x, y, flags=None, param=None) -> None:
        px, py = x - self.origin[0], y - self.origin[1]
        on_image = px < 0 or px >= self.width
        editor = getattr(self.app, "editor", None)
        if editor is not None and on_image and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE,
                                                         cv2.EVENT_LBUTTONUP):
            # the crop & mapping editor drags its handles on the camera image
            ix = x - (self.width if self.theme.panel_side == "left" else 0)
            self.app.editor_mouse(event, ix, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                for f in self.fields:
                    f.blur()
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            direction = -1 if wheel_delta(flags or 0) > 0 else 1     # wheel up = towards the top
            for w in self.widgets:
                if w.hit(px, py) and w.scroll(direction):
                    self.dirty = True
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.dirty = True
        if px < 0 or px >= self.width:
            # click on the camera image: leave any field
            for f in self.fields:
                f.blur()
            return
        for w in self.widgets:
            if w.hit(px, py):
                if not isinstance(w, TextField):
                    for f in self.fields:
                        f.blur()
                w.click(px, py)
                return
        for f in self.fields:
            f.blur()

    def on_key(self, code: int) -> str | None:
        """Returns "quit", "handled" or None (key not consumed by the panel)."""
        if code is None or code < 0:
            return None
        code &= 0xFF
        if code == 255:
            return None
        f = self.focused_field
        if f is not None:
            f.key(code)
            self.dirty = True
            return "handled"
        if code in (27, ord("q")):
            return "quit"
        return None

    # ------------------------------------------------------------------ drawing
    def render(self, frame: np.ndarray, status: list[str]) -> np.ndarray:
        th = self.theme
        fh, fw = frame.shape[:2]
        h = max(fh, self.height)
        panel = th.panel_background(th.panel_width, h)
        for w in self.widgets:
            w.draw(panel, th)
        th.draw_status(panel, status, th.margin, self.status_y)
        self._frame_width = fw
        self.origin = th.panel_origin(fw, fh)
        return th.compose(frame, panel)
