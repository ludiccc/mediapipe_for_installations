# Templates — the look & feel of the `mediapipe_osc.py` window

A template decides how the window *looks*: the colours and fonts of the
control panel, how each widget is drawn, and how the camera image and the
landmarks are rendered. It never changes what the app *does* (OSC, detectors,
sources).

```
python mediapipe_osc.py --template minimal        # a name from this folder
python mediapipe_osc.py --template ~/my_look.py   # or any file
```

While the app runs, the **Look** button (or `t`) cycles through `classic` and
every `*.py` file in this folder, and `r` reloads the current file — edit,
save, press `r`, see the result. A template that fails to load (syntax error,
exception) prints the error and leaves the current look in place.

## The examples

| file         | idea |
|--------------|------|
| `minimal.py` | White paper, thin black type, no boxes: toggles are a small square, radios a dot, buttons an underlined word, fields a hairline. The image is untouched; landmarks are thin white lines with tiny ids. Good for a quiet control screen. |
| `ascii.py`   | Everything is characters. The video is converted to ASCII art (one character per 8×14 pixel cell, chosen by brightness, green phosphor on black — `COLOR_MODE = "color"` tints each character with the camera colour); landmarks and blob outlines are typed over it; the panel uses `[x]` / `( )` / `+---+`. |
| `baroque.py` | Gold on burgundy velvet: double gold borders with corner flourishes, script section titles, gilded plaques for buttons and toggles, parchment fields. The picture gets sepia, a vignette and a gilded frame; landmarks are gold with bronze shadows. |

`classic` is the built-in default (`mp_osc/themes.py`) — the examples show
three different ways of overriding it: `minimal` mostly changes colours and
a few widget methods, `baroque` redraws every widget, `ascii` replaces the
text rendering itself and the whole preview.

## Writing your own

Copy the closest example, or start from the smallest possible template:

```python
from mp_osc.themes import Theme

class Template(Theme):
    name = "pink"
    bg = (200, 120, 255)      # colours are (B, G, R) like everywhere in OpenCV
    on = (120, 60, 220)
    text = (30, 20, 40)
```

The file must define `class Template(Theme)` (or a `THEME = ...` instance).
Everything below can be overridden; look at `mp_osc/themes.py` for the
defaults and the exact signatures.

**Geometry**: `panel_width`, `margin`, `row` (height of a row of buttons),
`list_row`, `section_gap`, `status_line`, `panel_side` (`"right"` or `"left"`).

**Colours**: `bg`, `letterbox` (around the image when the panel is taller),
`text`, `title`, `dim`, `on`, `off`, `button`, `field`, `field_border`,
`focus`, `error`, `accent`, `selected`, `mask_color`, and `track_colors` (a
list, one per id for the landmarks).

**Fonts**: `font`, `scale`, `title_scale`, `small_scale`, `thickness` for the
panel; `preview_font`, `preview_scale` for the ids on the image. Text is drawn
through `text_(img, s, x, y, color, scale)` / `text_width(s)` / `fit_text(s,
width)`, which you can override to use another renderer (`ascii.py` does).

**Panel drawing** — every method gets the panel image (numpy, BGR) and the
widget, which carries `x, y, w, h` and a `label`:

| method | draws |
|--------|-------|
| `panel_background(width, height)` | returns the empty panel image |
| `compose(frame, panel)` / `panel_origin(fw, fh)` | puts image and panel together (defaults handle `panel_side`) |
| `draw_section(img, label)` | a section title (`label.text`) |
| `draw_label(img, label)`, `draw_note(img, note, text, is_error)` | static / dynamic small text |
| `draw_button(img, button, pressed, text=None)` | push button |
| `draw_toggle(img, toggle, on, enabled=True)` | on/off button; `enabled=False` = greyed out (e.g. *Z from Kinect* without a Kinect) |
| `draw_radio(img, radio, selected)` | one option of a radio group |
| `draw_field(img, field, shown, focused, cursor, error)` | text field |
| `draw_list(img, listbox, items, selected, offset, visible, more_above, more_below)` | the camera list: `selected` is the list of checked rows; from the right: the scroll strip (`listbox.SCROLL_W`, only when the list overflows), the gear (`listbox.GEAR_W`, when `listbox.has_gear`; `listbox.gear_state(row)` is `""`, `"mapped"` or `"editing"`), then ◀ ▶ (`listbox.MOVE_W`) on checked rows when `listbox.can_move` |
| `draw_gear(img, cx, cy, r, state, on_selected_row)` | the gear icon of a row (crop & mapping editor) |
| `draw_stepper(img, stepper, value)` | `[-] label value [+]` (`stepper.button_w`) |
| `draw_status(img, lines, x, y)` | the status lines |

**Preview drawing**: `draw_preview(frame, poses, hands, faces, blobs,
show_ids, draw_landmarks, mask=None, status="")` must return the image to
show (the same array or a new one of the same size). `poses`, `hands`,
`faces`, `blobs` are lists of `Track` objects: `t.id`, `t.center` (x, y, z in
0..1), `t.landmarks` (normalized points), `t.velocity`, `t.detection` (the
raw detection: `bbox()`, and for blobs `area`, `angle`, `size`). `mask` is
the binary blob mask (when Blobs is on). `draw_banner(frame, text)` draws the
one-line messages such as *learning background*. Helpers in
`mp_osc.preview` (`draw_blob`, `label`, `mask_inset`, the MediaPipe
connection lists `POSE_CONNECTIONS` / `HAND_CONNECTIONS`) and in
`mp_osc.themes` (`rounded_rect`, `vertical_gradient`, `vignette`, `sepia`)
can be reused.

Keep `draw_preview` fast: it runs once per analyzed frame. A few
milliseconds is fine (the examples take 2–20 ms on a 640×480 image).
