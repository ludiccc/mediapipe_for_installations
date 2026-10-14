# How `mediapipe_osc.py` works — a guide for students

This document explains the application from the inside: what happens to a
camera frame between the moment it is grabbed and the moment an OSC message
leaves the computer, how the code is organised, and where to start when you
want to change something. The user manual (options, messages, panel) is
[README_mediapipe_osc.md](README_mediapipe_osc.md); this guide is about
*understanding and extending* the program.

Reading order suggestion: sections 1–3 first (the idea and the pipeline),
then open `mp_osc/app.py` next to section 4, and come back to the other
sections when you need them.

---

## 1. What the program does, in one paragraph

A camera (or several, or a Kinect, or a video file) produces images. For each
image, *detectors* find things: the 33 landmarks of every body MediaPipe
sees, the 21 of every hand, the 478 of every face, or — with plain OpenCV —
the shapes ("blobs") that stand out from the background, for a camera on the
ceiling. A *tracker* decides that the body found in this frame is the same
person as in the previous frame and gives it a stable number. Everything is
then written as *OSC messages* (`/mp/pose/center 3 0.52 0.41 -0.10`) and
sent over the network to whatever makes the installation react: Processing,
TouchDesigner, Pure Data, Max, SuperCollider, Unity, another Python
script... A window shows what the detectors see and lets you change settings
while the show runs.

## 2. The vocabulary

| word | meaning here |
|------|--------------|
| **frame** | one image from the source, a numpy array of shape `(height, width, 3)`, 8-bit, **BGR** (OpenCV's channel order; MediaPipe wants RGB, so the app converts) |
| **landmark** | one point of a MediaPipe model: `x, y` normalized to the image (0..1, origin top-left), `z` relative depth; pose landmarks also carry `visibility` and `presence` |
| **detection** | what a detector found in *one* frame — no memory, no identity |
| **track** | a detection followed over time: it has an `id`, a `center`, a `velocity`, a first/last seen time |
| **id** | an integer that stays the same while the person/hand/face/blob is in view; a new one when it comes back |
| **OSC** | *Open Sound Control*: small typed messages (`address`, then ints/floats/strings) sent as UDP packets. No connection, no confirmation, very low latency — perfect for "person 3 is at (0.5, 0.4)" 30 times per second |
| **source** | where frames come from: a `FrameSource` object (webcam, Kinect, composition, video, image) |
| **blob** | a connected shape in a binary mask — the OpenCV detector's unit; for a ceiling camera, a blob is usually a person |
| **template** | a Python file that defines how the window looks (colours, fonts, how widgets and landmarks are drawn) |
| **depth** | with a Kinect, the distance of every pixel to the sensor in metres |

## 3. The pipeline of one frame

```
  ┌─────────────┐   frame    ┌──────────┐  detections  ┌─────────┐  tracks   ┌────────────────┐  messages  ┌───────────┐
  │ FrameSource │ ─────────▶ │ Analyzer │ ───────────▶ │ Tracker │ ────────▶ │ MessageBuilder │ ─────────▶ │ OscSender │ ──▶ UDP
  │ capture.py  │            │detectors │              │tracking │           │   osc_out.py   │            │osc_out.py │
  └─────────────┘            │ blobs.py │              │   .py   │           └────────────────┘            └───────────┘
                             └──────────┘              └─────────┘
                                   │                        │
                                   ▼                        ▼
                              (Z from Kinect:          Theme.draw_preview ──▶ window + ControlPanel
                               depth.py)               themes.py, preview.py   ui.py
```

This is `App.step()` in `mp_osc/app.py`, numbered in the code:

1. **apply pending changes** — settings the panel changed since the last
   frame (see *threads* below);
2. **read a frame** from the source; mirror it if asked (and the depth map too);
3. **analyze**: the `Analyzer` runs every detector that is on and returns a
   `Detections` object — four lists (`poses`, `hands`, `faces`, `blobs`) of
   small dataclasses holding numpy arrays;
4. **Z from Kinect** (optional): every landmark's z is replaced by the
   distance the Kinect measured at that pixel; (the crop & mapping of each
   camera has already been applied to the frame, right after reading it)
5. **track**: each list goes through its own `Tracker`, which returns the
   active tracks (with ids) plus the ids that *entered* and *left* this frame;
6. **build and send** the OSC messages of this frame;
7. **draw** the preview image (only when there is a window).

Each step is a separate module with a clear input and output, so each can be
understood — and tested — on its own. Look at `tests/` for examples of every
module used in isolation.

## 4. The modules, one by one

### `mp_osc/config.py` — the options

`Config` is a *dataclass*: a plain object with one attribute per option
(`pose`, `port`, `blob_threshold`, ...) and their defaults. `build_parser()`
describes the same options for the command line (`argparse`) and
`parse_args()` turns `sys.argv` into a `Config`. Everything else in the app
reads `cfg.<option>` — the panel simply *writes* `cfg.<option>` at run time.
To add an option: add the attribute to `Config`, add the `--flag` to the
parser, read it where you need it.

### `mp_osc/capture.py` — where frames come from

All sources share the interface of `FrameSource`: `read()` → `(ok, frame)`,
`timestamp()`, `depth()`, `release()`. The factory `open_source(spec)` picks
the class from the *spec* string:

* `"0"`, `"1"` ... → `WebcamSource`. Note the background thread: OpenCV
  buffers frames, so a slow analysis would read *old* frames and lag behind
  reality. The thread keeps grabbing and keeps only the newest frame.
* `"kinect"` → `KinectSource`, with three *backends* (Kinect v2 via
  pylibfreenect2, Kinect v1 via libfreenect, OpenNI via OpenCV) behind one
  class. It can hand out the colour image or the depth map as a grey image,
  and — when `want_depth` is set — a depth map *aligned with the colour
  image* (for Kinect v2: libfreenect2's `Registration` re-projects the
  512×424 depth image onto the 1920×1080 colour image, the `bigdepth`).
* `"0+kinect"` → `CompositeSource`: several sources glued left to right into
  one wide frame. MediaPipe then simply sees one wide picture — the
  coordinates it returns are normalized to the whole composition. Depth
  exists only if *every* part has it.
* `"clip.mp4"`, `"photo.jpg"` → `VideoFileSource`, `ImageSource` — how the
  tests run without a camera.

`mp_osc/cameras.py` is the companion that *finds* cameras (webcams through
`/dev/video*` — asking the kernel which nodes really capture video — or by
probing indices, Kinects through their libraries) so the panel can list
them, webcams and Kinects in one list.

One rule worth knowing when you touch source switching (`App._apply_source`):
**a device is never opened twice**. Opening a webcam that is already open
fails, and opening a Kinect v2 that is already streaming crashes
libfreenect2. So when the selection changes, the parts that stay are reused
as they are, only the new ones are opened, and the removed ones are released
*after* the new composition is in place.

### `mp_osc/detectors.py` and `mp_osc/blobs.py` — finding things

`Analyzer` owns the MediaPipe *Tasks* (`PoseLandmarker`, `HandLandmarker`,
`GestureRecognizer`, `FaceLandmarker`), created lazily from the `.task`
model files in the repository and closed when switched off. They run in
`VIDEO` mode, which needs strictly increasing timestamps in milliseconds —
that is why `step()` passes the frame time. The results (MediaPipe's own
objects) are converted to numpy arrays right away, so nothing else in the
app depends on MediaPipe's classes: `PoseDetection.landmarks` is a `(33, 5)`
array, `HandDetection.landmarks` `(21, 3)`, etc. The properties `center` and
`bbox()` of those dataclasses define *where* a detection is for the tracker
and the messages (for a body: the visibility-weighted mean of nose,
shoulders and hips — a point that stays inside the image even when legs are
not visible).

`BlobDetector` (`blobs.py`) is the non-MediaPipe detector. It works on a
binary *mask* — which pixels are "something" — obtained in one of four ways
(difference to a learned background, adaptive model, or a plain threshold
for bright/dark objects), cleaned with morphology, and cut into connected
shapes with `cv2.findContours`. Each shape big enough becomes a
`BlobDetection` with centroid, bounding box, area, orientation (from image
moments) and a 16-point contour. It mirrors what the original Processing +
Kinect sketch of `Sonido_Cuadrafonico` did by hand.

### `mp_osc/mapping.py` — crop & mapping of a camera

The tpf-tools idea (`mapping_in_mapping_out`) as a class: `CameraMapping`
holds an *input* quad (which part of the camera picture) and an *output*
quad (where it goes in the camera's slot), both normalized, and `apply()`
turns them into one `cv2.getPerspectiveTransform` + `cv2.warpPerspective`
(the depth map gets the same warp, nearest-neighbour). `apply_mappings()`
does it for every slot of a composed frame, *before* the detectors run — so
the OSC coordinates are coordinates of the mapped picture (with a floor
mapped to the whole slot: positions on the floor). `MappingEditor` is the
state of the editor (mode, selected corner, dragging) with the drawing of
the handles; the app only converts mouse positions to normalized
coordinates and forwards keys. JSON in the tpf-tools format, so files are
interchangeable.

### `mp_osc/depth.py` — Z from Kinect

MediaPipe's `z` is *relative* (how far in front of / behind the hips or the
wrist, in image units). A Kinect gives an *absolute* distance per pixel.
`sample_depth()` looks up, for each landmark, the median of the valid depth
pixels in a 5×5 window around its (x, y) — robust against the holes of depth
images and against landmarks sitting on the silhouette's edge — and
`apply_depth()` writes it into the landmarks' z, in metres (0 = no reading).
The receivers are told which kind of z they get with `/mp/z_source`.

### `mp_osc/tracking.py` — who is who

MediaPipe returns detections in arbitrary order and does not remember
anybody. `Tracker.update(detections, now)` matches them to the tracks of the
previous frame by nearest centre (greedy: closest pair first, never farther
than `--track-distance`). A detection nobody claims becomes a new track
(`entered`); a track not seen for `--track-timeout` seconds is dropped
(`left`). Velocity is the smoothed displacement per second. It is a
deliberately simple tracker: predictable, good for a handful of people in an
installation, not for a crowd.

### `mp_osc/osc_out.py` — the messages

`MessageBuilder.frame(...)` turns the tracks into a list of
`(address, [args])` tuples — nothing is sent yet, which is what makes the
schema testable. `OscSender.send()` encodes and sends them over UDP, one
packet per message or all in one bundle. Three *formats* can be produced at
the same time: `mp` (this app's numeric schema), `legacy` (the string
messages of the original `ejemplo-*.py` scripts) and `kinect`
(`/kinect/person id x y speed` for the quad-sound scripts). The complete
list of addresses is in the module docstring and in the README.

### `mp_osc/themes.py`, `mp_osc/preview.py`, `templates/` — the look

Everything drawn goes through a `Theme`: colours, fonts, one `draw_*` method
per widget, and `draw_preview()` for the camera image with the landmarks.
The base class *is* the default look; a template is a file with
`class Template(Theme)` overriding what it wants. `preview.py` holds the
default drawing helpers (MediaPipe's drawing utilities for landmarks, blob
outlines, the mask inset). Read `templates/README.md` before writing one.

### `mp_osc/ui.py` — the control panel

OpenCV windows have no buttons, so the panel draws its own: `Toggle`,
`RadioButton` (+`RadioGroup`), `Button`, `TextField`, `ListBox`, `Stepper`,
`Label`, `Note`. A widget knows its rectangle and what to call when clicked;
*how* it looks is the theme's job. `ControlPanel._build()` is the layout —
a list of widgets with y advancing row by row; `on_mouse()` finds the widget
under the click; `on_key()` feeds keys to the focused text field;
`render()` draws the panel next to the frame. The panel never touches
detectors or sockets: it calls `app.set_detectors(...)`, `app.set_osc(...)`,
`app.set_mode(...)`, and so on.

### `mp_osc/app.py` — putting it together

`App.__init__` builds the objects above; `App.step()` is the pipeline;
`App.run()` starts the threads; the `set_*` methods are what the panel (and
the keyboard) call. Read its module docstring: it is the map of the program.

## 5. Threads — why there are two, and the one rule

With a window, the app runs two threads:

* the **analysis thread** loops over `step()`;
* the **main thread** loops over `_window_loop()`: show the latest drawn
  frame, poll the keyboard and mouse with `cv2.waitKey(5)`.

If both were the same thread, a slow model (say 150 ms per frame) would make
the window react to clicks 6 times per second and lose key presses. Two
threads keep the window smooth however slow the models are.

The rule: **MediaPipe objects and the frame source are touched only by the
analysis thread.** When the panel wants to switch a detector or a camera, it
does not do it itself — it queues a small function with
`App._on_analysis_thread(fn)`, and `step()` runs the queue between two
frames (`_apply_pending()`). Simple attribute writes (`app.mirror = True`,
`sender.paused = True`) need no queue: reading a stale value for one frame is
harmless. Without a window (`--no-preview`) there is only one thread and the
queue is executed immediately, which is also what the tests use.

A third, short-lived thread scans the cameras in the background at start-up
(opening every webcam for a moment), so the window appears right away.

## 6. Coordinates, time and units

* `x, y` are normalized to the analyzed frame: 0..1, origin at the top-left.
  With `--mirror` the frame is flipped *before* analysis, so x grows to the
  viewer's right. With several cameras side by side, the frame is the whole
  composition: a person in the right-hand camera of two equal cameras has
  x > 0.5. Landmarks MediaPipe extrapolates outside the picture can be < 0 or
  > 1.
* `z` is MediaPipe's relative depth (`/mp/z_source mediapipe`) or, with
  *Z from Kinect*, metres from the sensor (`/mp/z_source kinect`; 0 = no
  reading at that pixel).
* Time: live sources use the monotonic clock; a video file uses its own frame
  rate (so the same file gives the same velocities whether the computer is
  fast or slow). `App` keeps time increasing across source switches.
* Velocities are in normalized units per second; blob areas are fractions of
  the image; angles in degrees, clockwise on screen.

## 7. How to ...

**... see what is sent.** `python osc_monitor.py --port 5005` in a second
terminal prints every message; `--rate` counts per address; `--filter
/mp/blob` narrows it down. `--verbose` on the app prints them too.

**... add an OSC message.** In `osc_out.py`, `MessageBuilder._mp_frame()`
builds the per-track messages: add a line such as
`out.append(self._msg("pose", "height", t.id, [float(box[3] - box[1])]))`,
document it in the module docstring and in the README, and add a test in
`tests/test_osc_out.py` (the `build()` helper there runs fake detections
through trackers and the builder without any model).

**... add a setting to the panel.** Add the attribute to `Config` (and the
`--flag`), a `set_<thing>()` method in `App` if applying it needs the
analysis thread, and one line in `ControlPanel._build()`:
`toggle("Label", lambda: app.cfg.thing, lambda v: app.set_thing(v))`.
Tests drive the panel with `panel.click("Label")`.

**... add a detector.** Produce a dataclass with `landmarks` (an `(N, ≥2)`
array of normalized points) and a `center` property, run it in
`Analyzer.analyze()` into a new list of `Detections`, give it a `Tracker` in
`App`, messages in `MessageBuilder`, and a drawing in `preview.draw()`.
`BlobDetector` is the smallest complete example of all four steps.

**... save a show's setup.** *Save config* writes a JSON with everything the
panel can change (`App.to_config()`); *Load config…* (an OS file dialog,
`mp_osc/dialogs.py`) reads it back and `App.apply_config()` applies it live
— sources included. `--config file.json` does it at start-up. If you add a
setting to the panel, add it to `to_config` / `apply_config` too.

**... change how the window looks.** Copy `templates/minimal.py`, change
colours or override a `draw_*` method, run with `--template yourfile.py`,
press `r` to reload after each edit.

**... run without a camera.** `--source photo.jpg` or `--source clip.mp4`.
The tests do exactly that with the pictures in the repository.

**... use two Kinects on the ceiling.** `--source kinect+kinect:1 --blobs
--format kinect --port 12000`: the two depth images side by side, blobs on
the composition, `/kinect/person` for the quad-sound scripts. Add
`--z-from-kinect` and the blobs' z is their distance to the sensors.

## 8. Testing

`python -m pytest tests/ -q` runs about 150 tests in a minute, no camera
needed: unit tests per module (`test_tracking`, `test_osc_out`,
`test_blobs`, `test_depth`, `test_capture`, `test_cameras`, `test_kinect`
— the Kinect one uses a stand-in `pylibfreenect2` module), end-to-end runs
on the repository's pictures (`test_app`), the panel driven through its mouse
and keyboard entry points (`test_ui`, `test_themes`) and, on Linux with
`Xvfb` + `xdotool`, the real window (`test_gui_xvfb`). When you change
something, run the tests; when you add something, add one — the existing
ones show how to fake every piece of hardware.

## 9. Exercises

1. Send the *height* of every person (bounding box height) as
   `/mp/pose/height id h` and make a Processing sketch change a circle's size.
2. Add a *Smooth* stepper to the panel bound to `cfg.smooth` (the trackers'
   `smooth` attribute can be changed at run time).
3. Write a template where each tracked id has its own colour from a palette
   you like, and the ids are drawn as big numbers.
4. With a ceiling camera and Blobs, make a Pure Data / SuperCollider patch
   that plays one voice per blob, panned by `x` — then try `--z-from-kinect`
   and map the distance to the filter cutoff.
5. Make the composite source draw a thin separator line between cameras (in
   `CompositeSource.read`) and add a `--separator` option for its colour.
