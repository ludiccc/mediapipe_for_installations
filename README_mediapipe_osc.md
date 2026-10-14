# mediapipe_osc.py — live camera → MediaPipe / OpenCV blobs → OSC

One configurable app that analyzes a live camera with MediaPipe (body pose,
hands, gestures, face) or with plain OpenCV (**Blobs**: shapes seen from a
ceiling camera or a Kinect) and broadcasts the results as OSC messages, meant
for installations driven by Processing, TouchDesigner, Pure Data, Max,
SuperCollider, Unity, Python, etc.

This is the user manual. **How the program works inside** — the pipeline,
the modules, the threads, how to extend it — is explained for students in
[GUIDE_mediapipe_osc.md](GUIDE_mediapipe_osc.md).

Compared with the `ejemplo-*.py` scripts in this repository it adds:

* **all detectors in one process** (`--pose --hands --face --gesture --blobs`), any combination
* **Blobs**: an OpenCV detector for installations where MediaPipe's body
  models do not apply — a camera on the ceiling, silhouettes, objects on a
  table, a Kinect depth image. Shapes are enumerated, followed with stable ids
  and sent as `/mp/blob/...` (and as `/kinect/person` for `Sonido_Cuadrafonico`)
* **numeric OSC arguments** (ints and floats) instead of `"x y z"` strings
* **stable IDs**: person/hand/face/blob *3* keeps its number while it is in front
  of the camera, plus `enter` / `leave` events and a **velocity** per track
* **compatibility modes** for the existing receivers of this repo: the
  Processing sketches (`--format legacy`) and the `Sonido_Cuadrafonico`
  quad-sound scripts (`--format kinect`, a webcam or the Blobs detector replaces
  the Kinect + Processing sketch)
* a **control panel inside the preview window**: detectors on/off, the video
  **source** (a list of the cameras and Kinects found on the computer — check
  several and they are analyzed **side by side**, in the order you choose —,
  pause), IP/port, formats, mirror — changed live, without stopping the stream
* **Kinect** input (v2 via pylibfreenect2, v1 via libfreenect, OpenNI): colour
  or depth image, and **Z from Kinect** — the z of every landmark becomes the
  measured distance in metres
* **templates** for the look & feel of the window (`--template`): `classic`,
  `minimal`, `ascii` and `baroque` are included, your own is one small Python
  file away
* mirror mode, smoothing, frame-rate cap, headless mode, OSC bundles
* a video file or still image can replace the webcam (`--source`), so the
  whole pipeline is testable without a camera — see `tests/`

```
python mediapipe_osc.py                                   # pose, webcam 0 → 127.0.0.1:5005
python mediapipe_osc.py --pose --hands --face --mirror    # everything, mirrored like a mirror
python mediapipe_osc.py --gesture --ip 192.168.1.20 --port 9000
python mediapipe_osc.py --list-cameras                    # which cameras are there?
python mediapipe_osc.py --source 2 --template minimal     # camera 2, minimalist window
python mediapipe_osc.py --source 0+2                      # two webcams side by side, analyzed as one image
python mediapipe_osc.py --no-kinects                      # webcams only (libfreenect2 never loaded)
python mediapipe_osc.py --no-webcams --blobs              # Kinects only; starts on the first one
python mediapipe_osc.py --source kinect --pose --z-from-kinect   # body landmarks with z in metres
python mediapipe_osc.py --blobs --format kinect --port 12000           # ceiling webcam → Sonido_Cuadrafonico
python mediapipe_osc.py --source kinect --blobs --format kinect --port 12000   # ceiling Kinect v2/v1 (depth)
python mediapipe_osc.py --source kinect --pose --kinect-pipeline cpu           # Kinect colour camera, CPU depth pipeline
python mediapipe_osc.py --format kinect --port 12000 --kinect-y size     # frontal webcam, body poses
python mediapipe_osc.py --source video.mp4 --no-preview --verbose        # no camera needed
python mediapipe_osc.py --help
```

## The window

The preview window has a control panel on its right side, so the usual
settings can be changed while the app runs, without stopping the preview or
the OSC stream:

* **Detectors**: Body / Hands / Gestures / Face / Blobs toggles (Gestures
  implies Hands; any combination can be on). Switching a detector off sends
  `leave` for its tracks; switching it on loads the model (about a second).
  For Blobs: **Learn bg** captures the empty scene again as the background
  (`b`), and two steppers set the **sensitivity** (grey-level threshold) and
  the **minimum area** (% of the image). While Blobs is on, a small copy of
  the detection mask is shown in the corner of the preview to help tuning.
* **Source** — a radio group, only one can be selected: **Cameras** (the
  entries checked in the list below), **Pause** (stop sending OSC; analysis
  and preview keep running so you can still see what happens), and **File**
  when the app was started on a video/image (to return to it).
  Below it, **one list with every camera found on the computer — webcams and
  Kinects alike** (a Kinect v2 shows as `kinect  Kinect v2 <serial>`). Click an
  entry to check it; **check several and their pictures are placed side by
  side**, left to right in the order of the numbers shown (`1.`, `2.` ...),
  and MediaPipe / Blobs analyze that wide composed image as one picture. The
  ◀ ▶ buttons at the right of a checked row move that camera left/right in
  the composition. Clicking a checked entry removes it (the last one stays).
  Cameras that stay in the composition are never re-opened: only the added
  one is opened and only the removed one is released (everybody `leave`s,
  fresh ids start on the new picture). The list is filled by a background
  scan at start-up (each webcam is opened for a moment, which can take a
  second per device); **Rescan cameras** (`c`) looks again after plugging one
  in — a camera that was just plugged in is retried for a couple of seconds
  while its driver settles. If a source cannot be opened the error is shown
  under the list and the previous source keeps running.
  `--no-webcams` / `--no-kinects` leave one kind out of the list (with
  `--no-kinects` libfreenect2 is never loaded; with `--no-webcams` the
  default source is the first Kinect).
* **⚙ Crop & mapping** — the gear at the right of every list row opens the
  mapping editor of that camera (it is started first if it was not checked).
  It is the `mapping_in_mapping_out` tool of *tpf-tools* built in: an
  **input** quad (orange) chooses which part of the camera picture is used —
  put its four corners on the corners of a floor or a table seen in
  perspective and that surface is straightened —, an **output** quad (cyan)
  says where that region lands in the camera's slot of the analyzed image
  (black outside). MediaPipe and Blobs see the mapped picture, so the OSC
  coordinates follow it. While the editor is open the preview shows the raw
  camera picture (input mode) or the mapped composition (output mode) with
  four numbered handles: drag them with the mouse, or `1`–`4` to pick one and
  the arrow keys to nudge it; `i` / `o` switch between input and output; `r`
  resets the current quad; `Enter` / `Esc` close the editor (`g` opens it for
  the first camera). The gear turns cyan while editing and stays highlighted
  on cameras that have a mapping. Mappings are saved in the configuration
  file in the tpf-tools JSON format (`{"input": [{"x","y"}×4], "output": …}`),
  so a `mapping_in_mapping_out.json` can be pasted in.
* **Save config / Load config…** — write / read a JSON file with everything
  the panel can change: the cameras and their order, detectors, blob
  settings, OSC target and formats, image options, template, mappings.
  *Load* opens the operating system's file chooser (tkinter; without it the
  last file is used); *Save* writes to the file loaded/saved before, or asks
  for a name the first time. `--config file.json` loads one at start-up. The
  result is shown under the buttons.
* **Z from Kinect** (`z`, in the *Image* section): replace MediaPipe's relative
  z of every landmark (and the blobs' z) by the distance measured by the Kinect
  at that pixel, in **metres** (0 = no reading). It is only available when the
  running source has depth: one Kinect, or *only Kinects* side by side — with a
  webcam in the composition it is greyed out and z stays MediaPipe's.
  Receivers are told which one they get with `/mp/z_source`.
* **Image**: Mirror, Draw (landmark drawing in the preview only).
* **OSC output**: IP/host and Port fields (click, type, `Enter` to apply,
  `Esc` to cancel — invalid values show an error under the field), the
  `mp` / `legacy` / `kinect` format toggles (these *can* be combined) and
  `Landmarks` (include the landmark / contour lists in the messages). When the
  target changes, the old receiver gets `/mp/status stopped` and the new one
  `/mp/status started`.
* **Look** cycles through the templates (`t`; `r` reloads the current template
  file after editing it) — see *Templates* below.
* a status area (fps, counts, messages sent, current target, source) and **Quit**
  (the window's own close button quits too).

Keyboard shortcuts (when no text field is focused and the mapping editor is
closed): `q`/`Esc` quit, `m` mirror, `l` landmarks, `i` ids, `p` pause/resume
sending, `b` learn the blob background, `c` rescan cameras, `z` Z from Kinect,
`g` crop & mapping of the first camera, `t` next template, `r` reload the
template.
`--no-panel` gives a plain preview with only the shortcuts; `--no-preview`
runs headless (e.g. on a server or with `opencv-python-headless`).

### Cameras

`python mediapipe_osc.py --list-cameras` prints the cameras OpenCV can open
and their names, e.g.

```
cameras found (use --source <index>):
  0  Integrated Camera  (/dev/video0)
  2  Logitech BRIO      (/dev/video2)
```

On Linux the names come from the kernel (`/sys/class/video4linux`); on macOS
from `system_profiler`; on Windows from `pygrabber` if it is installed
(`pip install pygrabber`), otherwise cameras are called "Camera N". Kinects
are listed too (`kinect`, `kinect:1` ...). Names are best effort — the first
column is the *source spec*, and it is what `--source` takes:

| spec | source |
|------|--------|
| `0`, `2` | webcam by index |
| `kinect` | the first Kinect; `kinect:1` the second Kinect v2; `kinect:freenect` forces the v1 backend |
| `0+2`, `0+kinect`, `kinect+kinect:1` | several sources **side by side**, left to right (`CompositeSource`) |
| `clip.mp4`, `photo.jpg` | a video file / a still image |

With a composition, every part is scaled to the height of the first one and
the frame is as wide as all of them together; coordinates (x, y) are
normalized to that whole frame (a person in the right-hand one of two equal
cameras has x > 0.5). The same camera cannot be opened twice.

### Kinect

`--source kinect` (or the **Kinect** radio button) reads a depth sensor through
the first of these backends that finds a device (`--kinect-backend` forces one):

* **`freenect2` — Kinect v2** through [pylibfreenect2](https://github.com/r9y9/pylibfreenect2)
  (libfreenect2): the sensor is enumerated, opened by serial number with a
  depth-processing pipeline (`--kinect-pipeline opengl | opencl | cuda | cpu`,
  `auto` = the first one that initialises, in that order) and read through a
  `SyncMultiFrameListener`. Colour frames are 1920×1080, depth 512×424 in
  millimetres. If your pylibfreenect2 only imports once libfreenect2 is loaded
  (Windows, Homebrew, custom install folder), set `LIBFREENECT2_LIBRARY` to the
  full path of `freenect2.dll` / `libfreenect2.dylib` / `libfreenect2.so`.
* **`freenect` — Kinect v1** through the python bindings of libfreenect.
* **`openni`** — any OpenNI device, when OpenCV was built with OpenNI2.

A Kinect is simply another entry of the camera list (checked alone, or
together with webcams / other Kinects). `--source kinect` starts on it.
`--kinect-stream color` gives the colour camera (for the MediaPipe detectors),
`--kinect-stream depth` the depth map as a grey image (near = bright, unknown =
black, 0–4.5 m) — the signal the original `Sonido_Cuadrafonico` ceiling sketch
worked with, ideal for Blobs. The default `auto` picks depth when Blobs is the
only detector and colour otherwise. `--width` scales the Kinect frames to that
width (aspect kept; `--width 0` = native size). Without a sensor or the
libraries the panel says *no Kinect found* and the terminal lists what each
backend reported; the previous source keeps running.

**Z from Kinect** (`--z-from-kinect`, the toggle, or `z`): the sensor's depth
image is re-projected onto the colour image (Kinect v2: libfreenect2's
`Registration` / `bigdepth`; v1: `DEPTH_REGISTERED`; OpenNI: registration
on), so the distance of every colour pixel is known; each landmark's z is
then the median of the valid depth pixels in a 5×5 window around it, in
metres. This costs a few milliseconds per frame and is only computed while
the toggle is on. With two Kinects side by side (`kinect+kinect:1`) both depth
maps are composed as well; with a webcam in the composition there is no depth
and the toggle is greyed out.

(The Kinect code follows the documented APIs and the structure of a working
pylibfreenect2 program, and is exercised by tests against a stand-in module,
but it could not be tried on a real sensor in the development environment —
please report what the terminal prints if a backend fails.)

## Install

Python 3.9+ with:

```
pip install mediapipe opencv-python numpy python-osc
```

(`requirements.txt` of the repo pins mediapipe 0.10.3; the app was developed and
tested with 0.10.14 and works with both API-wise — it only uses the MediaPipe
*Tasks* API, `detect_for_video`.) The `.task` model files must be in the
repository folder (they are) or in `--models-dir`.

## OSC messages (`--format mp`, the default)

Types: `i` = int32, `f` = float32, `s` = string. `x`, `y` are normalized to the
image (0..1, origin top-left; with `--mirror`, x grows to the *viewer's* right;
with several cameras side by side, to the whole composed image).
`z` is MediaPipe's relative depth (smaller = closer) — or, with *Z from Kinect*,
the distance in metres (`/mp/z_source kinect`). Landmark coordinates of
body parts outside the image are extrapolated by MediaPipe and can fall outside
0..1. `<id>` is the stable tracking id; ids of poses, hands and faces are
independent counters. A returning person gets a new id.

```
/mp/status            s:"started"|"stopped"  s:version
/mp/z_source          s:"mediapipe"|"kinect"   what z means: relative depth, or metres from the Kinect
                                               (sent at start-up and whenever it changes)
/mp/frame             i:frame  f:time_s  f:fps  i:n_poses  i:n_hands  i:n_faces  i:n_blobs   (every frame)

/mp/pose/enter        i:id
/mp/pose/leave        i:id
/mp/pose/center       i:id  f:x f:y f:z      visibility-weighted mean of nose, shoulders, hips
/mp/pose/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
/mp/pose/velocity     i:id  f:vx f:vy f:speed      normalized units per second
/mp/pose/landmarks    i:id  33 × (f:x f:y f:z f:visibility)      (--no-landmarks to skip)
/mp/pose/world        i:id  33 × (f:x f:y f:z)  meters, hip-centered   (--world)

/mp/hand/enter|leave  i:id
/mp/hand/info         i:id  s:handedness ("Left"/"Right")  f:score
/mp/hand/center       i:id  f:x f:y f:z
/mp/hand/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
/mp/hand/velocity     i:id  f:vx f:vy f:speed
/mp/hand/pinch        i:id  f:distance                thumb tip ↔ index tip, ≈ meters
/mp/hand/gesture      i:id  s:name f:score            (--gesture) e.g. Open_Palm, Thumb_Up, Victory, "None"
/mp/hand/landmarks    i:id  21 × (f:x f:y f:z)

/mp/face/enter|leave  i:id
/mp/face/center       i:id  f:x f:y f:z              nose tip
/mp/face/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
/mp/face/velocity     i:id  f:vx f:vy f:speed
/mp/face/blendshapes  i:id  52 × f:value             0..1, order given by /mp/face/blendshape_names
/mp/face/blendshape_names   52 × s:name              sent once at start-up
/mp/face/landmarks    i:id  478 × (f:x f:y f:z)      (--face-landmarks, a big message)

/mp/blob/enter|leave  i:id                            (--blobs)
/mp/blobs             i:count  count × i:id           the blobs on screen, biggest first (every frame)
/mp/blob/center       i:id  f:x f:y f:0               centroid
/mp/blob/bbox         i:id  f:xmin f:ymin f:xmax f:ymax
/mp/blob/velocity     i:id  f:vx f:vy f:speed
/mp/blob/info         i:id  f:area f:angle f:length f:width
                      area = fraction of the image (0..1); angle = main axis in degrees
                      (-90..90, clockwise on screen); length/width = equivalent ellipse, normalized
/mp/blob/contour      i:id  16 × (f:x f:y)            outline, evenly spaced from the topmost point
                      (--no-landmarks / the Landmarks toggle turns it off)
```

Order within a frame: `/mp/frame`, then `enter` events, then the data of every
active track, then `leave` events (per kind: poses, hands, faces, blobs). On
exit the app sends `leave` for every track and `/mp/status stopped`, so
receivers can reset.

Landmark indices are the official ones:
[pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarker_model),
[hand](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models),
[face](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models).
Blendshape names are the 52 ARKit-style names (`jawOpen`, `eyeBlinkLeft`, `mouthSmileRight`, …).

### `--id-style address`

Moves the id into the address, which suits tools that route by address
(TouchDesigner OSC In CHOP, Pd `[routeOSC]`, Max `[route]`):

```
/mp/pose/3/center   f:x f:y f:z
/mp/pose/3/landmarks ...
/mp/pose/enter      i:3          (enter/leave keep the id as argument)
```

`--prefix` changes `/mp` (e.g. `--prefix /cam1` when running two cameras).
`--bundle` packs each frame's messages into one OSC bundle (atomic per frame).

### Other formats

`--format legacy` — the string messages of the original scripts, so the
existing sketches (`osc_mp_simple_example`, `osc_mp_hand_gesture_simple_example`)
keep working: `pose_<n>` with 33 `"x y z visibility presence"` strings,
`hand_landmarks` / `distancia_pulgar_indice` / `/gesture` like
`ejemplo-hand-landmarks.py` and `ejemplo-gesture-recognizer-osc.py`, and
`face_<n>` with 478 `"x y z"` strings (the original face script reused the
`pose_<n>` address; it is renamed here so pose and face can run together).
`<n>` is the position in the frame, not a stable id.

`--format kinect` — `/kinect/person i:id f:x f:y f:speed` (all 0..1) as
expected by `Sonido_Cuadrafonico/recepcion_datos.py`, which was written for a
ceiling Kinect + Processing. Use `--port 12000`. With a ceiling-mounted webcam
the image y is the room depth (`--kinect-y image`, default); with a frontal
webcam use `--kinect-y size`, which derives y from the person's apparent size
(bigger = closer = front of the room). `--invert-x` mirrors x, `--max-speed`
sets which speed (normalized units/s) maps to 1.0.

With `--blobs` the `/kinect/person` messages come from the blobs instead of the
body poses (that is what the original ceiling Kinect did); `--kinect-from pose`
or `blobs` forces one or the other.

Formats can be combined: `--format mp kinect`.

## Blobs: shapes instead of bodies

`--blobs` (or the **Blobs** toggle) runs an OpenCV pipeline with no MediaPipe
model: grey → blur → foreground mask → open/close → contours → the shapes
between `--blob-min-area` and `--blob-max-area` (fractions of the image;
`0.004` ≈ 1200 px at 640×480, the value of the original Processing sketch) →
the `--blob-max` largest ones. Each blob gets a centroid, bounding box, area,
orientation and a 16-point contour, and goes through the same tracker as the
other detectors: stable id, `enter` / `leave`, velocity.

How the foreground mask is made (`--blob-mode`):

* `background` (default) — the difference to a **learned static background**,
  thresholded by `--blob-threshold` (the sensitivity, 1..255; lower = more
  sensitive). The background is the average of the first
  `--blob-learn-frames` frames, so **start the app with the scene empty**, or
  press `b` / **Learn bg** when it is (the preview shows *learning
  background …* meanwhile). Anything that is not part of that picture is a
  blob — people standing still stay detected. `--blob-adapt 0.01` lets the
  background slowly follow lighting changes everywhere except inside the
  blobs.
* `motion` — OpenCV's adaptive MOG2 model: no calibration, but whatever stops
  moving fades into the background after some seconds.
* `bright` / `dark` — a fixed threshold on the grey level: light objects on a
  dark floor, dark objects on a light floor, or a Kinect depth image (near =
  bright).

`--blob-roi x0 y0 x1 y1` limits detection to a rectangle (normalized), like
the `minX..maxY` window of the original sketch; `--blob-blur` / `--blob-morph`
set the kernel sizes (0 = off).

A ceiling webcam feeding the quad-sound scripts of this repo:

```
python mediapipe_osc.py --blobs --format kinect --port 12000          # terminal 1
python Sonido_Cuadrafonico/10_instalacion_final.py                    # terminal 2
```

With the Kinect itself: `--source kinect --blobs --format kinect --port 12000`
(depth stream, `--blob-mode bright` also works well on it since near = bright).

## Templates: the look & feel of the window

Everything that is drawn — the panel, its widgets, the camera image, the
landmarks — goes through a *template* (a `Theme` subclass). The default look is
`classic`; three examples live in [`templates/`](templates/):

| template  | look |
|-----------|------|
| `minimal` | white panel, hairlines, nothing but type; thin white landmarks on the untouched image |
| `ascii`   | the whole window is a grid of characters: the video becomes ASCII art, landmarks are typed over it, `[x]` toggles and `(*)` radios |
| `baroque` | gold on burgundy, serif and script fonts, gilded plaques and frames, sepia + vignette on the picture |

```
python mediapipe_osc.py --template ascii
python mediapipe_osc.py --template path/to/my_template.py
```

The **Look** button / `t` cycles through all templates while running, `r`
reloads the current file (edit, save, press `r`). A template is a Python file
that defines `class Template(Theme)` and overrides what it wants: colours,
fonts, panel geometry (`panel_width`, `row`, `panel_side = "left"`), the
drawing of each widget (`draw_toggle`, `draw_radio`, `draw_button`,
`draw_field`, `draw_list`, `draw_stepper`, `draw_section`, `draw_status`) and
of the preview (`draw_preview`, `draw_banner`). A template can only change how
things look; it never touches OSC or the detectors, and a template that fails
to load leaves the current look in place. See [`templates/README.md`](templates/README.md).

## Tracking details

* Matching is nearest-center between consecutive frames, limited by
  `--track-distance` (normalized units, default 0.25). A detection farther than
  that from every known track becomes a new id.
* A track that is not detected keeps its id for `--track-timeout` seconds
  (default 0.5) and is then reported as `leave`. Nothing is sent for a track
  while it is missing.
* `--smooth 0.0–0.99` applies exponential smoothing to landmarks and centers
  (0 = raw). Velocity is always lightly smoothed.
* Video files use their own frame rate as clock, so velocities and timeouts
  are right even when a file is processed faster than real time; use
  `--max-fps 30` to play it in real time.

## Receivers

* `python-examples/`, `processing-examples/`, `openFrameworks-examples/`,
  `touchdesigner-examples/` — one receiver per detector (body, hands,
  gestures, face, blobs) in each environment, each with a README, a
  Makefile to build / run / test them, and the events an installation
  reacts to.  Start there.
* `osc_simulator.py` — sends made-up data with the exact schema of the `mp`
  format for every detector (people who walk in and out, hands with
  gestures, faces with blendshapes and mesh, blobs), to test any receiver
  without a camera: `python osc_simulator.py --only pose,blobs --port 5005`.
* `osc_mp_tracking_example/` — Processing sketch for the `mp` format (ids,
  bounding boxes, velocity arrows, hands). Written alongside the app but not
  run here (no Processing in the development environment).
* `osc_mp_blobs_example/` — Processing sketch for the blob messages (outlines,
  ids, velocity). Same caveat.
* `osc_monitor.py` — prints whatever arrives on a port.
* `Sonido_Cuadrafonico/*.py` — quad/stereo sound driven by `/kinect/person`
  (from body poses or from blobs).

## Tests

```
pip install pytest
python -m pytest tests/ -q
```

`tests/test_tracking.py` and `tests/test_osc_out.py` are pure unit tests
(tracker, message schema, encoding, CLI). `tests/test_app.py` runs the real
models on the images in the repository and on a synthetic video, checks the
messages over a real UDP socket, and checks that `Sonido_Cuadrafonico`'s own
receiver accepts the `kinect` format. `tests/test_capture.py` tests the
threaded webcam reader against a fake camera. `tests/test_ui.py` drives the
control panel through its mouse/keyboard entry points and checks runtime
reconfiguration (detectors on/off, OSC retargeting over UDP).
`tests/test_blobs.py` runs the Blobs detector on synthetic overhead scenes
(background learning, modes, areas, ROI, ids, OSC, the whole app on a video).
`tests/test_cameras.py` covers camera enumeration (fake probes, Kinects), the
Source radio group, switching cameras / Kinect / file at run time, several
cameras side by side and their ordering, Z from Kinect with two Kinects, and
failure handling. `tests/test_depth.py` tests the depth sampling. `tests/test_mapping.py`
covers the crop & mapping (warps, tpf-tools JSON, the editor's mouse/keys,
the gear, saving and loading configuration files). `tests/test_themes.py` loads and renders every template and checks
switching/reloading. `tests/test_kinect.py` drives the Kinect v2 backend
against a stand-in `pylibfreenect2` module (pipeline fallback, colour/depth
frames, errors). No webcam or display is required for these.
`tests/test_gui_xvfb.py` additionally drives the real window with `xdotool`
under `Xvfb` (Linux only, skipped if absent): clicks, typing, the Pause radio
button, a template switch, and closing the window with its [x].

## Code layout

```
mediapipe_osc.py      entry point
mp_osc/config.py      options (Config dataclass + argparse)
mp_osc/capture.py     webcam (threaded, latest frame), Kinect (3 backends), several side by side, video, image
mp_osc/cameras.py     enumeration of the webcams and Kinects of the computer (+ background scan)
mp_osc/detectors.py   MediaPipe Tasks wrappers → numpy arrays (+ runs the Blobs detector)
mp_osc/blobs.py       OpenCV blob detector (background / motion / bright / dark)
mp_osc/depth.py       Z from Kinect: landmark z ← measured distance
mp_osc/mapping.py     crop & mapping per camera (input/output quads, tpf-tools JSON) + the editor
mp_osc/dialogs.py     native open/save file dialogs (tkinter)
mp_osc/tracking.py    ids, enter/leave, velocity, smoothing
mp_osc/osc_out.py     message schema (mp / legacy / kinect) and UDP sending
mp_osc/themes.py      Theme base class (the classic look) and the template loader
mp_osc/preview.py     default drawing of landmarks / blobs / ids
mp_osc/ui.py          control panel widgets (toggle, radio, list, stepper, field) drawn inside the OpenCV window
mp_osc/app.py         main loop, runtime settings, source switching, window/analysis threads
templates/*.py        look & feel templates: minimal, ascii, baroque
```
