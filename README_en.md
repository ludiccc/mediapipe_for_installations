# MediaPipe for installations

Simple MediaPipe examples for electronic/digital/robotic art installations

These examples are for academic use. Do not take them as a formal reference on how to use MediaPipe. For that, refer to [https://developers.google.com/mediapipe/](https://developers.google.com/mediapipe/)

# mediapipe_osc.py: the all-in-one app

`mediapipe_osc.py` combines the examples below in a single configurable app:
pose + hands + gestures + face in one process, numeric OSC arguments, stable
IDs per person/hand/face with `enter`/`leave` events and velocity, a control
panel in the preview window (detectors, video source — a list of the available
cameras and Kinects; check several and they are analyzed side by side, in the
order you choose —, pause, IP/port, formats, mirror — all changeable while
running), Kinect input (v2 and v1) with **Z from Kinect** (every landmark's z
becomes the measured distance in metres), mirror mode, smoothing, headless
mode, and compatibility formats for the existing Processing sketches
(`--format legacy`) and the `Sonido_Cuadrafonico` scripts (`--format kinect`).

It also has a **Blobs** detector (OpenCV, no models): for a ceiling camera,
silhouettes or objects it finds the shapes that stand out from the background,
enumerates and follows them with IDs (`/mp/blob/...`, and `/kinect/person` for
the quad-sound scripts, as the original Kinect did). And **templates** for the
look of the window (`--template minimal | ascii | baroque`, or your own file;
see `templates/`).

It can also run on a video file or an image instead of a webcam, and comes
with tests. See [README_mediapipe_osc.md](README_mediapipe_osc.md) (manual)
and [GUIDE_mediapipe_osc.md](GUIDE_mediapipe_osc.md) (how it works inside,
for students). Spanish versions are
[README_mediapipe_osc_es.md](README_mediapipe_osc_es.md) and
[GUIDE_mediapipe_osc_es.md](GUIDE_mediapipe_osc_es.md).

```
python mediapipe_osc.py --pose --hands --mirror
python mediapipe_osc.py --list-cameras                      # which cameras are there
python mediapipe_osc.py --blobs --format kinect --port 12000   # ceiling camera -> Sonido_Cuadrafonico
python mediapipe_osc.py --template ascii                    # another look
python osc_monitor.py          # in another terminal: shows what arrives
```

# Receiver examples: Python, Processing, openFrameworks, TouchDesigner

For each detector of `mediapipe_osc.py` — **body** (pose), **hands**, **hand
gestures**, **face** and **blobs** — there is a receiver example in each of
the four environments used in the course.  They are starting points for an
installation: they receive the OSC messages, keep every person / hand / face /
blob by its id and draw what arrives; each one also prints the typical
*events* (somebody entered, a pinch closed, a gesture changed, a mouth opened,
a blob changed zone).

| folder | what | how to try it |
|--------|------|---------------|
| [`python-examples/`](python-examples/) | 5 scripts (python-osc + OpenCV) | `make all` runs them one after the other; `make test` without a camera |
| [`processing-examples/`](processing-examples/) | 5 sketches (oscP5) | the Processing IDE, or `make all` with `processing-java` |
| [`openFrameworks-examples/`](openFrameworks-examples/) | 5 projects (ofxOsc), Makefiles + Xcode / Visual Studio project generation | `make all` inside `<openFrameworks>/apps/` |
| [`touchdesigner-examples/`](touchdesigner-examples/) | 5 scripts that build the network inside TouchDesigner | drag the `.py` into a network, *Run Script* |

Every folder and every example has its own `README.md` with the installation
and the commands.  Without a camera, `python osc_simulator.py` sends made-up
data in the exact format of the app for all detectors; `make test` in each
folder uses it (and, for Processing and openFrameworks, compiles the examples
against stand-in classes in `tests/stubs/`, so neither Processing nor
openFrameworks needs to be installed for the check).

# How to use this examples

An installed Python and a Mediapipe environment is mandatory for using these examples. You can follow the official Mediapipe installation guide [https://developers.google.com/mediapipe/solutions/setup_python](https://developers.google.com/mediapipe/solutions/setup_python)

These python script examples have three versions: one for face landmarks recognition (face mesh), one for body landmarks recognition (pose landmark), and one for hand landmarks recognition (hand landmarks).

Taking the recognition of body nodes as an example:

Two processes are executed, one in python to do the recognition and another in whatever is going to receive the OSC messages, for example, the Processing sketch that is in this repository:

```python ejemplo-pose-landmark-osc.py```

The OSC messages coming from the python script contain a number of arguments equal to the number of detected nodes. That is, in the case of *ejemplo-pose-landmark-osc.py*, up to 32 nodes are recognized, according to ![Pose Landmarks list](https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png). The script will then send a message with 32 arguments, each containing the three position values (*x*, *y*, *z*) for that node, separated by a space. The values are normalized, that is, they are between 0 and 1.

In the Processing example, these values are broken down and added to a PVector, then drawn on the screen.
