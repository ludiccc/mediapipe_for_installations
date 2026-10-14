# Hand gestures — openFrameworks

_Castellano:_ recibe por OSC (ofxOsc) los gestos de mano que manda
`mediapipe_osc.py --gesture` (puño, palma abierta, pulgar arriba, victoria...)
y los convierte en eventos: cada cambio de gesto se registra, cada gesto tiñe
el fondo de un color y un pulgar arriba hace un flash.  `make && make RunRelease`
en esta carpeta (dentro de `<openFrameworks>/apps/openFrameworks-examples/`),
o abrir el proyecto de Xcode / Visual Studio generado con el projectGenerator.

Receives the **Gestures** detector (`/mp/hand/gesture`, plus the hand
messages) and shows the recognised gesture of every hand with its score bar.
The useful part for an installation is turning the stream of gestures (one per
frame) into *events*: the app logs every gesture **change**, tints the
background with a colour per gesture and flashes the screen on a `Thumb_Up`.

The eight gestures of MediaPipe's recogniser: `None`, `Closed_Fist`,
`Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

## Install

openFrameworks 0.10 or newer; ofxOsc ships with it.  This folder must be
inside `<openFrameworks>/apps/<something>/` or `OF_ROOT` must be set — see
`../README.md`.

## Build and run

```
# terminal 1: the camera  (--gesture implies --hands)
python mediapipe_osc.py --gesture --mirror

# terminal 2, in this folder:
make && make RunRelease          # or from ../:  make run-hand_gestures
```

Xcode / Visual Studio: generate the project (`../generate_projects.sh`,
`..\generate_projects.bat` or the projectGenerator app), open it, run.
`q` or `Esc` quits.  Without a camera: `python osc_simulator.py` (the
simulated hands change gesture every two seconds).

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/hand/gesture` | `id name score` | every frame, per hand |
| `/mp/hand/enter`, `/mp/hand/leave` | `id` | |
| `/mp/hand/info` | `id handedness score` | |
| `/mp/hand/center`, `/mp/hand/bbox` | `id x y z`, `id xmin ymin xmax ymax` | |
| `/mp/hand/landmarks` | `id` + 21 × `x y z` | drawn as a thin outline |

## Where to start changing it

* the `/mp/hand/gesture` branch of `handleMessage()` — the event logic.
  `MIN_SCORE` filters doubtful classifications; a gesture that must be *held*
  for a while before it counts is a timer away.
* `GESTURE_NAMES` / `GESTURE_COLORS` / `GESTURE_ICONS` (`ofApp.cpp`) — one
  colour / icon per gesture; map them to sounds, scenes, lights...
* The gesture comes with the hand's `center` and `handedness`, so "Open_Palm
  with the left hand on the right side of the picture" is a simple `if`.
