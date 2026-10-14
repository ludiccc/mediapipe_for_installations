# Hand landmarks — openFrameworks

_Castellano:_ recibe por OSC (ofxOsc) las manos que manda
`mediapipe_osc.py --hands`: 21 puntos por mano, izquierda/derecha y la distancia
entre pulgar e índice (pinza), que se usa como evento.  `make && make RunRelease`
en esta carpeta (dentro de `<openFrameworks>/apps/openFrameworks-examples/`),
o abrir el proyecto de Xcode / Visual Studio generado con el projectGenerator.

Receives the **Hands** detector (`/mp/hand/...`) and draws every hand: the 21
landmarks with their connections, the fingertips bigger, a colour per
handedness (Left / Right), the bounding box, the velocity arrow and the pinch —
the distance between the thumb tip and the index tip — as a circle that is
filled while the fingers are "pinching".  The pinch closing / opening is
logged as an event.

## Install

openFrameworks 0.10 or newer; ofxOsc ships with it.  This folder must be
inside `<openFrameworks>/apps/<something>/` or `OF_ROOT` must be set — see
`../README.md`.

## Build and run

```
# terminal 1: the camera
python mediapipe_osc.py --hands --mirror

# terminal 2, in this folder:
make && make RunRelease          # or from ../:  make run-hand_landmarks
```

Xcode / Visual Studio: generate the project (`../generate_projects.sh`,
`..\generate_projects.bat` or the projectGenerator app), open it, run.
`q` or `Esc` quits.  Without a camera: `python osc_simulator.py`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/hand/enter`, `/mp/hand/leave` | `id` | |
| `/mp/hand/info` | `id handedness score` | `"Left"` / `"Right"` (with `--mirror`, as the viewer sees them) |
| `/mp/hand/center` | `id x y z` | |
| `/mp/hand/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/hand/velocity` | `id vx vy speed` | |
| `/mp/hand/pinch` | `id distance` | thumb tip – index tip, roughly metres |
| `/mp/hand/landmarks` | `id` + 21 × `x y z` | 0 wrist, 4 thumb tip, 8 index tip, 12, 16, 20 other tips |

## Where to start changing it

* `PINCH_CLOSED` (`ofApp.h`) — the threshold of the pinch event (depends on
  the distance to the camera; watch the logged values).
* the `/mp/hand/pinch` branch of `handleMessage()` — the event: start
  something when it closes, stop when it opens.
* `landmarks[INDEX_TIP]` — a fingertip as a cursor: draw where it points, or
  test whether it is inside a region of the window.
* Two hands: `hands` holds all of them; the distance between two `center`s is
  a simple "stretch" control.
