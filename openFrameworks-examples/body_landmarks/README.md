# Body landmarks — openFrameworks

_Castellano:_ recibe por OSC (ofxOsc) los 33 puntos del cuerpo que manda
`mediapipe_osc.py --pose` y dibuja el esqueleto de cada persona con su id, en
qué tercio de la imagen está y si levanta una mano.  `make && make RunRelease`
en esta carpeta (puesta dentro de `<openFrameworks>/apps/openFrameworks-examples/`),
o abrir el proyecto de Xcode / Visual Studio generado con el projectGenerator.

Receives the **Body** detector (`/mp/pose/...`) and draws, for every person:
the skeleton (33 landmarks joined by MediaPipe's connections, bigger points
when more visible), the bounding box, the centre with a velocity arrow, the
id, the third of the picture the person stands in (left / centre / right) and
a "HAND UP!" flag when a wrist is above the shoulders.

## Install

openFrameworks 0.10 or newer; the ofxOsc addon ships with it (`addons.make`
lists it).  This folder must be inside `<openFrameworks>/apps/<something>/`
(e.g. `apps/openFrameworks-examples/body_landmarks`) or `OF_ROOT` must be set
— see `../README.md`.

## Build and run

```
# terminal 1: the camera
python mediapipe_osc.py --pose --mirror

# terminal 2, in this folder (Linux, macOS command line, MSYS2):
make && make RunRelease          # or from ../:  make run-body_landmarks
```

Xcode / Visual Studio: generate the project (`../generate_projects.sh`,
`..\generate_projects.bat`, or the projectGenerator app with *import*), open
`body_landmarks.xcodeproj` / `body_landmarks.sln`, run.
`q` or `Esc` quits.  Without a camera: `python osc_simulator.py` instead of the app.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/pose/enter` | `id` | a new person |
| `/mp/pose/leave` | `id` | gone |
| `/mp/pose/center` | `id x y z` | centre of the body (visibility-weighted) |
| `/mp/pose/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/pose/velocity` | `id vx vy speed` | normalized units per second |
| `/mp/pose/landmarks` | `id` + 33 × `x y z visibility` | the skeleton |

`x`, `y` are 0..1 (origin top-left); landmarks outside the picture are
extrapolated and can be < 0 or > 1.  Numbering: 0 nose, 11/12 shoulders,
13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles
([full list](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)).

## Where to start changing it

* `Person::zone()` — the "where is the visitor" logic (three columns).
* `Person::handRaised()` — a gesture made from landmarks; compare any two
  points the same way.
* `ofApp::draw()` — replace the skeleton by your own graphics (`ofImage`,
  particles, `ofMesh`...); multiply the normalized coordinates by the window size.
* the `enter` / `leave` branches of `handleMessage()` — start and stop sounds
  or animations per person.
