# Body landmarks — Processing

_Castellano:_ recibe por OSC (oscP5) los 33 puntos del cuerpo que manda
`mediapipe_osc.py --pose` y dibuja el esqueleto de cada persona con su id, en
qué tercio de la imagen está y si levanta una mano.  Abrir `body_landmarks.pde`
en Processing y apretar Run (y en una terminal `python mediapipe_osc.py --pose`).

Receives the **Body** detector (`/mp/pose/...`) and draws, for every person:
the skeleton (33 landmarks joined by MediaPipe's connections, bigger points
when more visible), the bounding box, the centre with a velocity arrow, the
id, the third of the picture the person stands in (left / centre / right) and
a "HAND UP!" flag when a wrist is above the shoulders.

## Install

Processing 3 or 4 and the **oscP5** library (*Sketch > Import Library... >
Manage Libraries... > oscP5*).

## Run

```
# terminal: the camera
python ../../mediapipe_osc.py --pose --mirror
```

Open `body_landmarks.pde` in Processing and press **Run**, or from
`processing-examples/`: `make run-body_landmarks`
(= `processing-java --sketch=$PWD/body_landmarks --run`).

Without a camera: `python ../../osc_simulator.py` instead of the app.

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

* `Person.zone()` — the "where is the visitor" logic (three columns).
* `Person.handRaised()` — a gesture made from landmarks; compare any two
  points the same way.
* `draw()` — replace the skeleton by your own graphics (`image()`, particles,
  `PShape`...); multiply the normalized coordinates by `width` / `height`.
* the `enter` / `leave` branches of `oscEvent()` — start and stop sounds or
  animations per person.
