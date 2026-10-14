# Hand landmarks — Processing

_Castellano:_ recibe por OSC (oscP5) las manos que manda
`mediapipe_osc.py --hands`: 21 puntos por mano, izquierda/derecha y la distancia
entre pulgar e índice (pinza), que se usa como evento.  Abrir
`hand_landmarks.pde` en Processing y apretar Run (y en una terminal
`python mediapipe_osc.py --hands`).

Receives the **Hands** detector (`/mp/hand/...`) and draws every hand: the 21
landmarks with their connections, the fingertips bigger, a colour per
handedness (Left / Right), the bounding box, the velocity arrow and the pinch —
the distance between the thumb tip and the index tip — as a circle that is
filled while the fingers are "pinching".  The pinch closing / opening is
printed as an event.

## Install

Processing 3 or 4 and the **oscP5** library (*Sketch > Import Library... >
Manage Libraries... > oscP5*).

## Run

```
# terminal: the camera
python ../../mediapipe_osc.py --hands --mirror
```

Open `hand_landmarks.pde` in Processing and press **Run**, or from
`processing-examples/`: `make run-hand_landmarks`.
Without a camera: `python ../../osc_simulator.py` instead of the app.

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

* `PINCH_CLOSED` — the threshold of the pinch event (depends on the distance
  to the camera; watch the printed values).
* the `/mp/hand/pinch` branch of `oscEvent()` — the event: start something
  when it closes, stop when it opens.
* `landmarks[INDEX_TIP]` — a fingertip as a cursor: draw where it points, or
  test whether it is inside a region of the window.
* Two hands: `hands` holds all of them; the distance between two `center`s is
  a simple "stretch" control.
