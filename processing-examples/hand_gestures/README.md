# Hand gestures — Processing

_Castellano:_ recibe por OSC (oscP5) los gestos de mano que manda
`mediapipe_osc.py --gesture` (puño, palma abierta, pulgar arriba, victoria...)
y los convierte en eventos: cada cambio de gesto se registra, cada gesto tiñe
el fondo de un color y un pulgar arriba hace un flash.  Abrir
`hand_gestures.pde` en Processing y apretar Run (y en una terminal
`python mediapipe_osc.py --gesture`).

Receives the **Gestures** detector (`/mp/hand/gesture`, plus the hand
messages) and shows the recognised gesture of every hand with its score bar.
The useful part for an installation is turning the stream of gestures (one per
frame) into *events*: the sketch logs every gesture **change**, tints the
background with a colour per gesture and flashes the screen on a `Thumb_Up`.

The eight gestures of MediaPipe's recogniser: `None`, `Closed_Fist`,
`Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`.

## Install

Processing 3 or 4 and the **oscP5** library (*Sketch > Import Library... >
Manage Libraries... > oscP5*).

## Run

```
# terminal: the camera  (--gesture implies --hands)
python ../../mediapipe_osc.py --gesture --mirror
```

Open `hand_gestures.pde` in Processing and press **Run**, or from
`processing-examples/`: `make run-hand_gestures`.
Without a camera: `python ../../osc_simulator.py` (the simulated hands change
gesture every two seconds).

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/hand/gesture` | `id name score` | every frame, per hand |
| `/mp/hand/enter`, `/mp/hand/leave` | `id` | |
| `/mp/hand/info` | `id handedness score` | |
| `/mp/hand/center`, `/mp/hand/bbox` | `id x y z`, `id xmin ymin xmax ymax` | |
| `/mp/hand/landmarks` | `id` + 21 × `x y z` | drawn as a thin outline |

## Where to start changing it

* the `/mp/hand/gesture` branch of `oscEvent()` — the event logic.
  `MIN_SCORE` filters doubtful classifications; a gesture that must be *held*
  for a while before it counts is a timer away.
* `GESTURE_NAMES` / `GESTURE_RGB` / `GESTURE_ICONS` — one colour / icon per
  gesture; map them to sounds, scenes, lights...
* The gesture comes with the hand's `center` and `handedness`, so "Open_Palm
  with the left hand on the right side of the picture" is a simple `if`.
