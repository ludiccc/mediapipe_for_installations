# Face landmarks — Processing

_Castellano:_ recibe por OSC (oscP5) las caras que manda
`mediapipe_osc.py --face --face-landmarks`: caja, punta de la nariz, malla de
478 puntos, 52 *blendshapes* (boca abierta, sonrisa, parpadeo...) y dibuja una
"máscara" orientada con la inclinación de la cabeza, donde una instalación
pondría su propia imagen.  Abrir `face_landmarks.pde` en Processing y apretar
Run (y en una terminal `python mediapipe_osc.py --face --face-landmarks`).

Receives the **Face** detector (`/mp/face/...`) and draws, for every face: the
bounding box, the nose tip, the 478-point mesh and the face oval (when the app
sends the mesh), the head roll computed from the outer eye corners, an ellipse
"mask" placed and rotated over the face — where an installation would draw its
own image, a bird's head for instance — and bars for the most expressive
blendshapes (`jawOpen`, `mouthSmileLeft/Right`, `eyeBlinkLeft/Right`,
`browInnerUp`, `mouthPucker`, `cheekPuff`).  Opening the mouth is printed as an event.

## Install

Processing 3 or 4 and the **oscP5** library (*Sketch > Import Library... >
Manage Libraries... > oscP5*).

**Datagram size.** The 478-point mesh is a ~7 KB OSC message and oscP5's
default receive buffer is 1536 bytes: a truncated message is lost.  `setup()`
therefore creates `OscP5` with an `OscProperties` whose
`setDatagramSize(65536)` — keep that in your own sketch.

## Run

```
# terminal: the camera.  --face-landmarks adds the 478-point mesh
python ../../mediapipe_osc.py --face --face-landmarks --mirror
```

Open `face_landmarks.pde` in Processing and press **Run**, or from
`processing-examples/`: `make run-face_landmarks`.
Without `--face-landmarks` the sketch still works (box, nose, blendshapes, mask
without roll); the bottom line says whether the mesh arrives.
Without a camera: `python ../../osc_simulator.py` instead of the app.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/face/enter`, `/mp/face/leave` | `id` | |
| `/mp/face/center` | `id x y z` | the nose tip |
| `/mp/face/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/face/velocity` | `id vx vy speed` | |
| `/mp/face/blendshapes` | `id` + 52 × `value` | 0..1, in the order of `blendshape_names` |
| `/mp/face/blendshape_names` | 52 × `name` | sent once when the app starts |
| `/mp/face/landmarks` | `id` + 478 × `x y z` | only with `--face-landmarks` |

Useful mesh indices: 1 nose tip, 10 forehead, 152 chin, 33 / 263 outer eye
corners, 61 / 291 mouth corners, 13 / 14 upper / lower lip
([map](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)).

## Where to start changing it

* the "mask" in `draw()`: inside the same `pushMatrix()` / `translate()` /
  `rotate(roll)`, draw your own `PImage` with `imageMode(CENTER)` and
  `image(img, 0, 0, x1 - x0, y1 - y0)`.
* `MOUTH_OPEN` and the `/mp/face/blendshapes` branch — events from
  expressions: a smile (`mouthSmileLeft + mouthSmileRight`), a blink, raised brows...
* `Face.value(name)` gives any blendshape by name; `SHOWN_BLENDSHAPES` picks
  the bars.
