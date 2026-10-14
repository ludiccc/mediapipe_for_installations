# Face landmarks — TouchDesigner

_Castellano:_ `face_landmarks.py`, ejecutado dentro de TouchDesigner (arrastrar
el archivo a la red y *Run Script*), construye `mp_face`: un OSC In DAT que
recibe las caras de `mediapipe_osc.py --face --face-landmarks` (caja, nariz,
52 *blendshapes* por nombre, malla de 478 puntos), tablas, Script CHOPs y un
render con la caja, la malla y una "máscara" girada con la inclinación de la
cabeza.  En una terminal: `python mediapipe_osc.py --face --face-landmarks`.

Builds, inside TouchDesigner, a `mp_face` component that receives the
**Face** detector (`/mp/face/...`):

| operator | what it holds |
|----------|---------------|
| `faces` (Table DAT) | one row per face: `id x y z vx vy speed xmin ymin xmax ymax roll mouth_open` — `roll` = head tilt in degrees from the outer eye corners (needs the mesh), `mouth_open` from `jawOpen` |
| `blendshapes` (Table DAT) | one row per face, one column per blendshape **by name** (`jawOpen`, `eyeBlinkLeft`, `mouthSmileRight` ... the header is taken from `/mp/face/blendshape_names`) |
| `landmarks` (Table DAT) | 478 rows per face: `id index x y z` (only with `--face-landmarks`) |
| `faces_chop` (Script CHOP) | one sample per face: the `faces` columns, `tx ty`, and every blendshape as a channel |
| `landmarks_chop` (Script CHOP) | one sample per mesh point: `id index x y z tx ty` |
| `geo_points`, `geo_lines`, `render1`, `out1` | a dot per mesh point; the bounding box, the face oval, the eye line and an ellipse "mask" rotated with the head — where your own image would go |

The Textport prints `face N entered` / `left` / `mouth open` / `mouth closed`.

## Install

TouchDesigner 2021 or newer.  Nothing to add.  TouchDesigner receives the
7 KB mesh message without any setting to change.

## Run

```
# terminal: the camera.  --face-landmarks adds the 478-point mesh
python mediapipe_osc.py --face --face-landmarks --mirror     # or: python osc_simulator.py
```

In TouchDesigner: drag `face_landmarks.py` into the network, right-click the
Text DAT > **Run Script** — or paste in the Textport:
`exec(open(r'<path>/face_landmarks.py').read())`.  Then enter `mp_face`.
Without `--face-landmarks` everything but the mesh, the oval and the roll works.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/face/enter`, `/mp/face/leave` | `id` | |
| `/mp/face/center` | `id x y z` | the nose tip |
| `/mp/face/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/face/velocity` | `id vx vy speed` | |
| `/mp/face/blendshapes` | `id` + 52 × `value` | → the `blendshapes` table, `mouth_open` |
| `/mp/face/blendshape_names` | 52 × `name` | → the header of `blendshapes` |
| `/mp/face/landmarks` | `id` + 478 × `x y z` | → `landmarks`, `roll` |

Useful mesh indices: 1 nose tip, 10 forehead, 152 chin, 33 / 263 outer eye
corners, 61 / 291 mouth corners, 13 / 14 upper / lower lip
([map](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)).

## Where to start changing it

* the mask: in `geo_lines` replace the ellipse by your own geometry, or
  instance a Geometry COMP with a textured rectangle on `faces_chop` using
  `tx`, `ty` for the position, `xmax - xmin` for the size and `roll` for the
  rotation (Instance *Rotate Z* = `roll`, negated: y is up in the render).
* expressions: `op('mp_face/faces_chop')['jawOpen'][0]`,
  `['mouthSmileLeft']`, `['eyeBlinkRight']` ... drive anything from an expression.
* `MOUTH_OPEN` and the `/mp/face/blendshapes` branch of the callbacks — the events.
