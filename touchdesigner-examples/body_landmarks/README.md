# Body landmarks — TouchDesigner

_Castellano:_ `body_landmarks.py`, ejecutado dentro de TouchDesigner (arrastrar
el archivo a la red y *Run Script*), construye `mp_body`: un OSC In DAT que
recibe los 33 puntos del cuerpo de `mediapipe_osc.py --pose`, tablas con una
fila por persona y una por punto, Script CHOPs con esos datos como canales y un
render con el esqueleto.  En una terminal: `python mediapipe_osc.py --pose`.

Builds, inside TouchDesigner, a `mp_body` component that receives the **Body**
detector (`/mp/pose/...`):

| operator | what it holds |
|----------|---------------|
| `people` (Table DAT) | one row per person: `id x y z vx vy speed xmin ymin xmax ymax zone hand_up` — `zone` is the third of the picture (0 left, 1 centre, 2 right), `hand_up` = 1 when a wrist is above the shoulders |
| `landmarks` (Table DAT) | 33 rows per person: `id index x y z visibility` |
| `people_chop` (Script CHOP) | the `people` table as channels, one sample per person, plus `count`, `tx`, `ty` (render units) |
| `landmarks_chop` (Script CHOP) | one sample per landmark: `id index x y z visibility tx ty size` |
| `geo_points`, `geo_skeleton`, `render1`, `out1` | a circle per landmark (instancing) and the skeleton lines (Script SOP) |

The Textport prints `person N entered` / `left`.

## Install

TouchDesigner 2021 or newer.  Nothing to add.

## Run

```
# terminal: the camera
python mediapipe_osc.py --pose --mirror        # or: python osc_simulator.py
```

In TouchDesigner: drag `body_landmarks.py` into the network, right-click the
Text DAT > **Run Script** — or paste in the Textport (`Alt+T`):
`exec(open(r'<path>/body_landmarks.py').read())` (`make run-body_landmarks`
prints it with the path).  Then enter `mp_body` and look at `out1`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/pose/enter`, `/mp/pose/leave` | `id` | rows appear / disappear |
| `/mp/pose/center` | `id x y z` | → `x y z zone` |
| `/mp/pose/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/pose/velocity` | `id vx vy speed` | |
| `/mp/pose/landmarks` | `id` + 33 × `x y z visibility` | → `landmarks`, `hand_up` |

`x`, `y` are 0..1 (origin top-left); `tx`, `ty` are the same in the render's
orthographic camera (x −aspect..aspect, y −1..1, y up).  Landmark numbering:
0 nose, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees,
27/28 ankles ([full list](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)).

## Where to start changing it

* use the channels: `op('mp_body/people_chop')['x'][0]` in any expression,
  or a Select CHOP on `people_chop`; `zone` and `hand_up` are ready-made
  triggers (a Count CHOP or a CHOP Execute DAT on them).
* `OSC_CALLBACKS` in the script — the parsing; add columns (e.g. the distance
  between the wrists) in `onReceiveOSC`.
* `geo_points` — replace `circle1` by any SOP, or instance your own Geometry
  COMP on `landmarks_chop` (`tx`, `ty`, `size`).
