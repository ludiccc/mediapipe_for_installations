# Hand landmarks — TouchDesigner

_Castellano:_ `hand_landmarks.py`, ejecutado dentro de TouchDesigner (arrastrar
el archivo a la red y *Run Script*), construye `mp_hands`: un OSC In DAT que
recibe las manos de `mediapipe_osc.py --hands` (21 puntos, izquierda/derecha,
pinza), tablas, Script CHOPs y un render con el esqueleto de cada mano y el
círculo de la pinza.  En una terminal: `python mediapipe_osc.py --hands`.

Builds, inside TouchDesigner, a `mp_hands` component that receives the
**Hands** detector (`/mp/hand/...`):

| operator | what it holds |
|----------|---------------|
| `hands` (Table DAT) | one row per hand: `id handedness score x y z vx vy speed xmin ymin xmax ymax pinch pinching` — `pinching` = 1 while thumb and index touch |
| `landmarks` (Table DAT) | 21 rows per hand: `id index x y z` |
| `hands_chop` (Script CHOP) | the table as channels, one sample per hand (`handedness` becomes `right` = 1/0), plus `count`, `tx`, `ty` |
| `landmarks_chop` (Script CHOP) | one sample per landmark: `id index x y z tx ty size` (fingertips bigger) |
| `geo_points`, `geo_skeleton`, `render1`, `out1` | a circle per landmark, the hand connections and a circle between thumb and index whose radius is the pinch |

The Textport prints `hand N entered` / `left` / `pinch!` / `released`.

## Install

TouchDesigner 2021 or newer.  Nothing to add.

## Run

```
# terminal: the camera
python mediapipe_osc.py --hands --mirror       # or: python osc_simulator.py
```

In TouchDesigner: drag `hand_landmarks.py` into the network, right-click the
Text DAT > **Run Script** — or paste in the Textport:
`exec(open(r'<path>/hand_landmarks.py').read())`.  Then enter `mp_hands`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/hand/enter`, `/mp/hand/leave` | `id` | |
| `/mp/hand/info` | `id handedness score` | `"Left"` / `"Right"` |
| `/mp/hand/center` | `id x y z` | |
| `/mp/hand/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/hand/velocity` | `id vx vy speed` | |
| `/mp/hand/pinch` | `id distance` | → `pinch`, `pinching` (threshold `PINCH_CLOSED` in the callbacks) |
| `/mp/hand/landmarks` | `id` + 21 × `x y z` | 0 wrist, 4 thumb tip, 8 index tip, 12, 16, 20 other tips |

## Where to start changing it

* `pinching` in `hands_chop` is a ready-made button; `pinch` a continuous
  control (a Lag CHOP smooths it).
* the index tip as a cursor: a Select CHOP on `landmarks_chop` with
  `index == 8` (or read `op('mp_hands/landmarks')` in a Script CHOP).
* `PINCH_CLOSED` and the `/mp/hand/pinch` branch of `OSC_CALLBACKS` — the event.
