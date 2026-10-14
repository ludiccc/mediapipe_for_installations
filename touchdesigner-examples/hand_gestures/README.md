# Hand gestures — TouchDesigner

_Castellano:_ `hand_gestures.py`, ejecutado dentro de TouchDesigner (arrastrar
el archivo a la red y *Run Script*), construye `mp_gestures`: un OSC In DAT que
recibe los gestos de `mediapipe_osc.py --gesture`, una tabla con el gesto de
cada mano, una tabla de *eventos* (cada cambio de gesto), un Script CHOP con
el gesto como número y como canales 1/0, y un Text TOP que muestra el gesto
actual sobre un color por gesto.  En una terminal:
`python mediapipe_osc.py --gesture`.

Builds, inside TouchDesigner, a `mp_gestures` component that receives the
**Gestures** detector (`/mp/hand/gesture` and the hand messages):

| operator | what it holds |
|----------|---------------|
| `hands` (Table DAT) | one row per hand: `id handedness gesture score x y xmin ymin xmax ymax` |
| `events` (Table DAT) | the last 20 gesture **changes**: `time id handedness gesture score` — the events an installation reacts to |
| `hands_chop` (Script CHOP) | one sample per hand: `gesture_index` (0 None, 1 Closed_Fist, 2 Open_Palm, 3 Pointing_Up, 4 Thumb_Down, 5 Thumb_Up, 6 Victory, 7 ILoveYou), one 1/0 channel per gesture (`open_palm`, `thumb_up`, ...), `score`, `right`, `x y tx ty` |
| `gesture_text` (Text TOP), `out1` | the gesture of the strongest hand, on a background colour per gesture (set by the callbacks every frame) |

The Textport prints every change: `hand 0 (Right): Open_Palm (0.87)`.

## Install

TouchDesigner 2021 or newer.  Nothing to add.

## Run

```
# terminal: the camera  (--gesture implies --hands)
python mediapipe_osc.py --gesture --mirror     # or: python osc_simulator.py
```

In TouchDesigner: drag `hand_gestures.py` into the network, right-click the
Text DAT > **Run Script** — or paste in the Textport:
`exec(open(r'<path>/hand_gestures.py').read())`.  Then enter `mp_gestures`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/hand/gesture` | `id name score` | every frame; a change → a row in `events` |
| `/mp/hand/enter`, `/mp/hand/leave` | `id` | |
| `/mp/hand/info` | `id handedness score` | |
| `/mp/hand/center`, `/mp/hand/bbox` | `id x y z`, `id xmin ymin xmax ymax` | |

## Where to start changing it

* a **DAT Execute DAT** on `events` (*Row Change* / *Table Change*) runs
  your Python when a gesture changes — the cleanest way to trigger scenes.
* `thumb_up`, `open_palm` ... in `hands_chop` are 1/0 channels: a CHOP
  Execute DAT (*Value Change*) or a Logic CHOP turns them into triggers.
* `MIN_SCORE` in the callbacks filters doubtful classifications; `COLORS`
  maps gestures to the background colour of `gesture_text`.
