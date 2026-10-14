# TouchDesigner examples — OSC receivers for `mediapipe_osc.py`

_Castellano:_ cinco ejemplos de TouchDesigner (uno por detector: cuerpo, manos,
gestos, cara y blobs).  Como los proyectos de TouchDesigner (`.toe`) son
binarios, cada ejemplo es un script de Python que, ejecutado **dentro** de
TouchDesigner (arrastrar el `.py` a la red y *Run Script*, o una línea en el
Textport), construye la red completa: OSC In DAT con *callbacks* que llenan
tablas, Script CHOPs con los datos como canales, y un render que los dibuja.
`make test` prueba los scripts sin TouchDesigner con el simulador
`../osc_simulator.py`; `make run` abre TouchDesigner e imprime la línea a
pegar para cada ejemplo.

One build script per detector.  Run inside TouchDesigner, each script creates
a Base COMP (`mp_body`, `mp_hands`, `mp_gestures`, `mp_face`, `mp_blobs`) in
`/project1` with everything wired up:

```
mp_body/
  oscin1              OSC In DAT, port 5005, Active, Callbacks DAT = oscin1_callbacks
  oscin1_callbacks    Text DAT: Python that parses every /mp/... message into the tables
  people              Table DAT, one row per person (id x y z vx vy speed xmin ymin xmax ymax zone hand_up)
  landmarks           Table DAT, 33 rows per person (id index x y z visibility)
  people_chop         Script CHOP: the people table as channels, one sample per person (+ tx ty)
  landmarks_chop      Script CHOP: one sample per landmark (+ tx ty size) — drives the instancing
  geo_points          Geometry COMP: a Circle SOP instanced on landmarks_chop
  geo_skeleton        Geometry COMP with a Script SOP that draws the skeleton lines
  cam1 render1 background over1 out1     the picture
```

The tables are the readable, debuggable form of the data (open them in a
viewer while somebody moves in front of the camera); the Script CHOPs are what
you wire into the rest of your network (`Select CHOP`, expressions like
`op('mp_body/people_chop')['x'][0]`, instancing, `Trail`, `Lag` ...).  The
render is a visual check and a starting point: replace the Circle SOP, the
material or the whole geometry by your own.

| folder | detector | what it builds | start the camera side with |
|--------|----------|----------------|----------------------------|
| [`body_landmarks/`](body_landmarks/) | Body landmarks | tables `people`, `landmarks`; CHOPs with centre, velocity, zone, hand_up, 33 points; skeleton render | `python mediapipe_osc.py --pose` |
| [`hand_landmarks/`](hand_landmarks/) | Hand landmarks | tables `hands`, `landmarks`; CHOPs with left/right, pinch, pinching, 21 points; hand + pinch circle render | `python mediapipe_osc.py --hands` |
| [`hand_gestures/`](hand_gestures/) | Hand gestures | tables `hands`, `events` (gesture changes); CHOP with `gesture_index` and one 1/0 channel per gesture; a Text TOP with the current gesture on a colour per gesture | `python mediapipe_osc.py --gesture` |
| [`face_landmarks/`](face_landmarks/) | Face landmarks | tables `faces`, `blendshapes` (52 columns by name), `landmarks` (478 points); CHOPs; box + mesh + rotated "mask" render | `python mediapipe_osc.py --face --face-landmarks` |
| [`blobs/`](blobs/) | Blobs | tables `blobs` (with zone col/row and rank), `contours`; CHOPs incl. `zones_chop` (blobs per cell of a 3×3 grid); contours + grid render | `python mediapipe_osc.py --blobs` |

## Requirements

* [TouchDesigner](https://derivative.ca/download) 2021 or newer (Windows or
  macOS; the free non-commercial licence is enough).  Nothing else: the scripts
  only use operators that ship with TouchDesigner.
* For `make test` (optional, no TouchDesigner needed): Python 3.

## Running an example

1. Start the camera side in a terminal, e.g. `python mediapipe_osc.py --pose --mirror`
   (or `python osc_simulator.py` without a camera).
2. Open TouchDesigner (a new project is fine) and run the script of the example:
   * **drag & drop:** drag `body_landmarks/body_landmarks.py` from the file
     browser into the network editor — it becomes a Text DAT —, right-click
     the DAT and choose **Run Script**; or
   * **Textport:** open the Textport (`Alt+T`) and paste
     `exec(open(r'C:\path\to\body_landmarks.py').read())` (`make run-body_landmarks`
     prints this line with the right path, and opens TouchDesigner).
3. A `mp_body` component appears.  Enter it (`Enter` key or double-click):
   the tables fill up as people are detected, the Textport prints the events
   (person 3 entered, hand 1 pinch!, gesture changed, mouth open, blob changed
   zone ...), and `out1` / `render1` show the picture (turn the viewer on,
   or click the node and press `p`).

Run the script again whenever you want a fresh copy: it deletes `mp_body`
and builds it again.  Each script has a `PORT` constant at the top (5005).  If
TouchDesigner runs on another computer than the camera, start
`mediapipe_osc.py` with `--ip <that computer>` and let UDP 5005 through the
firewall.

`make all` = `make test` + `make run`; `make run` opens TouchDesigner (when
it is installed on this machine) and prints the Textport line for each of
the five examples, since TouchDesigner has no command-line way to run a script.

## Testing without TouchDesigner

`make test` runs every script in a small stand-in of TouchDesigner's Python
API (`../tests/stubs/touchdesigner/td_stub.py`: operators, parameters, Table
DATs, Script CHOP/SOP, an OSC In DAT that can be fed messages), feeds it
three seconds of `../osc_simulator.py`'s scene, cooks the Script CHOPs/SOPs
and fails if a callback raises or the tables stay empty.  It proves the
parsing and the data flow; the real test is TouchDesigner itself.  If
TouchDesigner refuses one of the parameters the scripts set (a renamed
parameter in a newer version, for instance), the script does not stop: it
prints a `WARNING ... set it by hand` line in the Textport for that
parameter and builds the rest.

## Building a network by hand

The scripts are also the documentation of what to build manually, if you
prefer to start from an empty network or to add the receiver to an existing
project:

1. an **OSC In DAT**: *Port* 5005, *Active* on;
2. a **Text DAT** with the text of `OSC_CALLBACKS` from the script, and the OSC
   In DAT's *Callbacks DAT* parameter pointing to it;
3. the **Table DATs** the callbacks write to, with the header rows listed at the
   top of each script (`people` / `landmarks`, `hands`, ...);
4. optionally a **Script CHOP** per table with the text of `PEOPLE_CHOP` etc.
   as its callbacks, and a Geometry COMP that instances on it.

Everything is addressed relative to the component that holds the operators
(`op('people')`), so the names matter but the location does not.

## The alternative: OSC In CHOP

For simple cases you do not need Python at all: start the app with
`--id-style address` and use an **OSC In CHOP** (port 5005).  Every address
becomes channels (`/mp/pose/3/center x y z` → three channels), so a single
person's centre or a hand's pinch can be read directly.  The Python route
above is there because ids come and go (the channel set keeps changing) and
because the landmark lists are long; tables and Script CHOPs keep that tidy.
