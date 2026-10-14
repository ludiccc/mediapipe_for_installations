# Processing examples — OSC receivers for `mediapipe_osc.py`

_Castellano:_ cinco sketches de Processing (uno por detector: cuerpo, manos,
gestos, cara y blobs) que reciben con oscP5 lo que manda `mediapipe_osc.py` y
lo dibujan.  Cada carpeta es un sketch independiente con su README; se abren
con el IDE de Processing (Run) o desde la terminal con `make all` /
`make run-blobs`; `make test` los compila y prueba sin Processing, con el
simulador `../osc_simulator.py`.

One sketch per detector.  Every sketch is self-contained: it listens to the
OSC messages of `mediapipe_osc.py` with the **oscP5** library, keeps the
people / hands / faces / blobs it knows about (by their stable id) and draws
them.  Copy the folder of the detector you need and start your installation
from it.

| folder | detector | start the camera side with |
|--------|----------|----------------------------|
| [`body_landmarks/`](body_landmarks/) | Body landmarks (33-point pose) | `python mediapipe_osc.py --pose` |
| [`hand_landmarks/`](hand_landmarks/) | Hand landmarks (21 points, pinch, left/right) | `python mediapipe_osc.py --hands` |
| [`hand_gestures/`](hand_gestures/) | Hand gestures (Open_Palm, Thumb_Up, Victory ...) | `python mediapipe_osc.py --gesture` |
| [`face_landmarks/`](face_landmarks/) | Face landmarks (478-point mesh, 52 blendshapes) | `python mediapipe_osc.py --face --face-landmarks` |
| [`blobs/`](blobs/) | Blobs (shapes from a ceiling camera) | `python mediapipe_osc.py --blobs` |

## Requirements

* [Processing](https://processing.org/download) 3 or 4 (Java mode).
* The **oscP5** library: in the Processing IDE, *Sketch > Import Library... >
  Manage Libraries...*, search `oscP5`, Install.  (It installs `netP5` with it.)
* For the command line (`make`): `processing-java`.
  * Linux: it is in the Processing folder (`processing-4.x/processing-java`);
    add that folder to the `PATH` or use `make PROCESSING_JAVA=/path/to/processing-java`.
  * macOS: in the IDE, *Tools > Install "processing-java"* (puts it in `/usr/local/bin`).
  * Windows: `processing-java.exe` is next to `processing.exe`; add that folder
    to the `PATH` (and run `make` from Git Bash or MSYS2, or just use the IDE).

## Running

Two programs run at the same time: the camera side and the sketch.

```
# terminal — the camera side (any detector; --mirror makes it behave like a mirror)
cd ..
python mediapipe_osc.py --pose --mirror
```

then open `body_landmarks/body_landmarks.pde` in the Processing IDE and press
**Run**, or from this folder:

```
make all                   # build the five sketches, then run them one after the other
make run-body_landmarks    # run one (run-hand_landmarks, run-hand_gestures, run-face_landmarks, run-blobs)
make build                 # only compile (into build/)
```

Close a sketch window to stop it (with `make all`, the next one starts).
The port is the `PORT` variable at the top of each sketch (5005).  If the
receiver runs on another computer, start the app with `--ip <that computer>`
and let UDP port 5005 through its firewall.

## Testing without a camera (or without Processing)

`../osc_simulator.py` sends made-up but well-formed data for all detectors.
Run it in a terminal (`make sim`, or `python ../osc_simulator.py`) and start
any sketch from the IDE.

`make test` goes further: it compiles every sketch **without Processing**
against small stand-in classes (`../tests/stubs/processing`: a `PApplet` whose
drawing calls do nothing and an `oscP5` that really listens on UDP), runs it
headless for three seconds while the simulator sends data, and fails if a
sketch does not compile, receives nothing or throws in `oscEvent`.  It needs a
JDK (`javac`) and Python 3.  It checks the OSC logic, not the drawing — the
real test is still the Processing IDE.

## How the sketches are built

All five have the same shape:

1. a class (`Person`, `Hand`, `Face`, `Blob`) with what we know about one
   tracked thing, in a `ConcurrentHashMap<Integer, ...>` keyed by id.  OSC
   events arrive on oscP5's thread while `draw()` runs on Processing's, and a
   `ConcurrentHashMap` can be iterated while it is modified (a `HashMap` would
   throw `ConcurrentModificationException`);
2. `oscEvent(OscMessage m)`: one `if` per address, reading `m.get(i).intValue()`,
   `.floatValue()`, `.stringValue()` — the first argument is always the id;
3. `draw()`: forgets things whose data stopped arriving, then draws the state.
   Coordinates are normalized (0..1), multiply by `width` / `height`.

`setup()` creates `OscP5` through an `OscProperties` object so the datagram
buffer is bigger than oscP5's default 1536 bytes: the app's face mesh message
is ~7 KB, and a truncated packet is lost.

`enter` / `leave` messages create and delete objects; data for an unknown id
creates it too, so a sketch started after the app still works.  The sketches
`println` the events an installation usually reacts to (somebody entered, a
pinch closed, a gesture changed, a mouth opened, a blob changed zone).
