# Python examples — OSC receivers for `mediapipe_osc.py`

_Castellano:_ cinco receptores OSC en Python (uno por detector: cuerpo, manos,
gestos, cara y blobs) que dibujan con OpenCV lo que manda `mediapipe_osc.py`.
Cada carpeta es un ejemplo independiente con su propio README; `make all`
comprueba los módulos y los ejecuta uno tras otro; `make test` los prueba sin
cámara usando el simulador `../osc_simulator.py`.

One folder per detector; every example is a single, self-contained script that
listens to the OSC messages of `mediapipe_osc.py`, keeps the people / hands /
faces / blobs it knows about (by their stable id) and draws them in a window.
Copy the folder of the detector you need and start your installation from it.

| folder | detector | start the camera side with |
|--------|----------|----------------------------|
| [`body_landmarks/`](body_landmarks/) | Body landmarks (33-point pose) | `python mediapipe_osc.py --pose` |
| [`hand_landmarks/`](hand_landmarks/) | Hand landmarks (21 points, pinch, left/right) | `python mediapipe_osc.py --hands` |
| [`hand_gestures/`](hand_gestures/) | Hand gestures (Open_Palm, Thumb_Up, Victory ...) | `python mediapipe_osc.py --gesture` |
| [`face_landmarks/`](face_landmarks/) | Face landmarks (478-point mesh, 52 blendshapes) | `python mediapipe_osc.py --face --face-landmarks` |
| [`blobs/`](blobs/) | Blobs (shapes from a ceiling camera) | `python mediapipe_osc.py --blobs` |

## Requirements

Python 3.9 or newer and three modules:

```
pip install -r requirements.txt          # python-osc, opencv-python, numpy
```

(`make deps` does the same.) They are the modules `mediapipe_osc.py` itself
needs, so if the app runs on the computer, the examples run too.  On some Linux
distributions `pip` refuses to install system-wide: use a virtual environment
(`python3 -m venv venv && source venv/bin/activate`) or add `--break-system-packages`.

## Running

Two programs run at the same time: the camera side and the receiver.

```
# terminal 1 — the camera side (any detector, --mirror makes it behave like a mirror)
cd ..
python mediapipe_osc.py --pose --mirror

# terminal 2 — the receiver
cd python-examples
python body_landmarks/body_landmarks.py
```

or, with make:

```
make all                 # check modules, then run the five examples one after the other
make run-body_landmarks  # run one (run-hand_landmarks, run-hand_gestures, run-face_landmarks, run-blobs)
make PORT=9000 run-blobs # listen on another port (the app must send there: --port 9000)
```

Press `q` or `Esc` in a window to close it (with `make all`, the next example
starts).  On Windows use `make PYTHON=python ...` or simply run the scripts with
`python`.  All scripts accept `--port`, `--ip`, `--headless` and `--seconds`
(`--help` lists them).

The receiver can run on another computer: start the app with
`--ip <that computer>` and make sure its firewall lets UDP port 5005 in.

## Testing without a camera

`../osc_simulator.py` sends made-up but well-formed data for all detectors
(two people who walk around, one leaving and coming back; two hands with
changing gestures; two faces with blinking and talking blendshapes and the
full mesh; three blobs).

```
make test                # headless: simulator + every example for 3 s, prints what arrived
make sim                 # or: run the simulator and start any example in another terminal
```

`make test` fails if an example receives nothing, so it is a quick way to
check that the modules and the port are fine.

## How the examples are built

All five scripts have the same shape, so once you read one you know them all:

1. a small class (`Person`, `Hand`, `Face`, `Blob`) with what we know about
   one tracked thing, and a dictionary `id -> object`;
2. one handler per OSC address (`on_center`, `on_landmarks`, ...) registered in
   `make_dispatcher()`.  They run on the server thread, so the shared state is
   protected by a lock;
3. `draw()` renders the current state into a numpy image with OpenCV;
4. `main()` starts the OSC server in a background thread, then loops: forget
   things whose data stopped arriving, draw, show, handle the keyboard.

`enter` / `leave` messages create and delete the objects; data arriving for an
unknown id creates it too, so a receiver started after the app still works.
The examples also print a line on the events an installation usually reacts to
(somebody entered, a pinch closed, a gesture changed, a mouth opened, a blob
changed zone): look for `print(` to find them.
