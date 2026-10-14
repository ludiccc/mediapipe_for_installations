# openFrameworks examples — OSC receivers for `mediapipe_osc.py`

_Castellano:_ cinco apps de openFrameworks (una por detector: cuerpo, manos,
gestos, cara y blobs) que reciben con ofxOsc lo que manda `mediapipe_osc.py` y
lo dibujan.  Cada carpeta es un proyecto independiente con su README.  Copiar
esta carpeta dentro de `<openFrameworks>/apps/`; en Linux y macOS `make all`
compila y ejecuta los cinco uno tras otro; en Windows (y macOS) `make projects`
o `generate_projects.bat` generan las soluciones de Visual Studio / Xcode con el
projectGenerator.  `make test` los compila y prueba sin openFrameworks, con el
simulador `../osc_simulator.py`.

One app per detector.  Each one is a standard openFrameworks project
(`src/`, `Makefile`, `config.make`, `addons.make`, `bin/data/`) that listens to
the OSC messages of `mediapipe_osc.py` with the **ofxOsc** addon, keeps the
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

* [openFrameworks](https://openframeworks.cc/download/) **0.10 or newer**
  (0.11 / 0.12 recommended) for your platform, set up as its own instructions
  say (on Linux: `scripts/linux/<distro>/install_dependencies.sh` and
  `scripts/linux/compileOF.sh`; on macOS: Xcode or its command line tools; on
  Windows: Visual Studio 2019/2022 with the C++ desktop workload, or the MSYS2
  release).
* The **ofxOsc** addon — it ships with openFrameworks (`addons/ofxOsc`), nothing
  to install.  Each example lists it in `addons.make`.

## Where to put the examples

The project Makefiles (like every openFrameworks project) expect to live two
folders below the openFrameworks root, so the simplest is:

```
<openFrameworks>/
  apps/
    openFrameworks-examples/        <- this folder, copied or symlinked here
      body_landmarks/
      hand_landmarks/
      ...
```

Anywhere else: `make OF_ROOT=/path/to/openFrameworks ...`, or uncomment and
set `OF_ROOT` in the `config.make` of each example.

## Building and running

Two programs run at the same time: the camera side and the app.

```
# terminal 1 — the camera side (any detector; --mirror makes it behave like a mirror)
python mediapipe_osc.py --pose --mirror
```

### Linux (and macOS with the command line tools)

```
make all                  # compile the five examples (Release), then run them one after the other
make run-body_landmarks   # compile + run one (run-hand_landmarks, run-hand_gestures, run-face_landmarks, run-blobs)
make build                # only compile
make clean
```

Inside one example folder the usual openFrameworks targets work too:
`make` (Release), `make Debug`, `make RunRelease`, `make RunDebug`, `make clean`.
The binary is `bin/<example>` (`bin/<example>.app` on macOS).  Close the
window or press `q` / `Esc` to quit (with `make all`, the next example starts).

### macOS with Xcode

```
make projects             # = ./generate_projects.sh osx  -> <example>/<example>.xcodeproj
```

then open the `.xcodeproj` of an example and press Run.  `make projects`
uses the command-line projectGenerator that comes with openFrameworks
(inside `projectGenerator.app`); if it is not found, open the
**projectGenerator** app, press *import*, choose the example folder (ofxOsc
is picked up from `addons.make`) and press *Generate*.

### Windows

* **Visual Studio:** `generate_projects.bat` (from a command prompt in this
  folder, or `generate_projects.bat C:\path\to\openFrameworks`) writes
  `<example>\<example>.sln` for every example with the projectGenerator; then
  open a solution, choose *Release | x64* and run.  Without the command-line
  tool, use the projectGenerator app (*import* an example folder, *Generate*).
* **MSYS2 release of openFrameworks:** from the MSYS2 MinGW 64-bit shell,
  `make all` / `make run-blobs` work like on Linux.

### Another port or computer

The port is `PORT` at the top of each `src/ofApp.h` (5005).  If the app runs
on another computer than the camera, start `mediapipe_osc.py` with
`--ip <that computer>` and let UDP port 5005 through its firewall.

## Testing without a camera (or without openFrameworks)

`../osc_simulator.py` sends made-up but well-formed data for all detectors.
Run it in a terminal (`make sim`, or `python ../osc_simulator.py`) and start
any example.

`make test` compiles every app **without openFrameworks** against small
stand-in headers (`../tests/stubs/openframeworks`: an `ofMain.h` whose drawing
calls do nothing and an `ofxOsc.h` that really listens on UDP), runs it
headless for three seconds while the simulator sends data, and fails if an
app does not compile or receives nothing.  It needs `g++` and Python 3 and
checks the OSC logic, not the drawing — the real test is still openFrameworks.

## A note on big messages (the face mesh)

The `/mp/face/landmarks` message (478 points) is about 7 KB.  ofxOsc's
bundled **oscpack** receives datagrams into a fixed buffer
(`IP_MTU_SIZE` in `addons/ofxOsc/libs/oscpack/src/ip/posix/UdpSocket.cpp` and
`ip/win32/UdpSocket.cpp`; 1536 bytes in stock oscpack).  A datagram bigger
than that buffer is discarded; every other message still arrives, so the
face example works without the mesh (its bottom line says `mesh: no`).  To
receive the mesh, raise that `#define` to `65536` and rebuild the example
(`make clean && make` — the addon is compiled into each project).  The
other detectors' messages are all well under 1 KB.

## How the apps are built

All five have the same shape:

1. `ofApp.h`: a struct (`Person`, `Hand`, `Face`, `Blob`) with what we know
   about one tracked thing, and a `std::map<int, ...>` keyed by id;
2. `update()`: `while (receiver.hasWaitingMessages())` read every message
   and pass it to `handleMessage()`, one `if` per address
   (`m.getArgAsInt32(0)` is always the id, then `getArgAsFloat` /
   `getArgAsString`); then forget things whose data stopped arriving;
3. `draw()`: the state, coordinates normalized 0..1 multiplied by
   `ofGetWidth()` / `ofGetHeight()`.

`enter` / `leave` messages create and delete objects; data for an unknown id
creates it too, so an app started after the camera side still works.  The
apps `ofLogNotice()` the events an installation usually reacts to (somebody
entered, a pinch closed, a gesture changed, a mouth opened, a blob changed
zone).  ofxOsc delivers messages on the main thread (in `update()`), so no
locking is needed.
