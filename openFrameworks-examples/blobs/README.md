# Blobs — openFrameworks

_Castellano:_ recibe por OSC (ofxOsc) los *blobs* (formas que se destacan del
fondo, típicamente personas vistas desde una cámara en el techo) que manda
`mediapipe_osc.py --blobs`: contorno, caja, centro, velocidad, área y ángulo de
cada uno, y divide el piso en una grilla 3×3 para saber en qué zona está cada
visitante.  `make && make RunRelease` en esta carpeta (dentro de
`<openFrameworks>/apps/openFrameworks-examples/`), o abrir el proyecto de
Xcode / Visual Studio generado con el projectGenerator.

Receives the **Blobs** detector (`/mp/blob/...`) and draws every blob: its
16-point outline, bounding box, centroid, velocity arrow, main axis, area (as
% of the picture), orientation and id, plus its rank by size (from
`/mp/blobs`).  The picture is divided into a 3 × 3 grid of zones: the cells
that contain a blob light up and a change of zone is logged as an event — the
usual "something happens where the visitor stands" logic.

## Install

openFrameworks 0.10 or newer; ofxOsc ships with it.  This folder must be
inside `<openFrameworks>/apps/<something>/` or `OF_ROOT` must be set — see
`../README.md`.

## Build and run

```
# terminal 1: the camera (ceiling camera; start it with the scene EMPTY: the first
# frames are learned as the background.  Press b in its window to learn again.)
python mediapipe_osc.py --blobs

# terminal 2, in this folder:
make && make RunRelease          # or from ../:  make run-blobs
```

Xcode / Visual Studio: generate the project (`../generate_projects.sh`,
`..\generate_projects.bat` or the projectGenerator app), open it, run.
Tuning on the camera side: `--blob-threshold`, `--blob-min-area`,
`--blob-mode motion|bright|dark` — see `README_mediapipe_osc.md`.
`q` or `Esc` quits.  Without a camera: `python osc_simulator.py`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/blobs` | `count id id ...` | every frame, biggest first |
| `/mp/blob/enter`, `/mp/blob/leave` | `id` | |
| `/mp/blob/center` | `id x y 0` | centroid |
| `/mp/blob/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/blob/velocity` | `id vx vy speed` | normalized units per second |
| `/mp/blob/info` | `id area angle length width` | area = fraction of the image; angle −90..90° |
| `/mp/blob/contour` | `id` + 16 × `x y` | outline, from the topmost point |

## Where to start changing it

* `GRID`, `Blob::col()` / `Blob::row()` (`ofApp.h`) — the zones.  Irregular
  zones: test the centroid against your own `ofPolyline::inside()`.
* the `/mp/blob/center` branch of `handleMessage()` — the zone-change event;
  `enter` / `leave` for arrivals.
* `area` tells big from small (a person from a bag), `speed` still from
  moving, `angle` which way an elongated shape points.
* the `order` vector (`/mp/blobs`) gives "the biggest one" without comparing.
