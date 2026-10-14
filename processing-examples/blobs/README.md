# Blobs — Processing

_Castellano:_ recibe por OSC (oscP5) los *blobs* (formas que se destacan del
fondo, típicamente personas vistas desde una cámara en el techo) que manda
`mediapipe_osc.py --blobs`: contorno, caja, centro, velocidad, área y ángulo de
cada uno, y divide el piso en una grilla 3×3 para saber en qué zona está cada
visitante.  Abrir `blobs.pde` en Processing y apretar Run (y en una terminal
`python mediapipe_osc.py --blobs`, con la escena vacía al arrancar).

Receives the **Blobs** detector (`/mp/blob/...`) and draws every blob: its
16-point outline, bounding box, centroid, velocity arrow, main axis, area (as
% of the picture), orientation and id, plus its rank by size (from
`/mp/blobs`).  The picture is divided into a 3 × 3 grid of zones: the cells
that contain a blob light up and a change of zone is printed as an event — the
usual "something happens where the visitor stands" logic.

## Install

Processing 3 or 4 and the **oscP5** library (*Sketch > Import Library... >
Manage Libraries... > oscP5*).

## Run

```
# terminal: the camera (ceiling camera; start it with the scene EMPTY: the first
# frames are learned as the background.  Press b in its window to learn again.)
python ../../mediapipe_osc.py --blobs
```

Open `blobs.pde` in Processing and press **Run**, or from
`processing-examples/`: `make run-blobs`.  Tuning on the camera side:
`--blob-threshold`, `--blob-min-area`, `--blob-mode motion|bright|dark` — see
`../../README_mediapipe_osc.md`.  Without a camera: `python ../../osc_simulator.py`.

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

* `GRID`, `Blob.col()` / `Blob.row()` — the zones.  Irregular zones: test the
  centroid against your own shapes.
* the `/mp/blob/center` branch of `oscEvent()` — the zone-change event;
  `enter` / `leave` for arrivals.
* `area` tells big from small (a person from a bag), `speed` still from
  moving, `angle` which way an elongated shape points.
* the `order` array (`/mp/blobs`) gives "the biggest one" without comparing.
