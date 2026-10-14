# Blobs — TouchDesigner

_Castellano:_ `blobs.py`, ejecutado dentro de TouchDesigner (arrastrar el
archivo a la red y *Run Script*), construye `mp_blobs`: un OSC In DAT que
recibe los *blobs* de `mediapipe_osc.py --blobs` (formas vistas desde una
cámara en el techo), una tabla con cada blob (centro, caja, área, ángulo, zona
de una grilla 3×3, orden por tamaño), una tabla con los contornos, Script
CHOPs — uno con cuántos blobs hay en cada zona — y un render con los contornos
y la grilla.  En una terminal: `python mediapipe_osc.py --blobs` (con la
escena vacía al arrancar).

Builds, inside TouchDesigner, a `mp_blobs` component that receives the
**Blobs** detector (`/mp/blob/...`):

| operator | what it holds |
|----------|---------------|
| `blobs` (Table DAT) | one row per blob: `id x y vx vy speed xmin ymin xmax ymax area angle length width col row rank` — `col`, `row` = the cell of a 3 × 3 grid, `rank` = 1 for the biggest blob (from `/mp/blobs`) |
| `contours` (Table DAT) | 16 rows per blob: `id index x y` |
| `blobs_chop` (Script CHOP) | the table as channels, one sample per blob, plus `count`, `tx`, `ty` |
| `zones_chop` (Script CHOP) | 9 channels `zone_<col>_<row>` = how many blobs stand in each cell — the "something happens where the visitor stands" signal |
| `geo_shapes`, `geo_centers`, `render1`, `out1` | the contour, bounding box and main axis of every blob, the grid, and a circle on every centroid |

The Textport prints `blob N entered` / `left` / `is now in zone column c, row r`.

## Install

TouchDesigner 2021 or newer.  Nothing to add.

## Run

```
# terminal: the camera (ceiling camera; start it with the scene EMPTY: the first
# frames are learned as the background.  Press b in its window to learn again.)
python mediapipe_osc.py --blobs                # or: python osc_simulator.py
```

In TouchDesigner: drag `blobs.py` into the network, right-click the Text DAT
> **Run Script** — or paste in the Textport:
`exec(open(r'<path>/blobs.py').read())`.  Then enter `mp_blobs`.
Tuning on the camera side: `--blob-threshold`, `--blob-min-area`,
`--blob-mode motion|bright|dark` — see `README_mediapipe_osc.md`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/blobs` | `count id id ...` | biggest first → `rank` |
| `/mp/blob/enter`, `/mp/blob/leave` | `id` | |
| `/mp/blob/center` | `id x y 0` | → `x y col row` (zone change printed) |
| `/mp/blob/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/blob/velocity` | `id vx vy speed` | |
| `/mp/blob/info` | `id area angle length width` | area = fraction of the image; angle −90..90° |
| `/mp/blob/contour` | `id` + 16 × `x y` | → `contours` |

## Where to start changing it

* `zones_chop` → a Logic / Count CHOP, or a CHOP Execute DAT, to trigger a
  sound or a scene per zone; change `GRID` (in the callbacks, `zones_chop`
  and `shapes`) for another grid.
* `rank == 1` in `blobs_chop` picks the biggest blob; `speed` still from
  moving; `area` big from small.
* `geo_shapes/shapes` — the Script SOP that draws the contours: use its
  polygons as masks, or extrude them.
