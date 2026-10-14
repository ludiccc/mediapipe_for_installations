# Face landmarks — openFrameworks

_Castellano:_ recibe por OSC (ofxOsc) las caras que manda
`mediapipe_osc.py --face --face-landmarks`: caja, punta de la nariz, malla de
478 puntos, 52 *blendshapes* (boca abierta, sonrisa, parpadeo...) y dibuja una
"máscara" orientada con la inclinación de la cabeza, donde una instalación
pondría su propia imagen.  `make && make RunRelease` en esta carpeta (dentro de
`<openFrameworks>/apps/openFrameworks-examples/`), o abrir el proyecto de
Xcode / Visual Studio generado con el projectGenerator.  Ojo con el tamaño del
buffer de ofxOsc para la malla (ver abajo).

Receives the **Face** detector (`/mp/face/...`) and draws, for every face: the
bounding box, the nose tip, the 478-point mesh and the face oval (when the
mesh arrives), the head roll computed from the outer eye corners, an ellipse
"mask" placed and rotated over the face — where an installation would draw its
own image, a bird's head for instance — and bars for the most expressive
blendshapes (`jawOpen`, `mouthSmileLeft/Right`, `eyeBlinkLeft/Right`,
`browInnerUp`, `mouthPucker`, `cheekPuff`).  Opening the mouth is logged as an event.

## Install

openFrameworks 0.10 or newer; ofxOsc ships with it.  This folder must be
inside `<openFrameworks>/apps/<something>/` or `OF_ROOT` must be set — see
`../README.md`.

**The mesh and the datagram size.** `/mp/face/landmarks` is a ~7 KB message.
The oscpack library bundled in ofxOsc reads datagrams into a fixed buffer —
`#define IP_MTU_SIZE 1536` in
`addons/ofxOsc/libs/oscpack/src/ip/posix/UdpSocket.cpp` (and
`ip/win32/UdpSocket.cpp` on Windows) in stock oscpack — and discards bigger
ones.  If the window says `mesh: no` while the app runs with
`--face-landmarks`, change that value to `65536` and rebuild (`make clean &&
make`; the addon is compiled into the project).  Everything else (box, nose,
blendshapes, mask) works without the mesh; only the roll needs it.

## Build and run

```
# terminal 1: the camera.  --face-landmarks adds the 478-point mesh
python mediapipe_osc.py --face --face-landmarks --mirror

# terminal 2, in this folder:
make && make RunRelease          # or from ../:  make run-face_landmarks
```

Xcode / Visual Studio: generate the project (`../generate_projects.sh`,
`..\generate_projects.bat` or the projectGenerator app), open it, run.
`q` or `Esc` quits.  Without a camera: `python osc_simulator.py`.

## Messages used

| address | arguments | |
|---------|-----------|-|
| `/mp/face/enter`, `/mp/face/leave` | `id` | |
| `/mp/face/center` | `id x y z` | the nose tip |
| `/mp/face/bbox` | `id xmin ymin xmax ymax` | |
| `/mp/face/velocity` | `id vx vy speed` | |
| `/mp/face/blendshapes` | `id` + 52 × `value` | 0..1, in the order of `blendshape_names` |
| `/mp/face/blendshape_names` | 52 × `name` | sent once when the app starts |
| `/mp/face/landmarks` | `id` + 478 × `x y z` | only with `--face-landmarks` |

Useful mesh indices: 1 nose tip, 10 forehead, 152 chin, 33 / 263 outer eye
corners, 61 / 291 mouth corners, 13 / 14 upper / lower lip
([map](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)).

## Where to start changing it

* the "mask" in `draw()`: inside the same `ofPushMatrix()` / `ofTranslate()` /
  `ofRotateDeg(roll)`, draw your own `ofImage` centred on 0,0 with the size of
  the box (`img.draw(-w / 2, -h / 2, w, h)`); put the image in `bin/data/`.
* `MOUTH_OPEN` and the `/mp/face/blendshapes` branch — events from
  expressions: a smile (`mouthSmileLeft + mouthSmileRight`), a blink, raised brows...
* `Face::value(names, name)` gives any blendshape by name; `SHOWN_BLENDSHAPES`
  picks the bars.
