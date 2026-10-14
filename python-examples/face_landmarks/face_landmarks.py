#!/usr/bin/env python3
"""Face landmarks receiver — Python + python-osc + OpenCV.

Listens to the OSC messages of mediapipe_osc.py for the **Face** detector and,
for every face: draws the bounding box and the nose tip, the 478-point mesh
(when the app sends it), the head roll computed from the eyes, a "mask"
placed over the face (where an installation would put its own image — a bird's
head, for instance), and bars for the most expressive blendshapes (jaw open,
smile, blinks, brows).  Opening the mouth is turned into an event.

    python ../../mediapipe_osc.py --face --face-landmarks --mirror   # terminal 1 (the camera)
    python face_landmarks.py                                         # terminal 2 (this)

    python face_landmarks.py --port 9000
    python face_landmarks.py --headless --seconds 5                  # no window (tests)

Messages used (first argument = stable id of the face):

    /mp/face/enter            id
    /mp/face/leave            id
    /mp/face/center           id x y z                 the nose tip
    /mp/face/bbox             id xmin ymin xmax ymax
    /mp/face/velocity         id vx vy speed
    /mp/face/blendshapes      id 52 x value            0..1; order = /mp/face/blendshape_names
    /mp/face/blendshape_names 52 x name                sent once when the app starts
    /mp/face/landmarks        id 478 x (x y z)         only with --face-landmarks (a 7 KB message)

Landmark indices: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
"""

# --- modules -------------------------------------------------------------------
# argparse   : command-line options (--port, --headless ...)
# threading  : the OSC server runs in its own thread, next to the drawing loop
# cv2        : OpenCV, used only to draw and to show a window
# numpy      : arrays; the landmarks of one hand/face/blob live in one array
# pythonosc  : the OSC library (pip install python-osc): Dispatcher maps addresses
#              to functions, BlockingOSCUDPServer listens on a UDP port
# math       : atan2 / degrees for the head roll
import argparse
import math
import sys
import threading
import time

import cv2
import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# --- settings ------------------------------------------------------------------
# Coordinates arrive normalized (0..1): multiply by WIDTH / HEIGHT to draw.
WIDTH, HEIGHT = 800, 600
STALE_SECONDS = 1.0
# Blendshapes are 52 numbers between 0 (neutral) and 1 (maximum) that describe
# the expression: jawOpen, eyeBlinkLeft, mouthSmileRight ...  A threshold on one
# of them turns an expression into an event.
MOUTH_OPEN = 0.4                    # jawOpen above this = "mouth open" event

# A few useful landmark indices of the 478-point mesh.
# The mesh has 478 points; these are the handful we need by name.  The eye
# corners give the head roll, the oval is drawn as an outline.  Note that "left"
# is the person's left: without --mirror it appears on the viewer's right.
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 263, 33        # "left" = the person's left (viewer's right without --mirror)
MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM = 291, 61, 13, 14
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Default blendshape order (the app sends the list at start-up too; we take whichever arrives).
# The 52 blendshape names, in the order the values arrive.  The app also sends
# this list at start-up (/mp/face/blendshape_names); we use whichever we have.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
# The few blendshapes drawn as bars next to the face (pick any of the 52).
SHOWN_BLENDSHAPES = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
                     "browInnerUp", "mouthPucker", "cheekPuff"]


# --- the data model --------------------------------------------------------------
# One object per face, by stable id.  `blendshapes` is a dictionary name -> value
# so the code can ask for f.value("jawOpen") instead of remembering index 25.
class Face:
    def __init__(self, fid):
        self.id = fid
        self.center = (0.5, 0.5, 0.0)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.blendshapes = {}            # name -> value
        self.landmarks = None            # (478, 3) or None
        self.mouth_open = False
        self.last_seen = time.time()

    # atan2 of the line between the two outer eye corners: 0 when the eyes are
    # level, positive when the head tilts one way, negative the other.  An
    # installation uses it to rotate an image placed over the face.
    def roll_degrees(self):
        """Head roll (tilt) from the line between the outer eye corners, 0 = upright."""
        if self.landmarks is None:
            return 0.0
        left, right = self.landmarks[LEFT_EYE_OUTER], self.landmarks[RIGHT_EYE_OUTER]
        return math.degrees(math.atan2(left[1] - right[1], left[0] - right[0]))

    def value(self, name):
        return self.blendshapes.get(name, 0.0)


# --- shared state ----------------------------------------------------------------
# Written by the OSC handlers (server thread), read by draw() (main thread), always
# under the lock.  `names` is the current blendshape order; stats["mesh"] remembers
# whether the 478-point message ever arrived (it needs --face-landmarks).
faces = {}
names = list(BLENDSHAPE_NAMES)
lock = threading.Lock()
stats = {"messages": 0, "frame": 0, "fps": 0.0, "status": "waiting for mediapipe_osc.py ...", "mesh": False}


# Fetch the Face for an id, creating it if needed, and refresh last_seen.
def get_face(fid):
    f = faces.get(fid)
    if f is None:
        f = faces[fid] = Face(fid)
    f.last_seen = time.time()
    return f


# ---------------------------------------------------------------------------- OSC handlers
# python-osc calls one of these functions for every message whose address was
# mapped in make_dispatcher(): first the address, then the OSC arguments already
# converted to int / float / str.  /mp/frame is a once-per-frame heartbeat.
def on_frame(address, frame, time_s, fps, *counts):
    with lock:
        stats["frame"], stats["fps"] = frame, fps


# /mp/status "started" | "stopped": on "stopped" we forget everything, because
# no more leave messages will come.
def on_status(address, text, version):
    print(f"mediapipe_osc.py {text} (version {version})")
    with lock:
        stats["status"] = f"app {text}"
        if text == "stopped":
            faces.clear()


# /mp/face/blendshape_names: 52 strings, sent once when the app starts.  If we
# receive it we trust it over the built-in list (names[:] = ... replaces the
# contents of the shared list in place).
def on_blendshape_names(address, *values):
    with lock:
        names[:] = [str(v) for v in values]


# /mp/face/enter id and /mp/face/leave id: faces appearing and disappearing.
def on_enter(address, fid):
    with lock:
        faces[fid] = Face(fid)
    print(f"face {fid} entered")


def on_leave(address, fid):
    with lock:
        faces.pop(fid, None)
    print(f"face {fid} left")


# /mp/face/center id x y z: the nose tip.
def on_center(address, fid, x, y, z):
    with lock:
        get_face(fid).center = (x, y, z)


# /mp/face/bbox id xmin ymin xmax ymax: the rectangle around the face.
def on_bbox(address, fid, *box):
    with lock:
        get_face(fid).bbox = tuple(box[:4])


# /mp/face/velocity id vx vy speed: movement of the nose tip, in picture sizes per second.
def on_velocity(address, fid, vx, vy, speed):
    with lock:
        f = get_face(fid)
        f.velocity, f.speed = (vx, vy), speed


# /mp/face/blendshapes id + 52 values, every frame.  zip(names, values) pairs
# each name with its value into a dictionary; then the jawOpen threshold is
# compared with the previous state to produce the "mouth open / closed" event.
def on_blendshapes(address, fid, *values):
    with lock:
        f = get_face(fid)
        f.blendshapes = dict(zip(names, values))
        is_open = f.value("jawOpen") > MOUTH_OPEN
        if is_open != f.mouth_open:                    # an event
            f.mouth_open = is_open
            print(f"face {fid} mouth {'open' if is_open else 'closed'}")


# /mp/face/landmarks id + 478 x (x y z): the mesh (only with --face-landmarks).
# n = number of points received; reshape(n, 3) gives one row per point.
def on_landmarks(address, fid, *values):
    n = len(values) // 3
    if n < 468:
        return
    with lock:
        get_face(fid).landmarks = np.array(values[:n * 3], dtype=np.float32).reshape(n, 3)
        stats["mesh"] = True


# Default handler: every address that was not mapped ends here.
def on_other(address, *args):
    """Poses, hands, blobs ... are ignored here."""


# A small wrapper ("decorator") applied to every handler: counts the messages
# for the status line, then calls the real handler.
def counted(handler):
    def wrapper(address, *args):
        with lock:
            stats["messages"] += 1
            if stats["messages"] == 1:
                stats["status"] = "receiving"
        handler(address, *args)
    return wrapper


# The dispatcher is the table address -> function.  Add a line here to listen
# to another address.
def make_dispatcher():
    d = Dispatcher()
    d.map("/mp/frame", counted(on_frame))
    d.map("/mp/status", counted(on_status))
    d.map("/mp/face/blendshape_names", counted(on_blendshape_names))
    d.map("/mp/face/enter", counted(on_enter))
    d.map("/mp/face/leave", counted(on_leave))
    d.map("/mp/face/center", counted(on_center))
    d.map("/mp/face/bbox", counted(on_bbox))
    d.map("/mp/face/velocity", counted(on_velocity))
    d.map("/mp/face/blendshapes", counted(on_blendshapes))
    d.map("/mp/face/landmarks", counted(on_landmarks))
    d.set_default_handler(counted(on_other))
    return d


# ---------------------------------------------------------------------------- drawing
# Normalized (0..1) -> window pixels.  Used for every coordinate we draw.
def to_px(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


# A colour per id (OpenCV uses BGR and a 0..179 hue).
def color_for(fid):
    hue = (fid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 160, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Draw everything we know.  `snapshot` is a list of Face objects copied under
# the lock, so the shared dictionary is never used while the server thread
# changes it.
def draw(img, snapshot):
    img[:] = (20, 20, 20)
    for f in snapshot:
        col = color_for(f.id)
        x0, y0 = to_px(f.bbox[0], f.bbox[1])
        x1, y1 = to_px(f.bbox[2], f.bbox[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        # the "mask": an ellipse the size of the face, rotated with the head.
        # An installation would draw its own image here instead.
        # cv2.ellipse takes the centre, the two half-axes and the rotation in degrees:
        # the roll from the eyes rotates the mask with the head.
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        axes = (max(2, (x1 - x0) // 2), max(2, (y1 - y0) // 2))
        roll = f.roll_degrees()
        cv2.ellipse(img, (cx, cy), axes, roll, 0, 360, (60, 60, 60), -1)
        cv2.ellipse(img, (cx, cy), axes, roll, 0, 360, col, 2)

        if f.landmarks is not None:
            lm = f.landmarks
            for i in range(len(lm)):                                    # the mesh
                cv2.circle(img, to_px(lm[i, 0], lm[i, 1]), 1, col, -1)
            # polylines wants an integer array of points; True = close the polygon.
            oval = np.array([to_px(lm[i, 0], lm[i, 1]) for i in FACE_OVAL], dtype=np.int32)
            cv2.polylines(img, [oval], True, col, 1)
            for i in (LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM, CHIN, FOREHEAD):
                cv2.circle(img, to_px(lm[i, 0], lm[i, 1]), 4, (255, 255, 255), -1)
            cv2.line(img, to_px(lm[LEFT_EYE_OUTER, 0], lm[LEFT_EYE_OUTER, 1]),
                     to_px(lm[RIGHT_EYE_OUTER, 0], lm[RIGHT_EYE_OUTER, 1]), (255, 255, 255), 1)

        nx, ny = to_px(f.center[0], f.center[1])                        # nose tip
        cv2.circle(img, (nx, ny), 5, (0, 0, 255), -1)
        cv2.arrowedLine(img, (nx, ny), (int(nx + f.velocity[0] * WIDTH * 0.25), int(ny + f.velocity[1] * HEIGHT * 0.25)), col, 2)

        label = f"F{f.id}  roll {roll:+.0f} deg{'  MOUTH OPEN' if f.mouth_open else ''}"
        cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        # blendshape bars, to the right of the face (or to the left near the border)
        # Bars go to the right of the box unless that would leave the window.
        bx = x1 + 10 if x1 + 150 < WIDTH else x0 - 150
        for k, name in enumerate(SHOWN_BLENDSHAPES):
            v = f.value(name)
            y = y0 + k * 14
            cv2.rectangle(img, (bx, y), (bx + int(80 * v), y + 8), col, -1)
            cv2.rectangle(img, (bx, y), (bx + 80, y + 8), (90, 90, 90), 1)
            cv2.putText(img, name, (bx + 86, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    mesh = "mesh: yes" if stats["mesh"] else "mesh: no (start the app with --face-landmarks)"
    cv2.putText(img, f"faces: {len(snapshot)}   {mesh}", (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stats['status']}   frame {stats['frame']}   analysis {stats['fps']:.1f} fps   messages {stats['messages']}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------- main
# 1. parse the options; 2. start the OSC server in a background thread (daemon =
# it dies with the program); 3. loop: drop stale objects, draw, show the window,
# read the keyboard, stop after --seconds if asked.
def main(argv=None):
    ap = argparse.ArgumentParser(description="Face landmarks OSC receiver (mediapipe_osc.py --face --face-landmarks)")
    ap.add_argument("--ip", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5005, help="OSC port (default 5005)")
    ap.add_argument("--headless", action="store_true", help="no window; print a summary (for tests)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = never)")
    ap.add_argument("--snapshot", default="", help="save the last drawn frame to this PNG file")
    args = ap.parse_args(argv)

    server = BlockingOSCUDPServer((args.ip, args.port), make_dispatcher())
    # the 478-point mesh message is ~7 KB: make sure the socket buffer takes it
    # python-osc's server reads each UDP packet into a buffer of max_packet_size
    # bytes (8192 by default); the mesh message is about 7200 bytes, so raising it
    # leaves room to spare.
    server.max_packet_size = 65535
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening for OSC on {args.ip}:{args.port} - run mediapipe_osc.py --face --face-landmarks (q or Esc quits)")

    # The image we draw into: HEIGHT x WIDTH pixels x 3 colour channels, 8 bit.
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t0 = time.time()
    seen = set()
    try:
        while True:
            now = time.time()
            with lock:
                # A leave message can be lost (UDP gives no guarantee), so a timeout is the safety net.
                for fid in [i for i, f in faces.items() if now - f.last_seen > STALE_SECONDS]:
                    del faces[fid]
                snapshot = list(faces.values())
                seen.update(faces.keys())
            draw(img, snapshot)
            # --headless (used by `make test`): no window, just keep the loop alive.
            # Otherwise show the image; waitKey(15) also gives OpenCV time to draw and
            # returns the pressed key (27 = Esc).
            if args.headless:
                time.sleep(0.2)
            else:
                cv2.imshow("face landmarks (OSC)", img)
                if cv2.waitKey(15) & 0xFF in (27, ord("q")):
                    break
            if args.seconds and now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        cv2.destroyAllWindows()

    if args.snapshot:
        cv2.imwrite(args.snapshot, img)
    with lock:
        n, mesh = stats["messages"], stats["mesh"]
    print(f"done: {n} messages received, faces seen: {sorted(seen)}, mesh received: {mesh}")
    # Exit code 2 = nothing arrived: `make test` uses it to detect a broken setup.
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
