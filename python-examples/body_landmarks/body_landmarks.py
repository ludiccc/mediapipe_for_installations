#!/usr/bin/env python3
"""Body landmarks (pose) receiver — Python + python-osc + OpenCV.

Listens to the OSC messages of mediapipe_osc.py for the **Body** detector and
draws every person it knows about: skeleton, bounding box, id, velocity arrow,
which third of the image they stand in, and whether a hand is raised.

    python ../../mediapipe_osc.py --pose --mirror        # terminal 1 (the camera)
    python body_landmarks.py                             # terminal 2 (this)

    python body_landmarks.py --port 9000                 # another port
    python body_landmarks.py --headless --seconds 5      # no window, print what arrives (tests)

Messages used (first argument = stable id of the person):

    /mp/pose/enter      id
    /mp/pose/leave      id
    /mp/pose/center     id x y z                 visibility-weighted centre of the body
    /mp/pose/bbox       id xmin ymin xmax ymax
    /mp/pose/velocity   id vx vy speed           normalized units per second
    /mp/pose/landmarks  id 33 x (x y z visibility)

x, y are normalized (0..1, origin top-left).  Landmarks outside the picture
are extrapolated by MediaPipe and can be < 0 or > 1.
Landmark indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

# --- modules -------------------------------------------------------------------
# argparse   : reads the command-line options (--port, --headless ...)
# threading  : the OSC server runs in its own thread, next to the drawing loop
# cv2        : OpenCV, used here only to draw and to show a window
# numpy      : fast arrays; the 33 landmarks of a person live in one (33, 4) array
# pythonosc  : the OSC library (pip install python-osc): Dispatcher maps addresses
#              to functions, BlockingOSCUDPServer listens on a UDP port
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
# The app sends normalized coordinates (0..1), so the window can be any size:
# every x is multiplied by WIDTH and every y by HEIGHT when drawing.
WIDTH, HEIGHT = 800, 600      # window size (4:3 like the default 640x480 camera)
STALE_SECONDS = 1.0           # forget a person if nothing arrived for this long (missed "leave")

# The 33 pose landmarks, by name, and the lines between them (MediaPipe's POSE_CONNECTIONS).
# Give names to the landmark indices we use, so the code reads "L_WRIST"
# instead of "15".  The full list (0 nose ... 32 right foot index) is in the
# MediaPipe documentation linked above.
NOSE, L_EYE, R_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
# Pairs of landmark indices joined by a line to draw the skeleton.  This is
# MediaPipe's own POSE_CONNECTIONS list, copied so the example needs no
# MediaPipe installed on the receiver side.
CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (11, 23), (12, 14), (12, 24), (13, 15), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (17, 19), (18, 20),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30), (28, 32),
    (29, 31), (30, 32),
]


# --- the data model --------------------------------------------------------------
# One object per person.  The app gives every person a stable id (the first
# argument of every /mp/pose/... message): the object with that id is updated by
# each message and deleted when the person leaves.  Methods like hand_raised()
# and zone() are examples of "reading" the data for an installation.
class Person:
    """Everything we know about one tracked body."""

    def __init__(self, pid):
        self.id = pid
        # Defaults until the first messages arrive (centre of the picture, empty box).
        self.center = (0.5, 0.5, 0.0)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.landmarks = None            # numpy array (33, 4): x y z visibility
        # When the last message for this person arrived (to drop stale people).
        self.last_seen = time.time()

    def hand_raised(self):
        """True when a wrist is above the shoulders (y grows downwards)."""
        if self.landmarks is None:
            return False
        # Column 1 of the array is y, column 3 is visibility (0..1).  In image
        # coordinates y grows downwards, so "above" means a smaller y.
        lm = self.landmarks
        shoulders = min(lm[L_SHOULDER, 1], lm[R_SHOULDER, 1])
        return bool((lm[L_WRIST, 1] < shoulders and lm[L_WRIST, 3] > 0.5) or
                    (lm[R_WRIST, 1] < shoulders and lm[R_WRIST, 3] > 0.5))

    def zone(self):
        """Which third of the picture the person stands in: 0 = left, 1 = centre, 2 = right."""
        # int(x * 3) is 0, 1 or 2 for x in 0..1; min/max clamp the values outside
        # the picture that MediaPipe can produce when a person is half out of view.
        return min(2, max(0, int(self.center[0] * 3)))


# ---------------------------------------------------------------------------- state
# These three objects are written by the OSC handlers (server thread) and read
# by draw() (main thread).  Every access goes through `with lock:` so that one
# thread never sees the other's half-finished change.
people = {}                  # id -> Person
lock = threading.Lock()      # OSC handlers run on another thread than the drawing loop
stats = {"messages": 0, "frame": 0, "fps": 0.0, "status": "waiting for mediapipe_osc.py ..."}


# Fetch the Person for an id, creating it if needed.  Creating on the fly means
# the receiver can be started *after* the app, or miss an enter message, and
# still work.  It also refreshes last_seen.
def get_person(pid):
    p = people.get(pid)
    if p is None:                     # data for somebody we did not see enter (started late)
        p = people[pid] = Person(pid)
    p.last_seen = time.time()
    return p


# ---------------------------------------------------------------------------- OSC handlers
# python-osc calls one of these functions for every message whose address was
# mapped in make_dispatcher().  The first parameter is always the address; the
# rest are the OSC arguments, already converted to int / float / str.
# /mp/frame arrives once per analysed camera frame: a heartbeat with counters.
def on_frame(address, frame, time_s, fps, n_poses, n_hands, n_faces, n_blobs):
    with lock:
        stats["frame"], stats["fps"] = frame, fps


# /mp/status "started" | "stopped": sent when the app starts and quits.  On
# "stopped" we forget everybody, because no more leave messages will come.
def on_status(address, text, version):
    print(f"mediapipe_osc.py {text} (version {version})")
    with lock:
        stats["status"] = f"app {text}"
        if text == "stopped":
            people.clear()


# /mp/pose/enter id: a new person.  This is where an installation would start
# something per person (a sound, a particle system, a colour ...).
def on_enter(address, pid):
    with lock:
        people[pid] = Person(pid)
    print(f"person {pid} entered")


# /mp/pose/leave id: the person disappeared (tracking lost for 0.5 s).  The id
# is never reused: somebody who comes back gets a new one.
def on_leave(address, pid):
    with lock:
        people.pop(pid, None)
    print(f"person {pid} left")


# /mp/pose/center id x y z: the centre of the body (z is MediaPipe's relative depth).
def on_center(address, pid, x, y, z):
    with lock:
        get_person(pid).center = (x, y, z)


# /mp/pose/bbox id xmin ymin xmax ymax: the rectangle around the person.
def on_bbox(address, pid, xmin, ymin, xmax, ymax):
    with lock:
        get_person(pid).bbox = (xmin, ymin, xmax, ymax)


# /mp/pose/velocity id vx vy speed: how fast the centre moves, in picture widths
# / heights per second (0.5 = half the picture in one second).
def on_velocity(address, pid, vx, vy, speed):
    with lock:
        p = get_person(pid)
        p.velocity, p.speed = (vx, vy), speed


# /mp/pose/landmarks id + 33 x (x y z visibility): the long one.  *values collects
# all 132 numbers in a tuple; reshape(33, 4) turns them into one row per landmark,
# so lm[15, 0] is the x of the left wrist and lm[15, 3] its visibility.
def on_landmarks(address, pid, *values):
    # 33 x (x, y, z, visibility) = 132 floats after the id
    if len(values) < 33 * 4:
        return
    with lock:
        get_person(pid).landmarks = np.array(values[:33 * 4], dtype=np.float32).reshape(33, 4)


# Default handler: every address that was not mapped ends here (the app may be
# sending hands, faces and blobs at the same time).
def on_other(address, *args):
    """Anything else (hands, faces, blobs ...) is ignored here."""


# A small wrapper ("decorator") applied to every handler: it counts the messages
# for the status line and then calls the real handler.  The first message also
# switches the status from "waiting" to "receiving".
def counted(handler):
    """Wrap a handler so every message is counted (shown in the window)."""
    def wrapper(address, *args):
        with lock:
            stats["messages"] += 1
            if stats["messages"] == 1:
                stats["status"] = "receiving"
        handler(address, *args)
    return wrapper


# The dispatcher is the table address -> function.  Add a line here to listen
# to a new address (for example "/mp/pose/world" with --world).
def make_dispatcher():
    d = Dispatcher()
    d.map("/mp/frame", counted(on_frame))
    d.map("/mp/status", counted(on_status))
    d.map("/mp/pose/enter", counted(on_enter))
    d.map("/mp/pose/leave", counted(on_leave))
    d.map("/mp/pose/center", counted(on_center))
    d.map("/mp/pose/bbox", counted(on_bbox))
    d.map("/mp/pose/velocity", counted(on_velocity))
    d.map("/mp/pose/landmarks", counted(on_landmarks))
    d.set_default_handler(counted(on_other))
    return d


# ---------------------------------------------------------------------------- drawing
# Derive a colour from the id so each person keeps the same colour while tracked.
# OpenCV wants BGR (blue, green, red) tuples, and its hue goes 0..179.
def color_for(pid):
    """A different colour per id (BGR)."""
    hue = (pid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Normalized (0..1) -> window pixels.  Used for every coordinate we draw.
def to_px(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


# Draw everything we know into the image.  `snapshot` is a plain list of Person
# objects copied under the lock, so this function never touches the shared dict
# while the server thread modifies it.  OpenCV draws with side effects on `img`.
def draw(img, snapshot):
    img[:] = (20, 20, 20)
    # thirds (the "zones")
    for i in (1, 2):
        cv2.line(img, (WIDTH * i // 3, 0), (WIDTH * i // 3, HEIGHT), (50, 50, 50), 1)
    zone_count = [0, 0, 0]

    for p in snapshot:
        col = color_for(p.id)
        zone_count[p.zone()] += 1

        # bounding box
        x0, y0 = to_px(p.bbox[0], p.bbox[1])
        x1, y1 = to_px(p.bbox[2], p.bbox[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        # Visibility is MediaPipe's confidence that a landmark is really in the picture;
        # we skip lines with an invisible end and draw bigger dots for visible points.
        # skeleton: lines between landmarks, then the points (brighter when more visible)
        if p.landmarks is not None:
            lm = p.landmarks
            for a, b in CONNECTIONS:
                if lm[a, 3] > 0.3 and lm[b, 3] > 0.3:
                    cv2.line(img, to_px(lm[a, 0], lm[a, 1]), to_px(lm[b, 0], lm[b, 1]), col, 2)
            for i in range(33):
                v = float(lm[i, 3])
                cv2.circle(img, to_px(lm[i, 0], lm[i, 1]), 3 + int(3 * v), col, -1)

        # centre, velocity arrow, label
        cx, cy = to_px(p.center[0], p.center[1])
        cv2.circle(img, (cx, cy), 8, col, -1)
        ax, ay = int(cx + p.velocity[0] * WIDTH * 0.25), int(cy + p.velocity[1] * HEIGHT * 0.25)
        cv2.arrowedLine(img, (cx, cy), (ax, ay), col, 2)
        # 'LCR'[0] is 'L', 'LCR'[1] is 'C' ...: a one-letter zone name.
        label = f"P{p.id}  speed {p.speed:.2f}  zone {'LCR'[p.zone()]}"
        if p.hand_raised():
            label += "  HAND UP!"
        cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(img, f"people: {len(snapshot)}   left {zone_count[0]}  centre {zone_count[1]}  right {zone_count[2]}",
                (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stats['status']}   frame {stats['frame']}   analysis {stats['fps']:.1f} fps   "
                f"messages {stats['messages']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------- main
# 1. parse the options; 2. start the OSC server in a background thread (daemon =
# it dies with the program); 3. loop: drop stale people, draw, show the window,
# read the keyboard, stop after --seconds if asked.
def main(argv=None):
    ap = argparse.ArgumentParser(description="Body landmarks OSC receiver (mediapipe_osc.py --pose)")
    ap.add_argument("--ip", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5005, help="OSC port (default 5005)")
    ap.add_argument("--headless", action="store_true", help="no window; print a summary (for tests)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = never)")
    ap.add_argument("--snapshot", default="", help="save the last drawn frame to this PNG file")
    args = ap.parse_args(argv)

    server = BlockingOSCUDPServer((args.ip, args.port), make_dispatcher())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening for OSC on {args.ip}:{args.port} - run mediapipe_osc.py --pose (q or Esc quits)")

    # The image we draw into: HEIGHT x WIDTH pixels x 3 colour channels, 8 bit.
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t0 = time.time()
    seen = set()
    try:
        while True:
            now = time.time()
            with lock:
                # A leave message can be lost (UDP gives no guarantee), so a timeout is the safety net.
                # forget people whose data stopped arriving (e.g. the app was killed)
                for pid in [i for i, p in people.items() if now - p.last_seen > STALE_SECONDS]:
                    del people[pid]
                snapshot = list(people.values())
                seen.update(people.keys())
            draw(img, snapshot)

            # --headless (used by `make test`): no window, just keep the loop alive.
            # Otherwise show the image; waitKey(15) also gives OpenCV time to draw and
            # returns the pressed key (27 = Esc).
            if args.headless:
                time.sleep(0.2)
            else:
                cv2.imshow("body landmarks (OSC)", img)
                key = cv2.waitKey(15) & 0xFF
                if key in (27, ord("q")):
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
        n = stats["messages"]
    print(f"done: {n} messages received, people seen: {sorted(seen)}")
    # Exit code 2 = nothing arrived: `make test` uses it to detect a broken setup.
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
