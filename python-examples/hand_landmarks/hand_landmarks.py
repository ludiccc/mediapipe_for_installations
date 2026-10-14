#!/usr/bin/env python3
"""Hand landmarks receiver — Python + python-osc + OpenCV.

Listens to the OSC messages of mediapipe_osc.py for the **Hands** detector and
draws every hand: the 21 landmarks with their connections, left/right, the
pinch (distance thumb tip - index tip) as a circle, and the fingertips.

    python ../../mediapipe_osc.py --hands --mirror       # terminal 1 (the camera)
    python hand_landmarks.py                             # terminal 2 (this)

    python hand_landmarks.py --port 9000
    python hand_landmarks.py --headless --seconds 5      # no window (tests)

Messages used (first argument = stable id of the hand):

    /mp/hand/enter      id
    /mp/hand/leave      id
    /mp/hand/info       id handedness score      handedness = "Left" | "Right"
    /mp/hand/center     id x y z
    /mp/hand/bbox       id xmin ymin xmax ymax
    /mp/hand/velocity   id vx vy speed
    /mp/hand/pinch      id distance              thumb tip <-> index tip, roughly metres
    /mp/hand/landmarks  id 21 x (x y z)

Note: with --mirror, "Right" is the hand the viewer sees on the right.
Landmark indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
"""

# --- modules -------------------------------------------------------------------
# argparse   : command-line options (--port, --headless ...)
# threading  : the OSC server runs in its own thread, next to the drawing loop
# cv2        : OpenCV, used only to draw and to show a window
# numpy      : arrays; the landmarks of one hand/face/blob live in one array
# pythonosc  : the OSC library (pip install python-osc): Dispatcher maps addresses
#              to functions, BlockingOSCUDPServer listens on a UDP port
import argparse
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
# The pinch is the distance between the thumb tip and the index tip, roughly in
# metres (it comes from MediaPipe's 3-D hand model).  Below PINCH_CLOSED we call
# it "closed": a free "click" for an installation.
PINCH_CLOSED = 0.04          # pinch distance below this = "pinching" (tune for your camera)

# Names for the landmark indices we use.  The 21 hand landmarks go wrist (0),
# then four points per finger: thumb 1-4, index 5-8, middle 9-12, ring 13-16,
# pinky 17-20; the last of each group is the fingertip.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 0, 4, 8, 12, 16, 20
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
# Pairs of landmarks joined by a line to draw the hand skeleton.
CONNECTIONS = [           # MediaPipe's HAND_CONNECTIONS
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),   # pinky + palm
]


# --- the data model --------------------------------------------------------------
# One object per hand, kept in a dictionary by the stable id the app gives it.
# `pinching` remembers the last state so that a change becomes an event.
class Hand:
    def __init__(self, hid):
        self.id = hid
        self.handedness = "?"
        self.score = 0.0
        self.center = (0.5, 0.5, 0.0)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.pinch = 1.0
        self.pinching = False
        # numpy array with one row per landmark: x, y, z (empty until the first landmarks message)
        self.landmarks = None          # (21, 3)
        self.last_seen = time.time()


# --- shared state ----------------------------------------------------------------
# Written by the OSC handlers (server thread), read by draw() (main thread):
# every access goes through `with lock:` so a half-updated object is never drawn.
hands = {}
lock = threading.Lock()
stats = {"messages": 0, "frame": 0, "fps": 0.0, "status": "waiting for mediapipe_osc.py ..."}


# Fetch the Hand for an id, creating it if needed (so a receiver started after the
# app still works), and refresh last_seen.
def get_hand(hid):
    h = hands.get(hid)
    if h is None:
        h = hands[hid] = Hand(hid)
    h.last_seen = time.time()
    return h


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
            hands.clear()


# /mp/hand/enter id: a new hand appeared.
def on_enter(address, hid):
    with lock:
        hands[hid] = Hand(hid)
    print(f"hand {hid} entered")


# /mp/hand/leave id: the hand disappeared (the id is never reused).
def on_leave(address, hid):
    with lock:
        hands.pop(hid, None)
    print(f"hand {hid} left")


# /mp/hand/info id handedness score: "Left" or "Right" (as seen by the camera;
# with --mirror, as the viewer sees it) and how sure the model is.
def on_info(address, hid, handedness, score):
    with lock:
        h = get_hand(hid)
        h.handedness, h.score = handedness, score


# /mp/hand/center id x y z: the centre of the hand.
def on_center(address, hid, x, y, z):
    with lock:
        get_hand(hid).center = (x, y, z)


# /mp/hand/bbox id xmin ymin xmax ymax: the rectangle around the hand.
def on_bbox(address, hid, *box):
    with lock:
        get_hand(hid).bbox = tuple(box[:4])


# /mp/hand/velocity id vx vy speed: movement of the centre, in picture sizes per second.
def on_velocity(address, hid, vx, vy, speed):
    with lock:
        h = get_hand(hid)
        h.velocity, h.speed = (vx, vy), speed


# /mp/hand/pinch id distance: arrives every frame.  We compare with the threshold
# and only *print* when the state changes - that is how a continuous value is
# turned into an event (pinch! / released).
def on_pinch(address, hid, distance):
    with lock:
        h = get_hand(hid)
        h.pinch = distance
        pinching = distance < PINCH_CLOSED
        if pinching != h.pinching:                 # an event you could trigger something with
            h.pinching = pinching
            print(f"hand {hid} ({h.handedness}) {'pinch!' if pinching else 'released'}")


# /mp/hand/landmarks id + 21 x (x y z): *values collects the 63 numbers;
# reshape(21, 3) gives one row per landmark, so lm[INDEX_TIP, 0] is the x of the
# index fingertip.
def on_landmarks(address, hid, *values):
    if len(values) < 21 * 3:
        return
    with lock:
        get_hand(hid).landmarks = np.array(values[:21 * 3], dtype=np.float32).reshape(21, 3)


# Default handler: every address that was not mapped ends here.
def on_other(address, *args):
    """Poses, faces, blobs, gestures ... are ignored here."""


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
    d.map("/mp/hand/enter", counted(on_enter))
    d.map("/mp/hand/leave", counted(on_leave))
    d.map("/mp/hand/info", counted(on_info))
    d.map("/mp/hand/center", counted(on_center))
    d.map("/mp/hand/bbox", counted(on_bbox))
    d.map("/mp/hand/velocity", counted(on_velocity))
    d.map("/mp/hand/pinch", counted(on_pinch))
    d.map("/mp/hand/landmarks", counted(on_landmarks))
    d.set_default_handler(counted(on_other))
    return d


# ---------------------------------------------------------------------------- drawing
# Normalized (0..1) -> window pixels.  Used for every coordinate we draw.
def to_px(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


# Draw everything we know.  `snapshot` is a list of Hand objects copied under the
# lock, so the shared dictionary is never used while the server thread changes it.
def draw(img, snapshot):
    img[:] = (20, 20, 20)
    for h in snapshot:
        # One colour per handedness (OpenCV colours are BGR: blue, green, red).
        col = (80, 200, 255) if h.handedness == "Right" else (255, 160, 60)    # BGR
        x0, y0 = to_px(h.bbox[0], h.bbox[1])
        x1, y1 = to_px(h.bbox[2], h.bbox[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        if h.landmarks is not None:
            lm = h.landmarks
            for a, b in CONNECTIONS:
                cv2.line(img, to_px(lm[a, 0], lm[a, 1]), to_px(lm[b, 0], lm[b, 1]), col, 2)
            for i in range(21):
                r = 6 if i in FINGERTIPS else 3
                cv2.circle(img, to_px(lm[i, 0], lm[i, 1]), r, col, -1)
            # The circle radius follows the pinch distance (x 400 px); -1 thickness = filled.
            # the pinch: a circle between thumb tip and index tip, filled when closed
            tx, ty = to_px(lm[THUMB_TIP, 0], lm[THUMB_TIP, 1])
            ix, iy = to_px(lm[INDEX_TIP, 0], lm[INDEX_TIP, 1])
            mx, my = (tx + ix) // 2, (ty + iy) // 2
            radius = max(4, int(h.pinch * 400))
            cv2.circle(img, (mx, my), radius, (255, 255, 255), -1 if h.pinching else 1)

        cx, cy = to_px(h.center[0], h.center[1])
        cv2.arrowedLine(img, (cx, cy), (int(cx + h.velocity[0] * WIDTH * 0.25), int(cy + h.velocity[1] * HEIGHT * 0.25)), col, 2)
        label = f"H{h.id} {h.handedness} ({h.score:.2f})  pinch {h.pinch:.3f}{'  PINCH' if h.pinching else ''}"
        cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(img, f"hands: {len(snapshot)}", (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stats['status']}   frame {stats['frame']}   analysis {stats['fps']:.1f} fps   messages {stats['messages']}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------- main
# 1. parse the options; 2. start the OSC server in a background thread (daemon =
# it dies with the program); 3. loop: drop stale objects, draw, show the window,
# read the keyboard, stop after --seconds if asked.
def main(argv=None):
    ap = argparse.ArgumentParser(description="Hand landmarks OSC receiver (mediapipe_osc.py --hands)")
    ap.add_argument("--ip", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5005, help="OSC port (default 5005)")
    ap.add_argument("--headless", action="store_true", help="no window; print a summary (for tests)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = never)")
    ap.add_argument("--snapshot", default="", help="save the last drawn frame to this PNG file")
    args = ap.parse_args(argv)

    server = BlockingOSCUDPServer((args.ip, args.port), make_dispatcher())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening for OSC on {args.ip}:{args.port} - run mediapipe_osc.py --hands (q or Esc quits)")

    # The image we draw into: HEIGHT x WIDTH pixels x 3 colour channels, 8 bit.
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t0 = time.time()
    seen = set()
    try:
        while True:
            now = time.time()
            with lock:
                # A leave message can be lost (UDP gives no guarantee), so a timeout is the safety net.
                for hid in [i for i, h in hands.items() if now - h.last_seen > STALE_SECONDS]:
                    del hands[hid]
                snapshot = list(hands.values())
                seen.update(hands.keys())
            draw(img, snapshot)
            # --headless (used by `make test`): no window, just keep the loop alive.
            # Otherwise show the image; waitKey(15) also gives OpenCV time to draw and
            # returns the pressed key (27 = Esc).
            if args.headless:
                time.sleep(0.2)
            else:
                cv2.imshow("hand landmarks (OSC)", img)
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
        n = stats["messages"]
    print(f"done: {n} messages received, hands seen: {sorted(seen)}")
    # Exit code 2 = nothing arrived: `make test` uses it to detect a broken setup.
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
