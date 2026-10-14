#!/usr/bin/env python3
"""Hand gestures receiver — Python + python-osc + OpenCV.

Listens to the OSC messages of mediapipe_osc.py for the **Gestures** detector
(hands + gesture classification) and shows, for every hand, the recognised
gesture with its score, plus a log of gesture *changes* — the events an
installation usually reacts to.  As a demo, each gesture tints the background
with its own colour and a Thumb_Up makes the screen flash.

    python ../../mediapipe_osc.py --gesture --mirror     # terminal 1 (the camera)
    python hand_gestures.py                              # terminal 2 (this)

    python hand_gestures.py --port 9000
    python hand_gestures.py --headless --seconds 5       # no window (tests)

Messages used (first argument = stable id of the hand):

    /mp/hand/enter      id
    /mp/hand/leave      id
    /mp/hand/info       id handedness score
    /mp/hand/center     id x y z
    /mp/hand/bbox       id xmin ymin xmax ymax
    /mp/hand/gesture    id name score           name: None, Closed_Fist, Open_Palm, Pointing_Up,
                                                      Thumb_Down, Thumb_Up, Victory, ILoveYou
    /mp/hand/landmarks  id 21 x (x y z)

(--gesture implies --hands, so all the hand messages of hand_landmarks.py arrive too.)
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
# The recogniser sends a score (0..1) with every gesture.  Below MIN_SCORE we
# treat the gesture as "None" to avoid flickering between doubtful guesses.
MIN_SCORE = 0.5                    # ignore gestures the classifier is not sure about

# A colour (BGR) and a small ASCII "icon" per gesture, to react to them.
# The eight gesture names MediaPipe's recogniser knows, each with a colour (BGR)
# for the background and a small ASCII icon.  The keys are exactly the strings
# that arrive in /mp/hand/gesture.
GESTURES = {
    "None":        ((40, 40, 40),    "   "),
    "Closed_Fist": ((60, 60, 180),   "(o)"),
    "Open_Palm":   ((60, 160, 60),   "\\o/"),
    "Pointing_Up": ((180, 120, 40),  " ^ "),
    "Thumb_Down":  ((120, 40, 120),  "-1 "),
    "Thumb_Up":    ((40, 200, 220),  "+1 "),
    "Victory":     ((200, 160, 40),  " V "),
    "ILoveYou":    ((180, 60, 200),  "\\m/"),
}
# Hand skeleton lines (MediaPipe's HAND_CONNECTIONS), drawn thin behind the gesture label.
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# --- the data model --------------------------------------------------------------
# One object per hand, by stable id.  `gesture` holds the *last* gesture, so a
# new name in the next message means "the gesture changed" = an event.
class Hand:
    def __init__(self, hid):
        self.id = hid
        self.handedness = "?"
        self.center = (0.5, 0.5, 0.0)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.gesture = "None"
        self.score = 0.0
        self.landmarks = None
        self.last_seen = time.time()


hands = {}
# --- shared state ----------------------------------------------------------------
# `events` is a short log of the things that happened, shown in the window.
# Everything here is written by the server thread and read by draw(), under the lock.
events = []                 # last gesture changes, newest last: (time, text)
lock = threading.Lock()
stats = {"messages": 0, "frame": 0, "fps": 0.0, "status": "waiting for mediapipe_osc.py ...", "flash": 0.0}


# Fetch the Hand for an id, creating it if needed, and refresh last_seen.
def get_hand(hid):
    h = hands.get(hid)
    if h is None:
        h = hands[hid] = Hand(hid)
    h.last_seen = time.time()
    return h


# Print an event and keep the last eight in the on-screen log (del events[:-8] drops the oldest).
def log_event(text):
    print(text)
    events.append((time.time(), text))
    del events[:-8]


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
    with lock:
        stats["status"] = f"app {text}"
        if text == "stopped":
            hands.clear()
        log_event(f"mediapipe_osc.py {text}")


# /mp/hand/enter id and /mp/hand/leave id: hands appearing and disappearing.
def on_enter(address, hid):
    with lock:
        hands[hid] = Hand(hid)
        log_event(f"hand {hid} entered")


def on_leave(address, hid):
    with lock:
        hands.pop(hid, None)
        log_event(f"hand {hid} left")


# /mp/hand/info id handedness score: "Left" / "Right".
def on_info(address, hid, handedness, score):
    with lock:
        get_hand(hid).handedness = handedness


# /mp/hand/center and /mp/hand/bbox: where the hand is (to place the label).
def on_center(address, hid, x, y, z):
    with lock:
        get_hand(hid).center = (x, y, z)


def on_bbox(address, hid, *box):
    with lock:
        get_hand(hid).bbox = tuple(box[:4])


# /mp/hand/gesture id name score, every frame for every hand.  The stream of
# names is turned into events by comparing with the previous one; Thumb_Up also
# triggers a reaction (the flash).  Replace the print / flash by your own action.
def on_gesture(address, hid, name, score):
    """The interesting one: a gesture per hand, every frame.  We turn it into events."""
    with lock:
        h = get_hand(hid)
        h.score = score
        if score < MIN_SCORE:
            name = "None"
        if name != h.gesture:                               # the gesture changed -> an event
            h.gesture = name
            log_event(f"hand {hid} ({h.handedness}): {name}  ({score:.2f})")
            if name == "Thumb_Up":
                stats["flash"] = time.time()                # react: flash the screen


# /mp/hand/landmarks id + 21 x (x y z): the skeleton, reshaped to one row per landmark.
def on_landmarks(address, hid, *values):
    if len(values) < 21 * 3:
        return
    with lock:
        get_hand(hid).landmarks = np.array(values[:21 * 3], dtype=np.float32).reshape(21, 3)


# Default handler: every address that was not mapped ends here.
def on_other(address, *args):
    """Everything else (poses, faces, blobs, pinch, velocity) is ignored here."""


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
    d.map("/mp/hand/gesture", counted(on_gesture))
    d.map("/mp/hand/landmarks", counted(on_landmarks))
    d.set_default_handler(counted(on_other))
    return d


# ---------------------------------------------------------------------------- drawing
# Normalized (0..1) -> window pixels.  Used for every coordinate we draw.
def to_px(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


# Draw everything we know.  `snapshot` is a list of Hand objects copied under the
# lock.  `now` is passed in so the flash and the fading log use the same clock.
def draw(img, snapshot, now):
    # max(..., key=score) picks the hand with the best score; default=None when there is no hand.
    # background: the colour of the strongest current gesture, brighter on a flash
    best = max(snapshot, key=lambda h: h.score, default=None)
    bg = GESTURES.get(best.gesture, GESTURES["None"])[0] if best else (20, 20, 20)
    img[:] = bg
    if now - stats["flash"] < 0.3:
        img[:] = (255, 255, 255)

    for h in snapshot:
        col = (255, 255, 255)
        x0, y0 = to_px(h.bbox[0], h.bbox[1])
        x1, y1 = to_px(h.bbox[2], h.bbox[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
        if h.landmarks is not None:
            lm = h.landmarks
            for a, b in CONNECTIONS:
                cv2.line(img, to_px(lm[a, 0], lm[a, 1]), to_px(lm[b, 0], lm[b, 1]), col, 1)
        # The bar length is the score (0..1) times the width of the box.
        # gesture name, icon and a score bar under the hand
        icon = GESTURES.get(h.gesture, GESTURES["None"])[1]
        cv2.putText(img, f"H{h.id} {h.handedness}: {h.gesture} {icon}", (x0, max(16, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y1 + 4), (x0 + int((x1 - x0) * h.score), y1 + 10), col, -1)
        cv2.rectangle(img, (x0, y1 + 4), (x1, y1 + 10), col, 1)

    # Older events are drawn darker (shade decreases 20 per second, never below 80).
    # event log
    for i, (t, text) in enumerate(events[-8:]):
        age = now - t
        shade = int(max(80, 255 - age * 20))
        cv2.putText(img, text, (10, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (shade, shade, shade), 1, cv2.LINE_AA)

    cv2.putText(img, "gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou",
                (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stats['status']}   frame {stats['frame']}   analysis {stats['fps']:.1f} fps   messages {stats['messages']}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------- main
# 1. parse the options; 2. start the OSC server in a background thread (daemon =
# it dies with the program); 3. loop: drop stale objects, draw, show the window,
# read the keyboard, stop after --seconds if asked.
def main(argv=None):
    ap = argparse.ArgumentParser(description="Hand gestures OSC receiver (mediapipe_osc.py --gesture)")
    ap.add_argument("--ip", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5005, help="OSC port (default 5005)")
    ap.add_argument("--headless", action="store_true", help="no window; print a summary (for tests)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = never)")
    ap.add_argument("--snapshot", default="", help="save the last drawn frame to this PNG file")
    args = ap.parse_args(argv)

    server = BlockingOSCUDPServer((args.ip, args.port), make_dispatcher())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening for OSC on {args.ip}:{args.port} - run mediapipe_osc.py --gesture (q or Esc quits)")

    # The image we draw into: HEIGHT x WIDTH pixels x 3 colour channels, 8 bit.
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t0 = time.time()
    seen = set()
    gestures_seen = set()
    try:
        while True:
            now = time.time()
            with lock:
                # A leave message can be lost (UDP gives no guarantee), so a timeout is the safety net.
                for hid in [i for i, h in hands.items() if now - h.last_seen > STALE_SECONDS]:
                    del hands[hid]
                snapshot = list(hands.values())
                seen.update(hands.keys())
                gestures_seen.update(h.gesture for h in snapshot)
            draw(img, snapshot, now)
            # --headless (used by `make test`): no window, just keep the loop alive.
            # Otherwise show the image; waitKey(15) also gives OpenCV time to draw and
            # returns the pressed key (27 = Esc).
            if args.headless:
                time.sleep(0.2)
            else:
                cv2.imshow("hand gestures (OSC)", img)
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
    print(f"done: {n} messages received, hands seen: {sorted(seen)}, gestures seen: {sorted(gestures_seen)}")
    # Exit code 2 = nothing arrived: `make test` uses it to detect a broken setup.
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
