#!/usr/bin/env python3
"""Blobs receiver — Python + python-osc + OpenCV.

Listens to the OSC messages of mediapipe_osc.py for the **Blobs** detector
(OpenCV shapes that stand out from the background — people seen from a
ceiling camera, objects on a table, silhouettes) and draws every blob: its
outline, bounding box, centroid, velocity arrow, area, orientation and id.
The floor is divided into a 3 x 3 grid of zones and the example tells which
zone every blob is in — the typical "something happens where the visitor
stands" logic of an installation.

    python ../../mediapipe_osc.py --blobs                # terminal 1 (ceiling camera; empty scene at start)
    python blobs.py                                      # terminal 2 (this)

    python blobs.py --port 9000
    python blobs.py --headless --seconds 5               # no window (tests)

Messages used (first argument = stable id of the blob):

    /mp/blobs           count id id ...          blobs on screen, biggest first (every frame)
    /mp/blob/enter      id
    /mp/blob/leave      id
    /mp/blob/center     id x y 0                 centroid
    /mp/blob/bbox       id xmin ymin xmax ymax
    /mp/blob/velocity   id vx vy speed
    /mp/blob/info       id area angle length width    area = fraction of the image (0..1),
                                                      angle = main axis in degrees (-90..90)
    /mp/blob/contour    id 16 x (x y)            outline, from the topmost point
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
# The picture is divided into GRID x GRID cells ("zones").  With a ceiling camera
# a zone is a part of the floor: the installation can react to *where* a visitor
# stands without any geometry beyond a division.
GRID = 3                           # zones per side


# --- the data model --------------------------------------------------------------
# One object per blob, by stable id.  A blob is a shape that stands out from the
# learned background - with a ceiling camera, usually one person.  `zone`
# remembers the last cell so that a change becomes an event.
class Blob:
    def __init__(self, bid):
        self.id = bid
        self.center = (0.5, 0.5)
        self.bbox = (0.0, 0.0, 0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.area = 0.0
        self.angle = 0.0
        self.length = 0.0
        self.width = 0.0
        self.contour = None            # (16, 2)
        self.zone = None               # (column, row)
        self.last_seen = time.time()

    # The cell (column, row) that contains the centroid: int(x * GRID), clamped to the grid.
    def zone_of(self):
        return (min(GRID - 1, max(0, int(self.center[0] * GRID))),
                min(GRID - 1, max(0, int(self.center[1] * GRID))))


# --- shared state ----------------------------------------------------------------
# Written by the OSC handlers (server thread), read by draw() (main thread),
# always under the lock.  `order` is the list of ids sorted by size that arrives
# every frame in /mp/blobs.
blobs = {}
order = []                  # ids biggest first, from /mp/blobs
lock = threading.Lock()
stats = {"messages": 0, "frame": 0, "fps": 0.0, "status": "waiting for mediapipe_osc.py ..."}


# Fetch the Blob for an id, creating it if needed, and refresh last_seen.
def get_blob(bid):
    b = blobs.get(bid)
    if b is None:
        b = blobs[bid] = Blob(bid)
    b.last_seen = time.time()
    return b


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
            blobs.clear()
            order[:] = []


# /mp/blobs count id id ...: every frame, the ids of the blobs on screen, biggest
# first.  order[:] = ... replaces the contents of the shared list in place.
def on_blobs(address, count, *ids):
    with lock:
        order[:] = list(ids[:count])


# /mp/blob/enter id and /mp/blob/leave id: blobs appearing and disappearing.
def on_enter(address, bid):
    with lock:
        blobs[bid] = Blob(bid)
    print(f"blob {bid} entered")


def on_leave(address, bid):
    with lock:
        blobs.pop(bid, None)
    print(f"blob {bid} left")


# /mp/blob/center id x y 0: the centroid.  We work out its zone and print only
# when it differs from the previous one: a position stream becomes a "changed
# zone" event.
def on_center(address, bid, x, y, z):
    with lock:
        b = get_blob(bid)
        b.center = (x, y)
        zone = b.zone_of()
        if zone != b.zone:                            # moved to another zone -> an event
            b.zone = zone
            print(f"blob {bid} is now in zone column {zone[0]}, row {zone[1]}")


# /mp/blob/bbox id xmin ymin xmax ymax: the rectangle around the blob.
def on_bbox(address, bid, *box):
    with lock:
        get_blob(bid).bbox = tuple(box[:4])


# /mp/blob/velocity id vx vy speed: movement of the centroid, in picture sizes per second.
def on_velocity(address, bid, vx, vy, speed):
    with lock:
        b = get_blob(bid)
        b.velocity, b.speed = (vx, vy), speed


# /mp/blob/info id area angle length width: area as a fraction of the picture
# (0.01 = 1 %), the angle of the main axis in degrees, and the length / width of
# the equivalent ellipse (normalized).
def on_info(address, bid, area, angle, length, width):
    with lock:
        b = get_blob(bid)
        b.area, b.angle, b.length, b.width = area, angle, length, width


# /mp/blob/contour id + 16 x (x y): the outline, 16 points starting at the top.
# reshape(n, 2) gives one row per point.
def on_contour(address, bid, *values):
    n = len(values) // 2
    if n < 3:
        return
    with lock:
        get_blob(bid).contour = np.array(values[:n * 2], dtype=np.float32).reshape(n, 2)


# Default handler: every address that was not mapped ends here.
def on_other(address, *args):
    """Poses, hands, faces ... are ignored here."""


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
    d.map("/mp/blobs", counted(on_blobs))
    d.map("/mp/blob/enter", counted(on_enter))
    d.map("/mp/blob/leave", counted(on_leave))
    d.map("/mp/blob/center", counted(on_center))
    d.map("/mp/blob/bbox", counted(on_bbox))
    d.map("/mp/blob/velocity", counted(on_velocity))
    d.map("/mp/blob/info", counted(on_info))
    d.map("/mp/blob/contour", counted(on_contour))
    d.set_default_handler(counted(on_other))
    return d


# ---------------------------------------------------------------------------- drawing
# Normalized (0..1) -> window pixels.  Used for every coordinate we draw.
def to_px(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


# A colour per id (OpenCV uses BGR and a 0..179 hue).
def color_for(bid):
    hue = (bid * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Draw everything we know.  `snapshot` (Blob objects) and `order_now` (ids by
# size) are copies made under the lock, so the server thread can keep writing.
def draw(img, snapshot, order_now):
    img[:] = (20, 20, 20)
    # zones: light up the cells that contain a blob
    # A set of the cells that contain a blob: those are filled a little lighter.
    occupied = {b.zone_of() for b in snapshot}
    cw, ch = WIDTH // GRID, HEIGHT // GRID
    for col in range(GRID):
        for row in range(GRID):
            x, y = col * cw, row * ch
            if (col, row) in occupied:
                cv2.rectangle(img, (x, y), (x + cw, y + ch), (45, 45, 70), -1)
            cv2.rectangle(img, (x, y), (x + cw, y + ch), (60, 60, 60), 1)

    for b in snapshot:
        col = color_for(b.id)
        if b.contour is not None:
            # fillPoly / polylines want an integer array of points: a filled, outlined contour.
            pts = np.array([to_px(x, y) for x, y in b.contour], dtype=np.int32)
            cv2.fillPoly(img, [pts], (col[0] // 3, col[1] // 3, col[2] // 3))
            cv2.polylines(img, [pts], True, col, 2)
        x0, y0 = to_px(b.bbox[0], b.bbox[1])
        x1, y1 = to_px(b.bbox[2], b.bbox[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        cx, cy = to_px(b.center[0], b.center[1])
        cv2.circle(img, (cx, cy), 6, col, -1)
        # main axis (angle is clockwise on screen, length normalized)
        ang = np.radians(b.angle)
        dx, dy = np.cos(ang) * b.length / 2 * WIDTH, np.sin(ang) * b.length / 2 * HEIGHT
        cv2.line(img, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (255, 255, 255), 1)
        cv2.arrowedLine(img, (cx, cy), (int(cx + b.velocity[0] * WIDTH * 0.25), int(cy + b.velocity[1] * HEIGHT * 0.25)), col, 2)

        # Position in the biggest-first list (1 = the biggest blob on screen).
        rank = order_now.index(b.id) + 1 if b.id in order_now else "?"
        label = f"B{b.id}  #{rank}  {b.area * 100:.1f}%  {b.angle:+.0f} deg  zone {b.zone_of()}"
        cv2.putText(img, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(img, f"blobs: {len(snapshot)}   biggest first: {' '.join(str(i) for i in order_now)}",
                (10, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{stats['status']}   frame {stats['frame']}   analysis {stats['fps']:.1f} fps   messages {stats['messages']}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------- main
# 1. parse the options; 2. start the OSC server in a background thread (daemon =
# it dies with the program); 3. loop: drop stale objects, draw, show the window,
# read the keyboard, stop after --seconds if asked.
def main(argv=None):
    ap = argparse.ArgumentParser(description="Blobs OSC receiver (mediapipe_osc.py --blobs)")
    ap.add_argument("--ip", default="0.0.0.0", help="interface to listen on (default: all)")
    ap.add_argument("--port", type=int, default=5005, help="OSC port (default 5005)")
    ap.add_argument("--headless", action="store_true", help="no window; print a summary (for tests)")
    ap.add_argument("--seconds", type=float, default=0.0, help="quit after this many seconds (0 = never)")
    ap.add_argument("--snapshot", default="", help="save the last drawn frame to this PNG file")
    args = ap.parse_args(argv)

    server = BlockingOSCUDPServer((args.ip, args.port), make_dispatcher())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"listening for OSC on {args.ip}:{args.port} - run mediapipe_osc.py --blobs (q or Esc quits)")

    # The image we draw into: HEIGHT x WIDTH pixels x 3 colour channels, 8 bit.
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    t0 = time.time()
    seen = set()
    try:
        while True:
            now = time.time()
            with lock:
                # A leave message can be lost (UDP gives no guarantee), so a timeout is the safety net.
                for bid in [i for i, b in blobs.items() if now - b.last_seen > STALE_SECONDS]:
                    del blobs[bid]
                snapshot = list(blobs.values())
                order_now = list(order)
                seen.update(blobs.keys())
            draw(img, snapshot, order_now)
            # --headless (used by `make test`): no window, just keep the loop alive.
            # Otherwise show the image; waitKey(15) also gives OpenCV time to draw and
            # returns the pressed key (27 = Esc).
            if args.headless:
                time.sleep(0.2)
            else:
                cv2.imshow("blobs (OSC)", img)
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
    print(f"done: {n} messages received, blobs seen: {sorted(seen)}")
    # Exit code 2 = nothing arrived: `make test` uses it to detect a broken setup.
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
