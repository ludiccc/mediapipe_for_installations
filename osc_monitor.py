#!/usr/bin/env python3
"""Print the OSC messages arriving on a port - handy to check what
mediapipe_osc.py is sending before wiring up Processing / TouchDesigner / Pd.

    python osc_monitor.py                 # listen on 5005
    python osc_monitor.py --port 12000    # Sonido_Cuadrafonico port
    python osc_monitor.py --filter /mp/pose/center --rate
    python osc_monitor.py --filter /mp/blob       # only the Blobs detector

--rate prints a per-address message count once per second instead of every
message (useful with the long landmark lists).
"""

from __future__ import annotations

import argparse
import collections
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return repr(value) if isinstance(value, str) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="0.0.0.0", help="interface to bind (0.0.0.0 = all)")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--filter", default="", help="only show addresses starting with this")
    parser.add_argument("--max-args", type=int, default=12, help="truncate long argument lists")
    parser.add_argument("--rate", action="store_true", help="print message rates per address instead")
    args = parser.parse_args()

    counts: collections.Counter = collections.Counter()
    lock = threading.Lock()

    def handler(address: str, *values) -> None:
        if args.filter and not address.startswith(args.filter):
            return
        if args.rate:
            with lock:
                counts[address] += 1
            return
        shown = " ".join(fmt(v) for v in values[: args.max_args])
        if len(values) > args.max_args:
            shown += f" ... ({len(values)} args)"
        with lock:  # handlers run on several threads
            print(f"{address:32s} {shown}", flush=True)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handler)
    server = ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    print(f"listening for OSC on {args.ip}:{args.port}  (Ctrl+C to stop)")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(1.0)
            if args.rate:
                with lock:
                    snapshot = dict(counts)
                    counts.clear()
                if snapshot:
                    print("--- messages/s ---")
                    for address in sorted(snapshot):
                        print(f"{snapshot[address]:5d}  {address}")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
