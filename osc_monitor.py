#!/usr/bin/env python3
"""Muestra los mensajes OSC que llegan a un puerto.

Print the OSC messages arriving on a port. It is useful for checking what
mediapipe_osc.py sends before connecting Processing, TouchDesigner or Pd.

Ejemplos / examples:
    python osc_monitor.py                 # listen on 5005
    python osc_monitor.py --port 12000    # puerto / port of Sonido_Cuadrafonico
    python osc_monitor.py --filter /mp/pose/center --rate
    python osc_monitor.py --filter /mp/blob       # solo Blobs / Blobs only

--rate muestra una cantidad de mensajes por dirección cada segundo, en vez de
cada mensaje; es útil con las listas largas de landmarks. It prints a
per-address message count once per second instead of every message.
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
    parser.add_argument("--ip", default="0.0.0.0", help="interfaz donde escuchar / interface to bind (0.0.0.0 = todas / all)")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--filter", default="", help="solo direcciones que empiezan así / only addresses starting with this")
    parser.add_argument("--max-args", type=int, default=12, help="recorta listas largas de argumentos / truncate long argument lists")
    parser.add_argument("--rate", action="store_true", help="muestra tasas por dirección / print message rates per address")
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
        with lock:  # los handlers corren en varios hilos / handlers run on several threads
            print(f"{address:32s} {shown}", flush=True)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handler)
    server = ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
    print(f"escuchando OSC / listening for OSC on {args.ip}:{args.port}  (Ctrl+C para detener / to stop)")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(1.0)
            if args.rate:
                with lock:
                    snapshot = dict(counts)
                    counts.clear()
                if snapshot:
                    print("--- mensajes/s | messages/s ---")
                    for address in sorted(snapshot):
                        print(f"{snapshot[address]:5d}  {address}")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
