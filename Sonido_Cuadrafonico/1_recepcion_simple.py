#!/usr/bin/env python3
"""Muestra en la terminal las posiciones recibidas por OSC."""

from __future__ import annotations

import argparse
import time

from recepcion_datos import ReceptorDatosOSC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puerto", type=int, default=12000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    receptor = ReceptorDatosOSC(host=args.host, puerto=args.puerto)
    try:
        receptor.iniciar()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Esperando /kinect/person en UDP {args.puerto}. Ctrl+C para salir.")
    try:
        while True:
            personas = sorted(
                receptor.personas_activas().values(), key=lambda persona: persona.id
            )
            if personas:
                datos = " | ".join(
                    f"id={p.id} x={p.x:.3f} y={p.y:.3f} velocidad={p.velocidad:.3f}"
                    for p in personas
                )
                print(f"\r{datos:<110}", end="", flush=True)
            else:
                print("\rSin personas. Esperando datos OSC...".ljust(110), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nReceptor detenido.")
    finally:
        receptor.detener()


if __name__ == "__main__":
    main()
