#!/usr/bin/env python3
"""Reproduce un sonido con el recorrido espacial asignado en el banco."""

from __future__ import annotations

import argparse
import time

from manipulacion_sonido import Mezclador
from soporte_ejemplos import agregar_argumentos_audio, crear_banco, listar_dispositivos_si_corresponde


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    agregar_argumentos_audio(parser)
    args = parser.parse_args()
    if listar_dispositivos_si_corresponde(args):
        return

    banco = crear_banco(args)
    sonido = banco.cargar(banco.sonidos[0])
    mezclador = Mezclador(modo=args.modo, dispositivo=args.dispositivo)
    mezclador.agregar_voz(-1, sonido)
    try:
        mezclador.iniciar()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{sonido.nombre}: movimiento '{sonido.movimiento}'. Ctrl+C para salir.")
    try:
        while mezclador.cantidad_de_voces():
            time.sleep(0.1)
        print("El sonido terminó.")
    except KeyboardInterrupt:
        pass
    finally:
        mezclador.detener()


if __name__ == "__main__":
    main()
