#!/usr/bin/env python3
"""La posición y velocidad de una persona modifican los efectos del sonido."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from manipulacion_sonido import Mezclador
from recepcion_datos import ReceptorDatosOSC
from soporte_ejemplos import agregar_argumentos_audio, crear_banco, listar_dispositivos_si_corresponde


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    agregar_argumentos_audio(parser)
    args = parser.parse_args()
    if listar_dispositivos_si_corresponde(args):
        return

    banco = crear_banco(args)
    sonido = replace(banco.cargar(banco.sonidos[0]), movimiento="persona")
    mezclador = Mezclador(modo=args.modo, dispositivo=args.dispositivo)
    receptor = ReceptorDatosOSC(puerto=args.puerto)
    mezclador.agregar_voz(-1, sonido)
    try:
        mezclador.iniciar()
        receptor.iniciar()
    except RuntimeError as exc:
        mezclador.detener()
        raise SystemExit(str(exc)) from exc
    print("X cambia el pitch, Y el filtro/reverb y la velocidad la saturación. Ctrl+C para salir.")
    try:
        while True:
            personas = sorted(receptor.personas_activas().values(), key=lambda p: p.desde)
            if personas:
                p = personas[0]
                mezclador.actualizar_control(-1, p.x, p.y, p.velocidad)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        receptor.detener()
        mezclador.detener()


if __name__ == "__main__":
    main()
