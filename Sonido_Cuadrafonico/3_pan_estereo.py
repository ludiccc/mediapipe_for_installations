#!/usr/bin/env python3
"""Un sonido sigue la posición OSC de la primera persona detectada."""

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
    print("La posición X mueve el sonido a izquierda/derecha (también en modo estéreo). Ctrl+C para salir.")
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
