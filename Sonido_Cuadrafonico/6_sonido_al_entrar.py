#!/usr/bin/env python3
"""Etapa 6: una persona nueva inicia un sonido aleatorio."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import sounddevice as sd

from manipulacion_sonido import BancoSonidos, Mezclador
from recepcion_datos import ReceptorDatosOSC


MAX_VOCES = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=("stereo", "cuadrafonico"), default="stereo")
    parser.add_argument("--sonidos", type=Path, default=Path(__file__).resolve().parent / "sonidos")
    parser.add_argument("--puerto", type=int, default=12000)
    parser.add_argument("--dispositivo", type=int, default=None)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--listar-dispositivos", action="store_true")
    args = parser.parse_args()
    if args.listar_dispositivos:
        print(sd.query_devices())
        return

    try:
        banco = BancoSonidos(args.sonidos)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    azar = random.Random(args.semilla)
    receptor = ReceptorDatosOSC(puerto=args.puerto)
    mezclador = Mezclador(modo=args.modo, dispositivo=args.dispositivo)
    try:
        mezclador.iniciar()
        receptor.iniciar()
    except RuntimeError as exc:
        mezclador.detener()
        raise SystemExit(str(exc)) from exc

    # En este paso un sonido termina y no se repone para la misma persona.
    sonidos_finalizados: set[int] = set()
    personas_sin_sonido: set[int] = set()
    print(f"Etapa 6: una voz como máximo; escuchando OSC UDP {args.puerto}.")
    print("Una persona nueva inicia un sonido al azar. Ctrl+C para salir.")
    try:
        while True:
            personas = receptor.personas_activas()
            ids_presentes = set(personas)
            sonidos_finalizados.intersection_update(ids_presentes)
            personas_sin_sonido.intersection_update(ids_presentes)

            # Si una persona deja de enviar posiciones, su sonido se desvanece.
            for persona_id in mezclador.personas_con_voz() - ids_presentes:
                mezclador.fundir_persona(persona_id)

            # En esta etapa solo registramos que el sonido de esa persona terminó.
            for finalizacion in mezclador.drenar_finalizaciones():
                personas_sin_sonido.clear()
                if finalizacion["motivo"] == "fin":
                    sonidos_finalizados.add(int(finalizacion["persona_id"]))

            # Mientras haya lugar, asignamos un sonido libre a una persona activa.
            ordenadas = sorted(personas.values(), key=lambda persona: persona.desde)
            personas_con_voz = mezclador.personas_con_voz()
            for persona in ordenadas:
                if mezclador.cantidad_de_voces() >= MAX_VOCES:
                    break
                if persona.id in personas_con_voz:
                    continue
                if persona.id in sonidos_finalizados or persona.id in personas_sin_sonido:
                    continue

                sonido = banco.elegir(mezclador.nombres_en_uso(), azar)
                if sonido is None:
                    print("No hay un archivo de audio libre para esta voz.")
                    personas_sin_sonido.add(persona.id)
                    continue
                mezclador.agregar_voz(persona.id, sonido)
                mezclador.actualizar_control(
                    persona.id, persona.x, persona.y, persona.velocidad
                )
                personas_con_voz = mezclador.personas_con_voz()
                print(f"Persona {persona.id}: empieza {sonido.nombre}.")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        receptor.detener()
        mezclador.detener()


if __name__ == "__main__":
    main()
