#!/usr/bin/env python3
"""Etapa 7: repone una voz con otro sonido aleatorio al terminar el clip."""

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

    personas_sin_sonido: set[int] = set()
    print(f"Etapa 7: una voz como máximo; escuchando OSC UDP {args.puerto}.")
    print("Al terminar un clip, se asigna otro sonido libre al azar. Ctrl+C para salir.")
    try:
        while True:
            personas = receptor.personas_activas()
            ids_presentes = set(personas)
            personas_sin_sonido.intersection_update(ids_presentes)

            for persona_id in mezclador.personas_con_voz() - ids_presentes:
                mezclador.fundir_persona(persona_id)

            # El mezclador ya liberó el archivo terminado: ahora puede volver a
            # elegirse si no hay otra voz que lo esté usando.
            for finalizacion in mezclador.drenar_finalizaciones():
                personas_sin_sonido.clear()
                persona_id = int(finalizacion["persona_id"])
                if finalizacion["motivo"] != "fin" or persona_id not in personas:
                    continue
                persona = personas[persona_id]
                sonido = banco.elegir(mezclador.nombres_en_uso(), azar)
                if sonido is None:
                    print("No hay un sonido libre para continuar esta voz.")
                    personas_sin_sonido.add(persona_id)
                    continue
                mezclador.agregar_voz(persona_id, sonido)
                mezclador.actualizar_control(
                    persona_id, persona.x, persona.y, persona.velocidad
                )
                print(f"Persona {persona_id}: continúa con {sonido.nombre}.")

            # La primera persona activa ocupa la única voz de esta etapa.
            ordenadas = sorted(personas.values(), key=lambda persona: persona.desde)
            personas_con_voz = mezclador.personas_con_voz()
            for persona in ordenadas:
                if mezclador.cantidad_de_voces() >= MAX_VOCES:
                    break
                if persona.id in personas_con_voz or persona.id in personas_sin_sonido:
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
