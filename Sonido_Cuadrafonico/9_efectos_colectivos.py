#!/usr/bin/env python3
"""Etapa 9: las personas extra influyen en efectos sin sumar voces."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import sounddevice as sd

from manipulacion_sonido import BancoSonidos, Mezclador
from recepcion_datos import ReceptorDatosOSC


MAX_VOCES = 4


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
    ultimo_estado = 0.0
    print(f"Etapa 9: máximo {MAX_VOCES} voces; escuchando OSC UDP {args.puerto}.")
    print("Las personas extra modifican los efectos, sin iniciar otra voz. Ctrl+C para salir.")
    try:
        while True:
            personas = receptor.personas_activas()
            ids_presentes = set(personas)
            personas_sin_sonido.intersection_update(ids_presentes)

            # Una voz que pierde su persona baja el volumen y libera su sonido.
            for persona_id in mezclador.personas_con_voz() - ids_presentes:
                mezclador.fundir_persona(persona_id)

            # Cuando termina un clip, se elige otro que no esté sonando.
            for finalizacion in mezclador.drenar_finalizaciones():
                personas_sin_sonido.clear()
                persona_id = int(finalizacion["persona_id"])
                if finalizacion["motivo"] != "fin" or persona_id not in personas:
                    continue
                persona = personas[persona_id]
                sonido = banco.elegir(mezclador.nombres_en_uso(), azar)
                if sonido is None:
                    print("No hay un sonido libre para reponer esta voz.")
                    personas_sin_sonido.add(persona_id)
                    continue
                mezclador.agregar_voz(persona_id, sonido)
                mezclador.actualizar_control(
                    persona_id, persona.x, persona.y, persona.velocidad
                )
                print(f"Persona {persona_id}: continúa con {sonido.nombre}.")

            # Las primeras cuatro personas por orden de llegada pueden tener sonido.
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

            # La quinta persona en adelante aporta a un control colectivo.
            # Su velocidad media aumenta la saturación y la reverb de las voces.
            personas_extra = ordenadas[MAX_VOCES:]
            cantidad_extra = len(personas_extra)
            actividad_colectiva = (
                sum(persona.velocidad for persona in personas_extra) / cantidad_extra
                if cantidad_extra else 0.0
            )
            for persona_id, persona in personas.items():
                mezclador.actualizar_control(
                    persona_id,
                    persona.x,
                    persona.y,
                    persona.velocidad,
                    personas_extra=cantidad_extra,
                    actividad_colectiva=actividad_colectiva,
                )

            ahora = time.monotonic()
            if ahora - ultimo_estado >= 1.0:
                print(
                    f"Personas: {len(personas)} | voces: "
                    f"{mezclador.cantidad_de_voces()}/{MAX_VOCES} | "
                    f"personas extra: {cantidad_extra}"
                )
                ultimo_estado = ahora
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        receptor.detener()
        mezclador.detener()


if __name__ == "__main__":
    main()
