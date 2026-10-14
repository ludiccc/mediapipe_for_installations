#!/usr/bin/env python3
"""Instalación final: Kinect/OSC, selección aleatoria y hasta cuatro voces."""

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
    parser.add_argument(
        "--modo", choices=("stereo", "cuadrafonico"), default="stereo",
        help="Usa stereo para dos salidas o cuadrafonico para cuatro.",
    )
    parser.add_argument(
        "--sonidos", type=Path,
        default=Path(__file__).resolve().parent / "sonidos",
        help="Carpeta que contiene los archivos MP3 o WAV.",
    )
    parser.add_argument("--puerto", type=int, default=12000, help="Puerto OSC de entrada.")
    parser.add_argument("--dispositivo", type=int, default=None, help="Índice de salida de audio.")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla opcional para repetir el azar.")
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
    print(f"Instalación en modo {args.modo}; escuchando OSC UDP {args.puerto}.")
    print(f"Sonidos disponibles: {len(banco.sonidos)}; máximo de voces simultáneas: {MAX_VOCES}.")
    print("Procesando. Ctrl+C para salir.")

    try:
        while True:
            # El receptor mantiene las posiciones OSC recientes. Si una persona
            # deja de enviar datos durante un segundo, desaparece de este mapa.
            personas = receptor.personas_activas()
            ids_presentes = set(personas)
            personas_sin_sonido.intersection_update(ids_presentes)

            # Al salir una persona, desvanece su audio en vez de cortarlo de golpe.
            for persona_id in mezclador.personas_con_voz() - ids_presentes:
                mezclador.fundir_persona(persona_id)

            # El audio informa cuando termina un archivo. Al reponerlo, el archivo
            # terminado ya no está en la lista de los sonidos que están en uso.
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

            # La antigüedad conserva el orden de entrada. Se inicia una voz por
            # persona hasta llenar cuatro lugares; las demás no reciben sonidos.
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

            # Las personas posteriores a las primeras cuatro modifican los
            # efectos de las voces existentes, pero no agregan nuevas voces.
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
