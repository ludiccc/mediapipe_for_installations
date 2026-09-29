#!/usr/bin/env python3
"""Simula por OSC la secuencia de entradas y salidas de personas."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

from pythonosc.udp_client import SimpleUDPClient


SEGUNDOS_POR_ESTADO = 5.0
FRECUENCIA_ENVIO = 30.0


@dataclass(frozen=True)
class PersonaSimulada:
    id: int
    fase: float


def crear_estados(ciclo: int) -> list[tuple[str, list[PersonaSimulada]]]:
    """Arma los estados solicitados; los IDs avanzan como en Processing."""
    base = ciclo * 6
    persona_a = PersonaSimulada(base + 1, 0.0)
    persona_a_vuelve = PersonaSimulada(base + 2, 0.8)
    persona_b = PersonaSimulada(base + 3, 1.7)
    persona_c = PersonaSimulada(base + 4, 2.6)
    persona_d = PersonaSimulada(base + 5, 3.5)
    persona_e = PersonaSimulada(base + 6, 4.4)

    return [
        ("Entra A", [persona_a]),
        ("Sale A", []),
        ("A vuelve a entrar (nuevo ID)", [persona_a_vuelve]),
        ("Entra B", [persona_a_vuelve, persona_b]),
        ("Sale A; B permanece", [persona_b]),
        ("Entra C", [persona_b, persona_c]),
        ("Entra D", [persona_b, persona_c, persona_d]),
        ("Entra E: quedan cuatro personas", [persona_b, persona_c, persona_d, persona_e]),
        ("Sale B", [persona_c, persona_d, persona_e]),
        ("Sale C", [persona_d, persona_e]),
        ("Sale D", [persona_e]),
        ("Sale E; campo vacío", []),
    ]


def enviar_estado(
    cliente: SimpleUDPClient,
    nombre: str,
    personas: list[PersonaSimulada],
    inicio: float,
) -> None:
    """Mantiene el estado cinco segundos y actualiza posiciones a 30 Hz."""
    print(
        f"{nombre:<38} personas={len(personas)} duración={SEGUNDOS_POR_ESTADO:.0f}s",
        flush=True,
    )
    siguiente_envio = time.monotonic()
    while time.monotonic() - inicio < SEGUNDOS_POR_ESTADO:
        ahora = time.monotonic()
        tiempo = ahora - inicio
        for persona in personas:
            fase = tiempo * 0.55 + persona.fase
            x = 0.5 + 0.42 * math.sin(fase)
            y = 0.5 + 0.40 * math.cos(fase * 0.73)
            velocidad = min(1.0, abs(0.42 * 0.55 * math.cos(fase)))
            cliente.send_message(
                "/kinect/person", [persona.id, x, y, velocidad]
            )

        siguiente_envio += 1.0 / FRECUENCIA_ENVIO
        demora = siguiente_envio - time.monotonic()
        if demora > 0:
            time.sleep(demora)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Equipo que recibe OSC.")
    parser.add_argument("--puerto", type=int, default=12000, help="Puerto OSC de destino.")
    args = parser.parse_args()

    cliente = SimpleUDPClient(args.host, args.puerto)
    print(f"Enviando /kinect/person a {args.host}:{args.puerto}.")
    print("Cada estado dura cinco segundos. Ctrl+C para detener.")

    ciclo = 0
    try:
        while True:
            for nombre, personas in crear_estados(ciclo):
                enviar_estado(cliente, nombre, personas, time.monotonic())
            ciclo += 1
    except KeyboardInterrupt:
        print("\nSimulación detenida.")


if __name__ == "__main__":
    main()
