"""Funciones espaciales pequeñas para estéreo y cuadrafonía.

Las posiciones que reciben estas funciones usan -1..1: X va de izquierda a
derecha e Y de frente a fondo. En estéreo solo se usa X.
"""

from __future__ import annotations

import math


MODOS_SALIDA = ("stereo", "cuadrafonico")
MOVIMIENTOS = ("orbita", "vaiven", "diagonal", "persona")


def limitar(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, float(valor)))


def ganancias_espaciales(x: float, y: float, modo: str = "stereo") -> tuple[float, ...]:
    """Devuelve una ganancia por parlante, con potencia aproximadamente constante.

    Orden cuadrafónico: frente izquierda, frente derecha, fondo derecha,
    fondo izquierda. En estéreo: izquierda, derecha.
    """
    x = limitar(x, -1.0, 1.0)
    y = limitar(y, -1.0, 1.0)
    if modo == "stereo":
        angulo = (x + 1.0) * math.pi / 4.0
        return math.cos(angulo), math.sin(angulo)
    if modo != "cuadrafonico":
        raise ValueError("El modo debe ser 'stereo' o 'cuadrafonico'.")

    u = (x + 1.0) * 0.5
    v = (y + 1.0) * 0.5
    ganancias = [
        (1.0 - u) * (1.0 - v),
        u * (1.0 - v),
        u * v,
        (1.0 - u) * v,
    ]
    potencia = math.sqrt(sum(ganancia * ganancia for ganancia in ganancias))
    if potencia > 0.0:
        ganancias = [ganancia / potencia for ganancia in ganancias]
    return tuple(ganancias)


def posicion_de_movimiento(
    movimiento: str, tiempo: float, x_persona: float = 0.0, y_persona: float = 0.0
) -> tuple[float, float]:
    """Calcula la posición de un sonido según su recorrido asignado.

    ``tiempo`` está en segundos. Los recorridos son intencionalmente simples
    para que se puedan modificar durante el ejercicio.
    """
    fase = 2.0 * math.pi * tiempo / 12.0
    x_persona = limitar(x_persona, -1.0, 1.0)
    y_persona = limitar(y_persona, -1.0, 1.0)

    if movimiento == "orbita":
        return 0.72 * math.cos(fase), 0.72 * math.sin(fase)
    if movimiento == "vaiven":
        return 0.8 * math.sin(fase), 0.0
    if movimiento == "diagonal":
        return 0.72 * math.sin(fase), 0.72 * math.sin(fase + math.pi)
    if movimiento == "persona":
        return x_persona, y_persona
    raise ValueError(f"Movimiento desconocido: {movimiento}")


def descripcion_salidas(modo: str) -> str:
    if modo == "stereo":
        return "2 canales: izquierda, derecha"
    if modo == "cuadrafonico":
        return "4 canales: frente izquierda, frente derecha, fondo derecha, fondo izquierda"
    raise ValueError("El modo debe ser 'stereo' o 'cuadrafonico'.")
