"""Recepción de personas enviadas por OSC desde Processing.

Protocolo de entrada:
    /kinect/person id x y velocidad

ID es entero; X, Y y velocidad son floats normalizados entre 0 y 1. Una
persona se considera fuera del campo si deja de enviar datos por un rato.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


@dataclass(frozen=True)
class Persona:
    id: int
    x: float
    y: float
    velocidad: float
    desde: float
    ultima_actualizacion: float


class ReceptorDatosOSC:
    """Guarda las posiciones recientes y comunica entradas y salidas."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        puerto: int = 12000,
        tiempo_sin_datos: float = 1.0,
    ) -> None:
        self.host = host
        self.puerto = puerto
        self.tiempo_sin_datos = tiempo_sin_datos
        self._personas: dict[int, Persona] = {}
        self._eventos: queue.Queue = queue.Queue(maxsize=512)
        self._candado = threading.Lock()
        self._servidor: BlockingOSCUDPServer | None = None
        self._hilo: threading.Thread | None = None

        dispatcher = Dispatcher()
        dispatcher.map("/kinect/person", self._recibir_persona)
        self._dispatcher = dispatcher

    def iniciar(self) -> None:
        """Abre el puerto OSC e inicia el servidor en segundo plano."""
        if self._hilo is not None:
            return
        try:
            self._servidor = BlockingOSCUDPServer(
                (self.host, self.puerto), self._dispatcher
            )
        except OSError as exc:
            raise RuntimeError(f"No se pudo abrir el puerto OSC {self.puerto}: {exc}") from exc
        self._hilo = threading.Thread(
            target=self._servidor.serve_forever, name="receptor-osc", daemon=True
        )
        self._hilo.start()

    def detener(self) -> None:
        """Cierra el servidor OSC si estaba iniciado."""
        servidor = self._servidor
        if servidor is None:
            return
        servidor.shutdown()
        servidor.server_close()
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
        self._servidor = None
        self._hilo = None

    def personas_activas(self) -> dict[int, Persona]:
        """Devuelve una copia de las personas que actualizaron su posición."""
        ahora = time.monotonic()
        with self._candado:
            self._caducar_personas(ahora)
            return dict(self._personas)

    def drenar_eventos(self) -> list[tuple[str, int]]:
        """Devuelve eventos nuevos como pares ``("entra"|"sale", id)``."""
        self.personas_activas()  # También detecta salidas por pérdida de señal.
        eventos: list[tuple[str, int]] = []
        while True:
            try:
                eventos.append(self._eventos.get_nowait())
            except queue.Empty:
                return eventos

    def _recibir_persona(self, direccion: str, *valores: object) -> None:
        if len(valores) != 4:
            return
        try:
            identificador_numero = float(valores[0])
            x = float(valores[1])
            y = float(valores[2])
            velocidad = float(valores[3])
        except (TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(valor) for valor in (identificador_numero, x, y, velocidad)):
            return
        identificador = int(identificador_numero)
        if identificador != identificador_numero:
            return

        ahora = time.monotonic()
        with self._candado:
            self._caducar_personas(ahora)
            anterior = self._personas.get(identificador)
            persona = Persona(
                id=identificador,
                x=max(0.0, min(1.0, x)),
                y=max(0.0, min(1.0, y)),
                velocidad=max(0.0, min(1.0, velocidad)),
                desde=anterior.desde if anterior is not None else ahora,
                ultima_actualizacion=ahora,
            )
            self._personas[identificador] = persona
            if anterior is None:
                self._registrar_evento(("entra", identificador))

    def _caducar_personas(self, ahora: float) -> None:
        vencidas = [
            identificador
            for identificador, persona in self._personas.items()
            if ahora - persona.ultima_actualizacion > self.tiempo_sin_datos
        ]
        for identificador in vencidas:
            del self._personas[identificador]
            self._registrar_evento(("sale", identificador))

    def _registrar_evento(self, evento: tuple[str, int]) -> None:
        """Evita que los eventos sin consumidor ocupen memoria indefinidamente."""
        try:
            self._eventos.put_nowait(evento)
        except queue.Full:
            try:
                self._eventos.get_nowait()
            except queue.Empty:
                pass
            self._eventos.put_nowait(evento)
