"""Lista, mezcla y modifica sonidos en tiempo real.

MP3 y WAV se decodifican con ffmpeg y se reproducen con sounddevice. El motor
acepta varias voces; cada voz puede tener pitch, filtro, saturación, reverb y
posición espacial propios.
"""

from __future__ import annotations

import math
import queue
import random
import subprocess
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import sounddevice as sd

from cuadrafonia import MOVIMIENTOS, ganancias_espaciales, posicion_de_movimiento


FRECUENCIA_MUESTREO = 48_000
FORMATOS_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".aiff", ".aif"}


@dataclass(frozen=True)
class Sonido:
    nombre: str
    ruta: Path
    muestras: np.ndarray | None
    movimiento: str


class BancoSonidos:
    """Lista los audios y decodifica cada uno cuando va a reproducirse."""

    def __init__(self, carpeta: Path, frecuencia: int = FRECUENCIA_MUESTREO) -> None:
        self.carpeta = Path(carpeta)
        self.frecuencia = frecuencia
        rutas = sorted(
            ruta for ruta in self.carpeta.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in FORMATOS_AUDIO
        ) if self.carpeta.is_dir() else []
        self.sonidos = [
            Sonido(
                nombre=ruta.name,
                ruta=ruta,
                muestras=None,
                movimiento=MOVIMIENTOS[indice % len(MOVIMIENTOS)],
            )
            for indice, ruta in enumerate(rutas)
        ]
        if not self.sonidos:
            raise FileNotFoundError(
                f"No hay archivos MP3/WAV en {self.carpeta}. "
                "Copia allí los sonidos de la instalación."
            )

    def elegir(self, nombres_en_uso: set[str], azar: random.Random) -> Sonido | None:
        """Elige y decodifica un sonido que no esté reproduciéndose ahora."""
        disponibles = [
            sonido for sonido in self.sonidos if sonido.nombre not in nombres_en_uso
        ]
        if not disponibles:
            return None
        elegido = azar.choice(disponibles)
        return replace(elegido, muestras=decodificar_audio(elegido.ruta, self.frecuencia))

    def cargar(self, sonido: Sonido) -> Sonido:
        """Decodifica un sonido elegido por nombre, útil en los ejemplos iniciales."""
        if sonido.muestras is not None:
            return sonido
        return replace(sonido, muestras=decodificar_audio(sonido.ruta, self.frecuencia))


def decodificar_audio(ruta: Path, frecuencia: int = FRECUENCIA_MUESTREO) -> np.ndarray:
    """Usa ffmpeg para convertir un archivo a audio mono float32."""
    comando = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(ruta), "-vn", "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(frecuencia), "pipe:1",
    ]
    try:
        resultado = subprocess.run(
            comando, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise RuntimeError("No encuentro ffmpeg. Instálalo y vuelve a ejecutar.") from exc
    except subprocess.CalledProcessError as exc:
        detalle = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg no pudo leer {ruta}: {detalle}") from exc
    muestras = np.frombuffer(resultado.stdout, dtype="<f4").copy()
    if muestras.size == 0:
        raise ValueError(f"El archivo no contiene audio: {ruta}")
    return muestras


def parametros_por_movimiento(
    x: float, y: float, velocidad: float, personas_extra: int = 0,
    actividad_colectiva: float = 0.0,
) -> dict[str, float]:
    """Traduce la posición a controles audibles, fáciles de cambiar en clase."""
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    velocidad = max(0.0, min(1.0, float(velocidad)))
    personas_extra = max(0, int(personas_extra))
    actividad_colectiva = max(0.0, min(1.0, float(actividad_colectiva)))
    return {
        "pitch": 0.88 + 0.24 * x + 0.05 * velocidad,
        "corte_filtro": 700.0 + 10_000.0 * y,
        "saturacion": 1.0 + 3.0 * velocidad + 1.5 * actividad_colectiva,
        "reverb": min(
            0.75,
            0.08 + 0.35 * (1.0 - y) + 0.08 * personas_extra
            + 0.20 * actividad_colectiva,
        ),
    }


def reproducir_archivo(
    ruta: Path, modo: str = "stereo", dispositivo: int | None = None
) -> None:
    """Reproduce un archivo completo, útil para el segundo ejemplo."""
    from cuadrafonia import descripcion_salidas

    muestras = decodificar_audio(Path(ruta))
    canales = 2 if modo == "stereo" else 4
    if modo not in ("stereo", "cuadrafonico"):
        raise ValueError("El modo debe ser 'stereo' o 'cuadrafonico'.")
    try:
        sd.check_output_settings(
            device=dispositivo, channels=canales,
            samplerate=FRECUENCIA_MUESTREO, dtype="float32",
        )
    except Exception as exc:
        raise RuntimeError(
            f"El dispositivo no ofrece {canales} salidas a {FRECUENCIA_MUESTREO} Hz. "
            "Prueba --listar-dispositivos o usa --modo stereo."
        ) from exc
    ganancia = np.asarray(ganancias_espaciales(0.0, 0.0, modo), dtype=np.float32)
    sd.play(
        muestras[:, None] * ganancia[None, :],
        samplerate=FRECUENCIA_MUESTREO,
        device=dispositivo,
        blocking=True,
    )
    print(f"Reproducción finalizada ({descripcion_salidas(modo)}).")


class _Voz:
    def __init__(self, persona_id: int, sonido: Sonido) -> None:
        if sonido.muestras is None:
            raise ValueError("El sonido debe decodificarse antes de agregarlo al mezclador.")
        self.persona_id = persona_id
        self.sonido = sonido
        self.cursor = 0.0
        self.tiempo = 0.0
        self.x_persona = 0.0
        self.y_persona = 0.0
        self.pitch = 1.0
        self.corte_filtro = 12_000.0
        self.saturacion = 1.0
        self.reverb = 0.0
        self.estado_filtro = 0.0
        self.buffer_reverb = np.zeros(int(FRECUENCIA_MUESTREO * 0.16), dtype=np.float32)
        self.posicion_reverb = 0
        self.ganancias_anteriores: np.ndarray | None = None
        self.fade_restante = 0
        self.fade_total = 0


class Mezclador:
    """Mezcla varias voces y llena un flujo de audio estéreo o cuadrafónico."""

    def __init__(
        self,
        modo: str = "stereo",
        dispositivo: int | None = None,
        frecuencia: int = FRECUENCIA_MUESTREO,
        tamano_bloque: int = 256,
    ) -> None:
        if modo not in ("stereo", "cuadrafonico"):
            raise ValueError("El modo debe ser 'stereo' o 'cuadrafonico'.")
        self.modo = modo
        self.dispositivo = dispositivo
        self.frecuencia = frecuencia
        self.tamano_bloque = tamano_bloque
        self.canales = 2 if modo == "stereo" else 4
        self._voces: list[_Voz] = []
        self._finalizaciones: queue.SimpleQueue = queue.SimpleQueue()
        self._candado = threading.Lock()
        self._flujo = None

    def iniciar(self) -> None:
        if self._flujo is not None:
            return
        try:
            self._flujo = sd.OutputStream(
                device=self.dispositivo,
                samplerate=self.frecuencia,
                blocksize=self.tamano_bloque,
                channels=self.canales,
                dtype="float32",
                callback=self._procesar_bloque,
            )
            self._flujo.start()
        except Exception as exc:
            self._flujo = None
            raise RuntimeError(
                f"No se pudo abrir la salida {self.modo}. Revisa el dispositivo "
                "con --listar-dispositivos."
            ) from exc

    def detener(self) -> None:
        if self._flujo is not None:
            self._flujo.stop()
            self._flujo.close()
            self._flujo = None

    def agregar_voz(self, persona_id: int, sonido: Sonido) -> None:
        voz = _Voz(persona_id, sonido)
        with self._candado:
            self._voces.append(voz)

    def personas_con_voz(self) -> set[int]:
        with self._candado:
            return {voz.persona_id for voz in self._voces}

    def nombres_en_uso(self) -> set[str]:
        with self._candado:
            return {voz.sonido.nombre for voz in self._voces}

    def cantidad_de_voces(self) -> int:
        with self._candado:
            return len(self._voces)

    def actualizar_control(
        self, persona_id: int, x: float, y: float, velocidad: float,
        personas_extra: int = 0, actividad_colectiva: float = 0.0,
    ) -> None:
        controles = parametros_por_movimiento(
            x, y, velocidad, personas_extra, actividad_colectiva
        )
        with self._candado:
            for voz in self._voces:
                if voz.persona_id != persona_id:
                    continue
                voz.x_persona = 2.0 * max(0.0, min(1.0, x)) - 1.0
                voz.y_persona = 2.0 * max(0.0, min(1.0, y)) - 1.0
                voz.pitch = controles["pitch"]
                voz.corte_filtro = controles["corte_filtro"]
                voz.saturacion = controles["saturacion"]
                voz.reverb = controles["reverb"]

    def fundir_persona(self, persona_id: int, segundos: float = 0.25) -> None:
        """Baja el volumen de una voz al desaparecer su persona."""
        cantidad = max(1, int(segundos * self.frecuencia))
        with self._candado:
            for voz in self._voces:
                if voz.persona_id == persona_id and voz.fade_restante == 0:
                    voz.fade_restante = cantidad
                    voz.fade_total = cantidad

    def drenar_finalizaciones(self) -> list[dict[str, object]]:
        finalizaciones = []
        while True:
            try:
                finalizaciones.append(self._finalizaciones.get_nowait())
            except queue.Empty:
                return finalizaciones

    def _procesar_bloque(self, salida, cuadros, tiempo, estado) -> None:
        salida.fill(0.0)
        with self._candado:
            voces = list(self._voces)
        escala = 1.0 / math.sqrt(max(1, len(voces)))
        terminadas: list[tuple[_Voz, str]] = []
        for voz in voces:
            bloque, motivo = self._renderizar_voz(voz, cuadros)
            if bloque is not None:
                salida[: len(bloque), :] += bloque * escala
            if motivo is not None:
                terminadas.append((voz, motivo))
        if terminadas:
            with self._candado:
                for voz, motivo in terminadas:
                    if voz in self._voces:
                        self._voces.remove(voz)
                        self._finalizaciones.put({
                            "persona_id": voz.persona_id,
                            "sonido": voz.sonido.nombre,
                            "motivo": motivo,
                        })

    def _renderizar_voz(self, voz: _Voz, cuadros: int) -> tuple[np.ndarray | None, str | None]:
        muestras = voz.sonido.muestras
        if voz.cursor >= len(muestras):
            return None, "fin"
        cantidad_disponible = int(math.ceil((len(muestras) - voz.cursor) / voz.pitch))
        cantidad = min(cuadros, cantidad_disponible)
        motivo = "fin" if cantidad_disponible <= cuadros else None
        if voz.fade_restante > 0:
            cantidad = min(cantidad, voz.fade_restante)
            if cantidad == voz.fade_restante:
                motivo = "salida"

        posiciones = voz.cursor + np.arange(cantidad, dtype=np.float32) * voz.pitch
        indices = posiciones.astype(np.int64)
        fraccion = posiciones - indices
        siguientes = np.minimum(indices + 1, len(muestras) - 1)
        mono = muestras[indices] * (1.0 - fraccion) + muestras[siguientes] * fraccion
        voz.cursor += cantidad * voz.pitch

        corte = min(self.frecuencia * 0.45, max(80.0, voz.corte_filtro))
        alpha = 1.0 - math.exp(-2.0 * math.pi * corte / self.frecuencia)
        filtrado = np.empty(cantidad, dtype=np.float32)
        estado_filtro = voz.estado_filtro
        for indice, muestra in enumerate(mono):
            estado_filtro += alpha * (float(muestra) - estado_filtro)
            filtrado[indice] = estado_filtro
        voz.estado_filtro = estado_filtro

        drive = max(1.0, voz.saturacion)
        procesado = np.tanh(filtrado * drive) / max(1e-6, math.tanh(drive))
        posiciones_reverb = (
            voz.posicion_reverb + np.arange(cantidad)
        ) % len(voz.buffer_reverb)
        retrasado = voz.buffer_reverb[posiciones_reverb].copy()
        procesado = procesado * (1.0 - voz.reverb) + retrasado * voz.reverb
        voz.buffer_reverb[posiciones_reverb] = filtrado + retrasado * (0.30 * voz.reverb)
        voz.posicion_reverb = (voz.posicion_reverb + cantidad) % len(voz.buffer_reverb)

        if voz.fade_restante > 0:
            envolvente = np.maximum(
                0.0,
                (voz.fade_restante - np.arange(cantidad, dtype=np.float32))
                / max(1, voz.fade_total),
            )
            procesado *= envolvente
            voz.fade_restante -= cantidad
        else:
            entrada = np.minimum(
                1.0,
                (voz.cursor - cantidad * voz.pitch
                 + np.arange(cantidad, dtype=np.float32) * voz.pitch)
                / max(1.0, 0.01 * self.frecuencia),
            )
            procesado *= entrada

        x, y = posicion_de_movimiento(
            voz.sonido.movimiento, voz.tiempo, voz.x_persona, voz.y_persona
        )
        ganancias = np.asarray(ganancias_espaciales(x, y, self.modo), dtype=np.float32)
        anteriores = (
            ganancias if voz.ganancias_anteriores is None else voz.ganancias_anteriores
        )
        rampa = np.linspace(anteriores, ganancias, cantidad, endpoint=True, dtype=np.float32)
        voz.ganancias_anteriores = ganancias
        voz.tiempo += cantidad / self.frecuencia
        return procesado[:, None] * rampa, motivo
