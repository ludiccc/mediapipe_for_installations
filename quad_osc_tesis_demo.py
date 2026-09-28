#!/usr/bin/env python3
"""Variante de quad_osc_demo para tesis3-juli-rojas, sin cambiar Processing.

Recibe /kinect/x, /kinect/y y /kinect/velocidad (un número de 0 a 127)
en UDP 12000. Processing ya invierte X: no volver a invertirla aquí.
Se interpreta arriba de la imagen como frente y abajo como fondo de la sala;
Y es una coordenada de imagen, no una medida física de profundidad.

La primera pareja X/Y inicia una vez la secuencia: pájaros siguiendo la
posición y luego autos orbitando. /sequence/start permite reiniciarla.
La velocidad se recibe para monitoreo, sin modificar el audio. Si se pierde
el tracking, se mantiene la última posición y la secuencia continúa.
Las teclas C/S de Processing envían MIDI, no controlan este reproductor.

Dependencias: python -m pip install numpy sounddevice python-osc
También requiere ffmpeg en PATH y una interfaz con cuatro salidas.
Mantener este archivo junto a quad_osc_demo.py, cuyo motor de audio reutiliza.
Ejemplo con los audios incluidos (no necesariamente pájaros y autos):
    python quad_osc_tesis_demo.py --birds tesis3-juli-rojas/data/audio1.wav \
        --cars tesis3-juli-rojas/data/audio2.wav
"""

from __future__ import annotations

import argparse
import math
import queue
import subprocess
import threading
import time
from pathlib import Path

from quad_osc_demo import (
    BLOCK_SIZE, HERE, SAMPLE_RATE, QuadEngine, decode_mp3_mono, np, sd,
)
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


class KinectReceiver:
    """Adapta el centroide único de Processing al motor de paneo."""

    def __init__(self, events: queue.SimpleQueue):
        self.events = events
        self.position = [None, None]
        self.velocity = 0.0
        self.started = False

    def receive(self, address, *values):
        if len(values) != 1 or isinstance(values[0], (str, bytes, bool)):
            return
        try:
            value = float(values[0])
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(value):
            return
        value = max(0.0, min(127.0, value))
        if address == "/kinect/velocidad":
            self.velocity = value
            return
        if address not in ("/kinect/x", "/kinect/y"):
            return
        axis = 0 if address == "/kinect/x" else 1
        self.position[axis] = value * 2.0 / 127.0 - 1.0
        if any(coord is None for coord in self.position):
            return
        # El motor acepta N puntos: aquí hay un solo centroide, no cinco personas.
        coords = np.array([self.position], dtype=np.float32)
        self.events.put(("people", coords))
        if not self.started:
            self.start_sequence(address)

    def start_sequence(self, address, *values):
        self.started = True
        self.events.put(("start", None))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--birds", type=Path, default=HERE / "pajaros.mp3",
                        help="Primer audio, sigue la posición (MP3 o WAV).")
    parser.add_argument("--cars", type=Path, default=HERE / "autos.mp3",
                        help="Segundo audio, gira por los cuatro parlantes.")
    parser.add_argument("--device", type=int, default=None,
                        help="Índice del dispositivo de salida.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--port", type=int, default=12000)
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return
    for path in (args.birds, args.cars):
        if not path.is_file():
            parser.error(f"Falta el audio: {path}. Indicá --birds y --cars.")
    try:
        # ffmpeg también decodifica los WAV incluidos en el sketch.
        bird = decode_mp3_mono(args.birds)
        car = decode_mp3_mono(args.cars)
    except FileNotFoundError as exc:
        raise SystemExit("No encuentro ffmpeg. Instalalo y volvé a ejecutar.") from exc
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(f"No se pudo decodificar el audio: {exc}") from exc
    try:
        sd.check_output_settings(device=args.device, channels=4,
                                 samplerate=SAMPLE_RATE, dtype="float32")
    except Exception as exc:
        raise SystemExit("Se necesitan cuatro salidas a 48 kHz. Usá "
                         "--list-devices y seleccioná --device ÍNDICE.") from exc

    events = queue.SimpleQueue()
    receiver = KinectReceiver(events)
    dispatcher = Dispatcher()
    for address in ("/kinect/x", "/kinect/y", "/kinect/velocidad"):
        dispatcher.map(address, receiver.receive)
    dispatcher.map("/sequence/start", receiver.start_sequence)
    try:
        server = BlockingOSCUDPServer(("127.0.0.1", args.port), dispatcher)
    except OSError as exc:
        raise SystemExit(f"No se pudo abrir UDP {args.port}: {exc}") from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    engine = QuadEngine(bird, car, events)
    print(f"Escuchando OSC en 127.0.0.1:{args.port}. Esperando X e Y de Kinect.")
    print("Salidas: 1=frente izquierda, 2=frente derecha, "
          "3=fondo derecha, 4=fondo izquierda.")
    print("Secuencia única al detectar posición; /sequence/start para repetir. Ctrl+C para salir.")
    try:
        with sd.OutputStream(device=args.device, samplerate=SAMPLE_RATE,
                             blocksize=BLOCK_SIZE, channels=4, dtype="float32",
                             callback=engine.callback):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
